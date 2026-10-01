# 在同一套集群上跑 Qwen3-30B-A3B

状态:**2026-10-01 真机跑通**,15 台全部在线、0–48 层全覆盖、客户端能生成中文。

`qwen3_moe` 适配器和 Qwen3.6 那个(`qwen3_5_moe`)是两套独立的适配器,
共用同一批脚本、同一套 systemd 单元、同一个控制节点流程。DHT 前缀带
`-petals-qwen3-moe-v1` 后缀,两个 swarm 天然隔离。

## 这个模型和 Qwen3.6 有什么不同

| | Qwen3.6-35B-A3B | Qwen3-30B-A3B |
|---|---|---|
| `model_type` | `qwen3_5_moe` | `qwen3_moe` |
| 层数 | 40 | **48** |
| 注意力 | 3:1 混合(linear + full) | **全部 full attention** |
| 专家 | 256 选 8 + shared expert | 128 选 8,**无 shared expert** |
| head_dim | 256 | 128(**不等于** hidden 2048 ÷ 32 头) |
| 权重/层 | ~1.59 GiB | **~1.17 GiB** |
| checkpoint 专家布局 | 打包 | **每个专家一个张量**(每层 384 个) |

三点值得注意:

**没有 linear attention,缓存就便宜得多。** Qwen3.6 每个会话要先付 12.98 MB 的
conv/recurrent 固定开销(每个 linear 层一份,和长度无关),这个模型一分都不用付。
8 层 span 上一个会话只要 `2048 字节 × 层数 × (prompt + 生成)`,于是:

```
并发会话数 = ATTN_CACHE_TOKENS ÷ (prompt + 生成)
```

按默认的 65536:2048+2048 的会话能并发 16 个,1024+1024 能 32 个,128+128 能 256 个。
真机上 `status` 报的 `cache_tokens_left` 可以直接验算:8 层 span 是 1048576,
7 层是 917504,和 `2048 × 65536 × 层数 ÷ 1024` 一字不差。会话回退也不用重放,
K/V 切一刀就行。

**head_dim 是 128,不是 hidden_size ÷ 头数(那是 64)。** Petals 通用的缓存路径按后者算,
会只分配一半。所以这个适配器和 Qwen3.6 一样走 `petals_custom_cache`,用自己的
`cache_specs`。`tests/test_qwen3_moe.py::test_cache_is_sized_from_head_dim_not_hidden_size`
盯着这一条。

**专家在 checkpoint 里是分开存的。** Hub 上是
`model.layers.N.mlp.experts.{0..127}.{gate,up,down}_proj.weight`,每层 384 个张量。
Petals 按参数名精确匹配加载,没有转换步骤,所以适配器保留了每个专家一个 `nn.Linear`,
没有用新版 Transformers 的打包写法。改成打包的话,每一个名字都会对不上分片索引,
一个 block 都加载不了。

---

## 0. 控制节点环境:`~/petals/env.sh`

**这是整个流程里最容易出事的一步,首次部署时一半的时间耗在它身上。**

```bash
cat > ~/petals/env.sh <<'EOF'
cd ~/petals
export NODE_PY=/home/ubuntu/anaconda3/envs/moe/bin/python   # 各节点已有的 conda 环境
export HF_HUB_DISABLE_XET=1        # 必须:集群到不了 xethub
export MAX_DISK_SPACE=30GB
export PROXY_NODE=N08
export PROXY_SKIP="N06 N07"        # 这两台自己有出口
export CLEANUP_ON_START=gpu
export MODEL_NAME=Qwen/Qwen3-30B-A3B
EOF
```

三条规矩:

1. **每开一个新的 ssh 会话都要 `source ~/petals/env.sh`。** 变量只活在那一个 shell 里。
2. **`ssh host && source ...` 是错的。** `ssh` 是交互式的,`&&` 后面要等你退出 ssh 才执行,
   而且是在本地执行。先 ssh 进去,再单独 source。
3. **`bash env.sh` 也是错的**,那是子进程,`export` 的东西跟着子进程一起消失。
   必须 `source`(或 `.`)。

每次动手前先确认:

```bash
echo "XET=[$HF_HUB_DISABLE_XET] PROXY=[$PROXY_NODE] NODE_PY=[$NODE_PY] MODEL=[$MODEL_NAME]"
```

四个方括号都得有内容。少一个都会以一种**不报错**的方式失败,见下面的故障对照。

---

## 1. 从 Qwen3.6 切过来

```bash
source ~/petals/env.sh

bash examples/qwen_cluster.sh service stop          # 必须用 service stop
bash examples/qwen_cluster.sh service status        # 15 台 inactive
bash examples/qwen_cluster.sh purge Qwen/Qwen3.6-35B-A3B        # 先空跑
bash examples/qwen_cluster.sh purge Qwen/Qwen3.6-35B-A3B --yes
```

**`stop` 不行,必须 `service stop`。** 单元是 `Restart=always` / `RestartSec=20`,
`stop` 只杀进程,20 秒后 systemd 把它拉回来,purge 会把每台都当 RUNNING 跳过。

**purge 的模型名永远写全。** 不写就取 `$MODEL_NAME` —— 你要是已经切到新模型了,
不带参数的 `purge` 删的是新模型。

bootstrap DHT 不用动,它和模型无关。

## 2. 先拿一台试水

**这一步省不得。** 首次部署时它在十分钟内暴露了 Xet 的问题;不然要等 15 台各下完 15 GB
才发现,那是几个小时。

```bash
cd ~/petals
awk '$1=="N01" {print $1"  "$2"  blocks=0:4"}' task/hosts.txt > task/smoke.txt
cat task/smoke.txt        # 确认有一行,BLOCKS 是 0:4

HOSTS_FILE=task/smoke.txt bash examples/qwen_cluster.sh deploy
HOSTS_FILE=task/smoke.txt bash examples/qwen_cluster.sh start
HOSTS_FILE=task/smoke.txt bash examples/qwen_cluster.sh status --watch
```

**用单机 hosts 文件之前,先确认代理地址已经登记。** `ensure_proxy_addr()` 要在
`HOSTS_FILE` 里找到 `PROXY_NODE` 才能恢复地址,smoke.txt 里只有 N01,它找不到 N08
就**一声不吭地不注入代理**,于是 N01 直连 CDN 然后超时。先用完整 hosts.txt 跑一次:

```bash
bash examples/qwen_cluster.sh proxy status     # 必须用完整 task/hosts.txt
bash examples/qwen_cluster.sh proxy start      # 白名单是从 HOSTS_FILE 的 IP 生成的
cat .qwen-cluster/proxy_addr                   # http://192.168.2.4:8899
```

试水要看到的是:N01 变 ONLINE、认领 0–4 层、`cache` 涨到 7.5 GB 左右
(4 层权重约 5 GB,但下的是整个分片,跨 2 个分片)。这证明配置被
`AutoDistributedConfig` 认了出来、48 层的 block 参数名和分片索引对得上、
服务端能起来并向 DHT 报到。

**启动后立刻验证环境真的传进去了**,别只看脚本的输出:

```bash
ssh ubuntu@192.168.1.2 'tr "\0" "\n" < /proc/$(cat ~/petals-qwen/run/server.pid)/environ | grep -E "XET|PROXY|MODEL_NAME"'
```

要四行:`MODEL_NAME`、`HF_HUB_DISABLE_XET`、`HTTPS_PROXY`、`NO_PROXY`。少一行就停下来查。

## 3. 全量铺开

```bash
source ~/petals/env.sh

bash examples/qwen_cluster.sh deploy
bash examples/qwen_cluster.sh preflight          # 服务停着,量到的才是真实空闲显存
bash examples/qwen_cluster.sh plan --cap 8       # 先看
bash examples/qwen_cluster.sh plan --cap 8 --write
bash examples/qwen_cluster.sh service install    # 每行确认 blocks=8 cache=65536/4096
bash examples/qwen_cluster.sh service start
bash examples/qwen_cluster.sh status --watch
```

`plan` 第一行会打 `Qwen/Qwen3-30B-A3B: 48 layers, 1.30 GiB per block
(weights 1.17 + cache at 65536 tokens)` —— 确认这行对了再看下面的表。
24 GB 的卡能放 15 层、T4 能放 9 层,`--cap 8` 之后 15 台 × 8 层 = 120 个层位盖 48 层,
**2.5 份副本**。

**`service install` 不能跳。** env 文件里写死了 `MODEL_NAME` 和 `NUM_BLOCKS`,
不重写的话 systemd 拉起来的还是上一个模型、上一套层数。
每行打出的 `blocks=` 和 `cache=` 就是实际写进去的值,看一眼。

**别把 `start`/`stop` 和 `service` 混用。** 单元装好之后统一走 `service`;
`stop` 杀掉的进程 systemd 20 秒后会拉回来,再 `start` 就会出现两个服务端抢同一个端口,
表现为客户端 `peer id mismatch`。真混用了就 `cleanup --stale` 看一眼。

## 4. 验证

```bash
bash examples/qwen_cluster.sh status             # 等 "Every layer is online"
bash examples/qwen_cluster.sh client --prompt '你好'
bash examples/qwen_cluster.sh bench --concurrency 1 4 8 16 --new-tokens 128 --timeout 600
```

**重启之后要等。** 每台要把 8 层(约 9.3 GB)装进显存、跑一遍吞吐测量,才向 DHT 报到,
要几分钟。重启后马上跑客户端会得到 `MissingBlocksError`,那不是故障。

真机实测的参考值:

- `inference_rps` 276–342(Qwen3.6 是 210–256)
- `cache_tokens_left` 8 层 1048576 / 7 层 917504
- 每台稳态缓存 22–27 GB。注意 `plan` 里那个分片上界(8 层约 15.3 GB)算的是
  **固定 span**;rebalancer 会让一台先后服务不同层段,缓存会累积到接近
  `MAX_DISK_SPACE` 才被 LRU 压住。30GB 是合适的,不是富余的。
- 客户端侧(embedding + 最终 norm + lm_head)要下 2 个分片约 7.6 GB,
  首次约 67 秒,之后从缓存加载 4 秒

---

## 5. 故障对照:部署时实际踩过的

### `ConnectionError ... cas-server.xethub.hf.co`

集群到不了 Xet 存储后端。意味着 `HF_HUB_DISABLE_XET` 没进到服务端进程里。

确认:`/proc/<pid>/environ` 那条命令。修:`source ~/petals/env.sh` 之后
`start --restart`(`--restart` 不能省,`start` 会跳过已在跑的服务端)。

### `ReadTimeout ... us.aws.cdn.hf.co`

Xet 关掉了,但普通 CDN 这条路也不通 —— 没有 `HTTPS_PROXY`。

常见原因是用了只含一台的 `HOSTS_FILE`(见第 2 节),或者
`.qwen-cluster/proxy_addr` 丢了。`proxy start` 用完整 hosts.txt 跑一次即可恢复。

### tokenizer: `data did not match any variant of untagged enum ModelWrapper`

Qwen3 的 `tokenizer.json` 是新版 `tokenizers` 生成的,而仓库钉的 Transformers 4.43.1
要求 `tokenizers>=0.19,<0.20`,老版 Rust 反序列化器读不懂。

Hub 上同时提供 `vocab.json` 和 `merges.txt`,所以慢速(纯 Python)tokenizer 能绕开。
`examples/qwen_generate.py` 和 `bench_qwen.py` 里的 `load_tokenizer()` 已经做了回落,
会打一行 `fast tokenizer unavailable (...), using the slow one`。编码一个 prompt
是毫秒级,相对每 token 几百毫秒的解码可以忽略。

### `ValidationError: local time must be within 3 seconds of others`

hivemind 的 `MAX_DHT_TIME_DISCREPANCY_SECONDS = 3`。**这一条会连带解释"某些节点缓存
永远停在 12.2KB 不下载"** —— `Server.__init__` 是先建 DHT、后下权重的,时钟超差的主机
在握手就被拒,根本走不到下载那一步。

**不要只看 `chronyc tracking` 的 `System time: ... slow of NTP time`。** 那句话只说明
"我和我的源一致",源本身可能早就联系不上了。首次部署时 N01 和 N06 的 `Ref time` 停在 **44 天前**,chrony 还在自信地报 0.0001 秒偏差,实际已经漂了 4 秒。

三条一起看才算数:

```bash
# 1. 守护进程最后一次真正同步是什么时候(Ref time 这一行)
for i in 1.2 1.3 1.6 1.5 1.4 2.2 2.3 2.4 3.4 3.2 3.3 3.5 4.2 4.3 4.4; do
  printf '%-14s ' 192.168.$i; ssh ubuntu@192.168.$i 'chronyc tracking | sed -n "3p"'
done

# 2. 主机之间的墙上时钟互相对不对得上(串行 ssh 每次约 0.4 秒,据此扣除)
for i in 1.2 2.4 2.2; do printf '%s ' 192.168.$i; ssh ubuntu@192.168.$i 'date +%s.%N'; done
```

`Ref time` 是陈的就重建同步:

```bash
ssh ubuntu@<ip> 'chronyc sources -v | tail -6'      # 源是不是都不可达
ssh ubuntu@<ip> 'sudo systemctl restart chrony && sleep 15 && sudo chronyc makestep'
```

改完**用墙上时钟复验**,不要信 chrony 的自述。bootstrap 节点(N01)对齐之后
最好重启一次 DHT,让它用新时间。

> `synctime` 子命令目前只读各主机 NTP 守护进程的自述,**看不出"两台各自同步到了
> 不同的源"这种情况**,会打出一片 `±0.000s`。修这个之前,以上面两条命令为准。

### `MissingBlocksError: No servers holding blocks [...]`

紧接着重启跑的话,就是太早了,等 `Every layer is online`。

如果等到全覆盖还报,再看缺的是哪些层 —— 缺一整段说明那几台没起来,
`diag` 看原因。

### `status` 顶上的 DHT 前缀是旧模型

```
Using DHT prefix: Qwen3-6-35B-A3B-petals-qwen-v1
```

这个 shell 里 `MODEL_NAME` 没设,回落到脚本默认值了,你查的是**另一个 swarm**,
下面的层覆盖信息全部无效。`source ~/petals/env.sh` 重跑。

认准这行:对的应该是 `Qwen3-30B-A3B-petals-qwen3-moe-v1` 和 `48 layers`。

---

## 不支持的

配置里出现这些会在启动时直接报错,而不是跑出错结果:

- `rope_scaling` / `rope_parameters` 里非 `default` 的 RoPE(YaRN 等长上下文扩展)
- 滑动窗口注意力(`use_sliding_window`)
- `mlp_only_layers` 非空,或 `decoder_sparse_step != 1` —— 这种 checkpoint 有两种块大小,
  Petals 会按先看到的那种给整个 swarm 排层,宁可拒绝也不要排错
- 预量化的 checkpoint、prompt tuning、LoRA、beam search、投机解码

`Qwen3-235B-A22B` 的结构参数同属这一类(`qwen3_moe`,94 层),配置检查会放行,
但每层更大、层数更多,得重新 `plan`。

## 验证到哪一步了

离线:

- 逐层数值对拍:和未经修改的 Transformers v4.51.0 `Qwen3MoeDecoderLayer` 比,
  长度 7 和 67 两组**逐位相同**(fixture 在 `tests/data/`,用
  `tests/make_qwen3_moe_reference.py` 重新生成)
- 另外和 Transformers 5.17 的打包实现独立对过一次,同样逐位相同
- 增量解码、会话回退、RoPE 表增长:和一次性 forward 相差 2.4e-7 以内
- 参数名与 Hub 分片索引的键集合逐一比对
- 缓存记账、配置注册与拒绝路径:`tests/test_qwen3_moe.py` 共 27 个用例全过

真机(2026-10-01,15 节点):

- 48 层全覆盖,15 台 ONLINE,客户端生成中文正常
- `cache_tokens_left` 在 8 层和 7 层两种 span 上都与离线公式逐位吻合

还没做的:**吞吐基准**。`bench` 还没在这个模型上跑出过基线。
注意 Petals **不跨会话做 batching**(`task_pool.py:35` 的注释),所以压测压出来的是
每步固定开销,不是 GPU 算力上限 —— 单客户端进程还会先撞上本机 CPU 上 fp32 的
lm_head。要测集群上限得从多个节点同时打。
