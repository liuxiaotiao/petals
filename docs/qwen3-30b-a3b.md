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

## 0. 控制节点环境:`~/petals-env.sh`

**这是整个流程里最容易出事的一步,首次部署时一半的时间耗在它身上。**

```bash
cat > ~/petals-env.sh <<'EOF'
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

**放在仓库外面。** 它最早在 `~/petals/env.sh`。10/01 把 Mac 上的仓库整个 rsync 到控制节点时,
`--delete` 把这个不在版本库里的文件一起删了,之后每条命令都静悄悄地回落到脚本默认的
Qwen3.6 —— 见第 5 节"`service install` 把旧模型写进了 unit"。

三条规矩:

1. **每开一个新的 ssh 会话都要 `source ~/petals-env.sh`。** 变量只活在那一个 shell 里。
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
source ~/petals-env.sh

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
source ~/petals-env.sh

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

确认:`/proc/<pid>/environ` 那条命令。修:`source ~/petals-env.sh` 之后
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

> `synctime` 现在会检查每台 `Ref time` 的年龄:超过 `NTP_REF_MAX_AGE`(默认 1 小时)
> 就不再相信守护进程的自述,改用 SSH 往返对时,并把这些主机单独点名为
> `Not actually disciplined`。以它的结论为准即可,上面两条命令留作手工复核。

### 15 台只剩 7 台在服务,`status` 却说 "The swarm is usable"

10/03 才发现的,但 10/01 下午就开始了。之后两天所有压测跑的都是 7 台。

**症状。** `status` 汇总行是 `6 server(s) online`(不是 15),主机表里 8 台 `DOWN`;
`service status` 里这 8 台是 `none ... restarts=5`。而汇总的最后一句仍然是
`Every layer is online ... The swarm is usable` —— 它只检查层有没有覆盖,不检查少了几台。
**以后看 `status` 先数 `N server(s) online`。**

**因果链。**

1. 这 8 台(N03 N04 N05 N07 N09 N12 N14 N15)的 chrony 只配了公网池,而它们没有 UDP 出网
   (NTP 是 UDP 123,HTTP 代理带不了),8 月 18 日之后一次都没同步过,以约 80 ms/天
   的速度一起往前漂
2. 10/01 下午某次重启时漂移已超过 3 秒,`Server.__init__` 加入 DHT 就撞 `ValidationError`
3. unit 里 `StartLimitIntervalSec=900`、`StartLimitBurst=5`:15 分钟内失败 5 次,
   systemd 就放弃、不再重试。这个上限本身是对的(防止无限重启),代价是一次暂时性故障
   变成了永久停机
4. 10/02 用 `synctime --yes` 对过表,但那时已经没有任何东西在重试,所以没有救回来

**修复,三步:**

```bash
# 1. 让 N01 对内授时(它自己能连上公网 NTP;上游断了也继续服务)
ssh ubuntu@192.168.1.2 sudo bash -s <<'X'
conf=/etc/chrony/chrony.conf
grep -q '^allow 192.168.0.0/16' "$conf" || printf '\nallow 192.168.0.0/16\nlocal stratum 10\n' >> "$conf"
systemctl restart chrony
X

# 2. 没有出网的主机都指向 N01(幂等,重跑无害)
for n in N03 N04 N05 N07 N09 N12 N14 N15; do
  ip=$(awk -v id=$n '$1==id{split($2,a,":");print a[1]}' task/hosts.txt)
  ssh ubuntu@$ip sudo bash -s <<'X'
conf=/etc/chrony/chrony.conf
grep -q '^server 192.168.1.2 ' "$conf" || printf '\nserver 192.168.1.2 iburst prefer minpoll 4 maxpoll 6\n' >> "$conf"
systemctl restart chrony
X
done
bash examples/qwen_cluster.sh synctime     # 15 台都应是 its own NTP daemon

# 3. 只重启死掉的那几台(service restart 自带 reset-failed,能清掉 StartLimit 状态)
grep -E '^(N03|N04|N05|N07|N09|N12|N14|N15) ' task/hosts.txt > task/hosts.revive
HOSTS_FILE=task/hosts.revive bash examples/qwen_cluster.sh service restart
```

`chronyc -n sources` 里 N01 那行开头是 `^?` 不一定是够不着 —— 刚重启 10 秒内样本不够。
看 `Reach` 列:不是 0 就是通的。最终确认看 N01 上的 `sudo chronyc -n clients`,
应该列出这几台。

### `MissingBlocksError: No servers holding blocks [...]`

紧接着重启跑的话,就是太早了,等 `Every layer is online`。

如果等到全覆盖还报,再看缺的是哪些层 —— 缺一整段说明那几台没起来,
`diag` 看原因。

### `status` 顶上的 DHT 前缀是旧模型

```
Using DHT prefix: Qwen3-6-35B-A3B-petals-qwen-v1
```

这个 shell 里 `MODEL_NAME` 没设,回落到脚本默认值了,你查的是**另一个 swarm**,
下面的层覆盖信息全部无效。`source ~/petals-env.sh` 重跑。

认准这行:对的应该是 `Qwen3-30B-A3B-petals-qwen3-moe-v1` 和 `48 layers`。

### `service install` 把旧模型写进了 unit

```
MODEL_NAME is not set, so this run uses the built-in default Qwen/Qwen3.6-35B-A3B.
Environment: MODEL_NAME=Qwen/Qwen3.6-35B-A3B ...
```

`install` 只警告、不拒绝,于是**把错的模型名写进了持久化的环境文件**,随后的 `restart`
让那台去服务一个不存在的 swarm。`install` 输出里的 `Environment:` 那行每次都要看一眼。
修:`source ~/petals-env.sh` 之后对那几台重跑 `install` + `restart`。

### `service install` 写出 `proxy=no`

```
No proxy registered: the units will be written WITHOUT HTTPS_PROXY.
```

代理进程可能还在 N08 上正常跑(`proxy status` 显示 `running, N tunnel(s)`),只是控制节点
`.qwen-cluster/proxy_addr` 里的登记丢了。`proxy start` 是幂等的:进程在跑就只补登记,
不会重启它。补完再 `install`。

N06、N07(在 `PROXY_SKIP` 里,有自己的出口)和 N08(代理本身)显示 `proxy=no` 是对的。

### 新节点上跑 `bench`,卡在 `ReadTimeout ... us.aws.cdn.hf.co`

客户端要的 embedding / lm_head 分片在 `~/.cache/huggingface`(不是服务端的
`~/petals-qwen/cache`),新节点上是冷的,要现下约 5 GB。几个节点同时冷启动、又都挤 N08
的代理,会一直超时。从已经有缓存的节点(N08)走局域网播种,**只拷这个模型的目录**:

```bash
src=$(awk '$1=="N08"{split($2,a,":");print a[1]}' task/hosts.txt)
rsync -az ubuntu@$src:.cache/huggingface/hub/models--Qwen--Qwen3-30B-A3B/ /tmp/seed30b/
rsync -az /tmp/seed30b/ ubuntu@<目标IP>:.cache/huggingface/hub/models--Qwen--Qwen3-30B-A3B/
```

整个 `~/.cache/huggingface` 一起拷会把别的模型也带过去 —— 这样干过一次,三台磁盘直接写满。

### `bench` 打完表格不退出

`os._exit` 跳过了 multiprocessing 的清理,hivemind 的 DHT 子进程活了下来,还握着继承的
stdout,ssh 要等这根管道关掉才返回。`42cb9e8` 之后的 bench 会先结束子进程再退出;
节点上还是旧版本就 `deploy` 一次。症状是多节点并发压测末尾的 `wait` 永远不返回。

---

## 6. 磁盘:换框架或换模型之后的清理

2026-10-02 这 15 台的根分区只剩 1.7–3.4 GB,一次 rsync 就 `No space left on device`,
而 Petals 自己占的是该占的那份。大头全在没人认领的缓存里:

| 占用 | 每台 | 性质 |
|---|---|---|
| `~/.cache/pip` | 13 GB | 下载缓存,删了无害 |
| `Anaconda3-*.sh` | 659 MB | 装完没删的安装包 |
| `~/.cache/huggingface` | 4.8–11 GB | **客户端**拉的模型分片 |
| `~/petals-qwen/cache` | 29–32 GB | server 的块权重,归 `purge` 管 |

客户端缓存是个盲区:`cmd_client` 不设 `CACHE_DIR`,它落在默认的
`~/.cache/huggingface`,于是既不计入 `MAX_DISK_SPACE`,`status` 看不到,
`purge` 也清不掉。N03 就这样一直留着 5.9 GB 的 Qwen3.6 客户端缓存,
换模型两周后才被发现。

`examples/cluster_disk.sh` 专门管这一类。它不依赖 Petals,换别的框架压测时照用:

```bash
examples/cluster_disk.sh survey                       # 只看不删,全部目标
examples/cluster_disk.sh clean pip installers         # 空跑
examples/cluster_disk.sh clean pip installers --yes
KEEP="Qwen/Qwen3-30B-A3B" examples/cluster_disk.sh clean hf --yes
examples/cluster_disk.sh survey --host N03            # 单台
```

目标:`pip` `conda` `torch` `hf` `hfdata` `installers`,全是缓存或安装包,
删掉的代价只是重新下载,不是状态。和 `purge` 同样的形状:默认空跑,
`--yes` 才动手,一台连不上不影响其余的,结尾单独列出没清到的主机。

**它碰不到 `~/petals-qwen/cache`。** 那是 server 的块权重,必须走 `purge`,
而 `purge` 知道在 server 还活着的时候拒绝执行。

那次清理一共腾出约 190 GB,15 台全部回到 17 GB 以上。

---

## 7. 吞吐与并发:实测

2026-10-02 至 10-04 在这 15 台上测的。先说结论,再说怎么测、数从哪来。

### 结论

- **容量 = 完整链数 × 每链约 32 tok/s。** 两条链同时压满,总吞吐恰好等于各自单独压满之和
  (65.04 vs 64.98),链与链之间没有共享瓶颈。
- **每链的上限来自每跳约 31 ms 的串行服务时间,其中 GPU 实际计算只占约 3 ms。**
  其余是 RPC、序列化、调度。满负载时 GPU 大约九成时间空闲 —— 这是 Petals 的软件开销上限,
  不是硬件上限。
- **`status` 里的 `inference_rps`(~300)不能用来做容量规划。** 实际可用的串行速率约 32 步/秒,
  差将近 10 倍。
- **网关(客户端)不要放在承担层的节点上。** 同样的在途会话数,入口放在 layer 0 节点比
  放在闲置机器上低约 20%,比完全分散低约 25%。
- **不钉死布局,多出来的机器几乎不涨吞吐。** 15 台自动布局只测到 34–35 tok/s,和 7 台一样。
  原因见下面"为什么是这个数"。

### 推荐布局:钉死成两条完整链

48 层按 8 层一段切成 6 段,12 张 24GB 卡每段 2 台,正好两条互不重叠的链:

| | 0:8 | 8:16 | 16:24 | 24:32 | 32:40 | 40:48 |
|---|---|---|---|---|---|---|
| **A** | N01 | N05 | N03 | N07 | N02 | N06 |
| **B** | N11 | N08 | N09 | N13 | N04 | N10 |

3 台 T4(N12 N14 N15)只装得下 7 层,不进任何链,`blocks=7` 交给均衡器,压测时当网关用。

`task/hosts.txt` 里写范围就是钉死(`blocks=0:8`),写数字是"带几层、位置交给 swarm":

```bash
pin() { sed -i -E "s/^($1 .*)blocks=[0-9:]+/\1blocks=$2/" task/hosts.txt; }
pin N01 0:8;   pin N11 0:8;   pin N05 8:16;  pin N08 8:16
pin N03 16:24; pin N09 16:24; pin N07 24:32; pin N13 24:32
pin N02 32:40; pin N04 32:40; pin N06 40:48; pin N10 40:48
```

**分两波重启**,每段始终留一台在线;**要换段的节点先从同段的节点播种分片**,不然它会经代理
重下约 13 GB(`seed_span` 的定义见下)。

```bash
grep -E '^(N01|N08|N06|N13|N03|N02) ' task/hosts.txt > task/hosts.wave1
grep -E '^(N11|N05|N10|N07|N09|N04) ' task/hosts.txt > task/hosts.wave2
for w in wave1 wave2; do
  HOSTS_FILE=task/hosts.$w bash examples/qwen_cluster.sh service install    # 看 Environment 行和 proxy=
  HOSTS_FILE=task/hosts.$w bash examples/qwen_cluster.sh service restart
  # 等到 status 显示 15 server(s) online, 0 still joining 再下一波
done
```

三个坑,都踩过:

- **只钉一部分,剩下的会连锁挪位。** 钉走 N05 之后,均衡器把 N13 从 24:32 挪到了 32:40,
  一台 T4 挪到 22:29,结果 29–31 层只剩一台。要钉就把 12 台一起钉
- **T4 的 7 层段会切出单点。** 自动布局里一台 T4 占了 `0:7`,第 7 层只剩 N01 一台,
  所有会话都得经过它
- **均衡器不会自己修好上面这种情况。** 每台只问"我自己挪是不是更好",是就挪,不是就不动
  (`if local_span.start == new_start: return False`)。15 台各自都在局部最优上,
  没有哪一台单独挪一下能补上那一层

播种函数:从源节点按 `model.safetensors.index.json` 挑出含指定层的分片,经控制节点的管道
直接流到目标节点,控制节点不落盘:

```bash
seed_span() {  # seed_span <源节点> <目标节点> <起始层> <结束层>
  local s d
  s=$(awk -v id=$1 '$1==id{split($2,a,":");print a[1]}' task/hosts.txt)
  d=$(awk -v id=$2 '$1==id{split($2,a,":");print a[1]}' task/hosts.txt)
  ssh ubuntu@$s "LO=$3 HI=$4 bash -s" <<'X' | ssh ubuntu@$d "cd ~/petals-qwen/cache/models--Qwen--Qwen3-30B-A3B && tar xf - --skip-old-files && echo received"
cd ~/petals-qwen/cache/models--Qwen--Qwen3-30B-A3B
snap=$(ls -d snapshots/*/ | head -1)
~/petals-qwen/venv/bin/python - "$snap" > /tmp/need.$$ <<'PY'
import json, os, sys
snap, lo, hi = sys.argv[1], int(os.environ["LO"]), int(os.environ["HI"])
wm = json.load(open(os.path.join(snap, "model.safetensors.index.json")))["weight_map"]
for f in sorted({f for k, f in wm.items() if any(k.startswith(f"model.layers.{i}.") for i in range(lo, hi))}):
    link = os.path.join(snap, f)
    print(link)
    print(os.path.normpath(os.path.join(snap, os.readlink(link))))
PY
echo "shipping $(grep -c safetensors /tmp/need.$$) shards, $(xargs du -chL < /tmp/need.$$ | tail -1 | cut -f1)" >&2
tar cf - -T /tmp/need.$$
rm -f /tmp/need.$$
X
}
seed_span N07 N13 24 32
```

8 层约 3–4 个分片、12–15 GB。不加 `-z`:权重不可压缩,压缩只会把 CPU 变成瓶颈。

### 怎么测

**工具。**

| | 作用 |
|---|---|
| `qwen_cluster.sh peers` | 每台当前的完整 peer id 和层段。`status` 只显示 id 末 6 位,不够用。每次重启 id 都会变,所以重启后要重新取 |
| `bench --allowed-servers <peer…>` | 这个客户端的所有会话只走列出的服务端,即钉在一条链上 |
| `bench --duration 900 --ramp 120` | 闭环:`--concurrency` 个 worker,每个会话结束立刻开下一个;只统计在窗口内结束的会话。最后一行 `RESULT key=value …` 便于多进程加总 |
| `PETALS_INFERENCE_ROUTING=max_throughput` | 推理改用随机挑副本(Petals 训练时用的策略)。**未验证**,见下 |

拼链和跑法:

```bash
bash examples/qwen_cluster.sh peers | tee /tmp/peers.txt
chain() { awk -v want=" $* " 'index(want, " "$1" ") {print $2}' /tmp/peers.txt | tr '\n' ' '; }
A=$(chain N01 N05 N03 N07 N02 N06)
B=$(chain N11 N08 N09 N13 N04 N10)
NODES=$(sed 's/#.*//' task/hosts.txt | awk 'NF {print $1}')

# 批量:15 台各起一个客户端,每个 2 个会话,全部钉在 A 链
for n in $NODES; do
  bash examples/qwen_cluster.sh bench --node $n --prompt-tokens 128 --new-tokens 256 \
       --concurrency 2 --timeout 1200 --allowed-servers $A > /tmp/b.A.$n.log 2>&1 &
done; wait
```

**汇总时的三个坑,每一个都让结论错过一次:**

1. **`tok/s/sess`(第 4 列)是每个会话的速度。** 一个客户端跑 `--concurrency c` 时,
   它对总吞吐的贡献是 `c × 第 4 列`,不是第 4 列本身
2. **日志里有 `\r`。** 进度提示原地刷新,重定向到文件后和数据行挤在同一行,先 `tr '\r' '\n'`
3. **第 5 列 `tok/s total` 把 TTFT 也算进了墙钟**,和上面的口径不同,不要混用

```bash
for f in /tmp/b.A.*.log; do tr '\r' '\n' < "$f" | grep -E '^ +2 +[0-9]+ +[0-9]+ +[0-9.]+ +[0-9.]+'; done \
  | awk '{n++; s += $4} END {printf "%d clients | %.2f tok/s\n", n, 2*s}'
# 闭环模式直接加 RESULT 行:
cat /tmp/cl.*.log | tr '\r' '\n' | grep '^RESULT' \
  | awk '{for (i=2;i<=NF;i++) {split($i,kv,"="); v[kv[1]]+=kv[2]}} END {print v["tps"], "tok/s"}'
```

**每台到底接了多少会话**,看服务端日志里的 `rpc_inference.close` 条数(批量模式下 bench 的
每个会话会调两次 `generate()`,所以一个会话在每台上记两条)。这是判断路由有没有分流的
唯一可靠办法 —— 之前就是靠它发现 8 台已经离线的。

### 结果

**单会话**(钉死布局,客户端在各自链的 layer 0 节点上):

| | 单独跑 | 两条链同时跑 |
|---|---|---|
| A 链(全 A30) | 279 ms/token | 301 |
| B 链(A30 + RTX 6000) | 328 | 325 |

**单链饱和曲线**(10/02 测于 7 台在线、实际只有一条链的状态,5 个客户端):

| 在途会话 | 1 | 3 | 5 | 10 | 20 | 40 |
|---|---|---|---|---|---|---|
| 总吞吐 tok/s | 3.04 | 8.13 | 12.52 | 24.14 | 30.84 | 31.60 |
| ms/token | 329 | ~371 | ~400 | 414 | 648 | 1266 |

20 个会话就拿到约 95% 的吞吐,再往上只增加延迟。饱和区里 `ms/token ≈ 31 ms × 会话数`,
截距几乎为零,纯排队。同一时期 15 个客户端 × 3 = 45 个会话测得 32.61,和 5 个客户端
× 8 = 40 个会话的 31.60 几乎一样,说明客户端不是瓶颈。

**两条链的可加性**(批量,每链 30 个在途会话,15 台各起客户端):

| | 单独跑 | 同时跑 |
|---|---|---|
| A 链 | 33.30 | 34.24 |
| B 链 | 31.68 | 30.80 |
| 合计 | 64.98 | **65.04** |

**网关放在哪**(闭环,128 token 一个会话,900 秒,前 120 秒不计):

| 客户端在哪 | 进程 | 每链在途 | A | B | 合计 |
|---|---|---|---|---|---|
| layer 0 节点 N01 / N11(本身在服务) | 1 | 16 | 21.00 | 21.00 | 42.0 |
| 闲置 T4 N12 / N14 | 1 | 15 | 25.93 | 27.08 | 53.0 |
| 15 台分散,每台 1 个 worker | 15 | 15 | 29.04 | 26.75 | 55.8 |

挪到闲置机器上就追回了约 80%:主要是客户端和服务端抢同一台机器的 CPU
(每个 token 客户端要做 embedding、CPU 上 fp32 的 lm_head,以及 6 跳的序列化)。
A 链从单进程到多进程还有 3 tok/s 的差,约等于一批会话的量化误差:单进程里的 worker
同时起步、长度相同,会一直成批完成,分辨率只有约 ±2.6 tok/s。

**不钉死的 15 台**(自动布局,10/03):30 个会话 34.14、45 个会话 34.44 —— 和 7 台时的
32.4 / 32.6 几乎一样。

### 为什么是这个数

**客户端站在每一跳中间。** `inference_session.py` 的 `step()` 逐段调用服务端,每段的输出
先回到客户端再发给下一段,没有服务端之间的直传。每个 token 是 6 次往返。

**服务端一次只做一个任务。** `PrioritizedTaskPool` 每次只取一个任务,不跨会话 batch。
于是每台的服务时间约 31 ms/步,而它自报的 `inference_rps≈313` 折合约 3.2 ms 的纯计算;
Petals 自己的路由代码里也写死了 `overhead_delay = 0.018`(序列化开销 18 ms)。

**推理路由是确定性的最短路。** `min_latency` 模式按 `0.018 + 层数 / inference_rps` 加上 RTT
跑 Dijkstra,所有客户端算出同一条路,每段公告 rps 最高的副本拿走全部流量。唯一的负载反馈是
`cache_tokens_left` 不够时的 10 秒惩罚,而 65536 的缓存能装 170 个 384-token 会话,
永远触发不了。实测:自动布局下 16:24 有 4 台,一台接了 100%,另外三台一个会话都没有;
只有两台 rps 相差不到 1 ms 每跳的段(32:40、40:48)才部分分流。钉死成链再加
`--allowed-servers` 后,这个问题就不存在了。

`PETALS_INFERENCE_ROUTING=max_throughput`(`a3d5f54`)本该让副本随机分流,但做 A/B 的那一轮
补丁还没部署到节点上,两轮实际都是 `min_latency`,**所以它的效果没有被验证过**。
在容器里按同样的随机策略回放 20 万次:只要还有单副本的层(比如上面的第 7 层),
那一台仍然接 100%,随机路由也没用;每层都 ≥2 时,最忙的一台降到 50%。

### 下一步(未验证)

既然每跳的代价基本是固定开销、和带几层关系不大,那每台带的层越多,一条链用的机器越少,
同样的卡就能组越多条链:

| 每台层数 | 每链几台 | 12 张 24GB 卡几条链 | 预测吞吐 |
|---|---|---|---|
| 8(现在) | 6 | 2 | 65(实测) |
| 12 | 4 | 3 | ~95 |
| 16 | 3 | 4 | ~120 |

16 层约 18.7 GiB 权重,加缓存约 21 GB,是 24GB 卡的极限(可以把 `ATTN_CACHE_TOKENS` 降到
32768 省出约 1 GB);分片约 26 GB,有几台磁盘偏紧。先用 3 台 A30 钉成 `0:16 / 16:32 / 32:48`
单独压一条链:还能到 30 tok/s 上下,这个思路就成立;明显掉下来,说明 16 层时计算已经
不能忽略,12 层会是更好的折中。

---

## 8. 真实负载:逐条回放 GSM8K / MBPP / No Robots

> 完整操作流程(复现、看结果、收尾、做新实验、排错)见 [workload-runbook.md](workload-runbook.md);
> 一键入口 `bash examples/workload_repro.sh all`。

第 7 节一直用同一条合成 prompt,量的是容量。这一节换成真实分布的 prompt,回答另一个问题:
**一个用户在空闲的链上,实际等多久。**

| 脚本 | 做什么 |
|---|---|
| `examples/workload_sample.py` | 从 HF dataset viewer API 抽样,写 `task/workload/prompts.jsonl`。只用标准库:gsm8k、mbpp 这种小的 split 按 100 行一页整个读下来再本地抽,lmsys 一百万条则每次请求取一行随机 offset。请求之间默认隔 1 s,遇到 429 按 `Retry-After` 等(第一次跑时连发 50 个请求就被限流过)。`--datasets norobots --keep` 只重抽一个、保留文件里其他的。每个数据集一个独立的随机数发生器,同一个 `--seed` 结果可复现,增删一个数据集不影响其他的 |
| `examples/workload_bench.py` | 逐条跑:上一条结束立刻发下一条,并发始终为 1。每条一个 inference session,每次 `generate()` 一个 token,所以 TTFT 和每两个 token 之间的间隔都能量到。每条打印一行 `REC {json}`,日志本身就是结果文件;`--report` 把多份日志合起来出报告,不需要 torch |
| `examples/cluster_gpumon.sh` | 每台一条长连接 ssh 跑 `nvidia-smi -l`,记 GPU 显存和利用率,供报告里的 Memory / Computation 用 |

数据来源:`openai/gsm8k` (main/test,1319 条)、`google-research-datasets/mbpp` (full/test,500 条,
用原论文的提示格式)、`HuggingFaceH4/no_robots` (default/test,500 条,人工写的指令和回答;
**只取单轮会话**:去掉 system 后恰好是一条 user 加一条 assistant,多轮的 Chat 类跳过;有 system
消息的保留它作为 prompt 的一部分,`meta.has_system` 标出来;超过 6000 字符的重抽)。

最早第三个数据集是 `lmsys/lmsys-chat-1m`(第一条用户消息),2026-10-05 起默认换成 No Robots 的
单轮会话:lmsys 是 gated 的、要 HF_TOKEN 和接受条款,而且真实用户输入里大量是极短或重复的
寒暄。lmsys 仍可用 `--datasets gsm8k mbpp lmsys` 选。因为每个数据集用自己的随机数发生器,
换掉第三个不改变 gsm8k 和 mbpp 抽到的题,前后两轮在这两个数据集上可以逐题对比。

### 指标口径

| 指标 | 怎么算 |
|---|---|
| TTFT | 发出请求到第一个 token 返回,包含整条链上的 prefill |
| generation latency | 端到端,直到 EOS 或 `--max-new-tokens` |
| p50 / p90 / p99 | nearest-rank。注意 50 条样本的 p99 就是最大值,150 条时是第二大 |
| TPOT | (latency − TTFT) / (输出 token − 1);另列每个 token 间隔的分布 |
| throughput | Σ输出 token / Σ请求耗时。并发 1 下这就是单链单用户的速度,不是第 7 节的容量;另列含 prompt 的 token/s 和每分钟请求数 |
| Memory | server 端 KV cache = (prompt + 输出) × 48 层 × 2048 B,即每 token 96 KiB,整条链合计;client 进程峰值 RSS;gpumon 记的每台 GPU 显存峰值 |
| Computation | 估算 FLOPs = 2 × 3.3B 激活参数 × token 数(未计 attention);client 每条请求的 CPU 秒;每台 GPU 利用率均值和峰值 |

默认值:`--max-new-tokens 256`;**关闭 thinking**(`enable_thinking=False`,否则 Qwen3 先写几百个
token 的思考,generation latency 量的就是思考长度);prompt 超过 2048 token 记为 skipped;
单条超过 `--request-timeout`(1800 s)视为链已经挂了,整轮以退出码 3 结束,而不是无声地卡住。
某一条抛异常只记 failed,接着跑下一条。

### 跑法

`$A` `$B` 是第 7 节里两条钉死的链的 peer ID 列表。

```bash
# 1. 抽样:需要能访问 huggingface.co 的机器(prin3 不行就在 N08 上跑,再拷回 task/workload/)
export HF_TOKEN=hf_...
python3 examples/workload_sample.py --out task/workload/prompts.jsonl   # 默认 3 × 50 条, seed 0
bash examples/qwen_cluster.sh deploy          # deploy 会把 task/ 一起带到各节点

# 2. 冒烟:每条链 3 条,确认 REC 行正常
CLIENT_SCRIPT=workload_bench.py bash examples/qwen_cluster.sh client --node N01 \
  --prompts task/workload/prompts.jsonl --allowed-servers $A --tag A --limit 3

# 3. 正式:两条链各从自己的 layer 0 节点出发,同时跑
examples/cluster_gpumon.sh /tmp/gpu.csv &
CLIENT_SCRIPT=workload_bench.py bash examples/qwen_cluster.sh client --node N01 \
  --prompts task/workload/prompts.jsonl --allowed-servers $A --tag A > /tmp/wl.A.log 2>&1 &
CLIENT_SCRIPT=workload_bench.py bash examples/qwen_cluster.sh client --node N11 \
  --prompts task/workload/prompts.jsonl --allowed-servers $B --tag B > /tmp/wl.B.log 2>&1 &
wait %2 %3; kill %1
python3 examples/workload_bench.py --report /tmp/wl.A.log /tmp/wl.B.log --gpu-csv /tmp/gpu.csv
```

两条链各跑全部 150 条,相当于同一个实验做两遍,报告最后按 tag 分行,可以直接对比 A 和 B。
想省一半时间,改成 A 用 `--shard 0/2`、B 用 `--shard 1/2`,各跑一半。

时间:单 session 每 token 要走完 6 跳,约 0.2 s,256 个 token 约 50 s。150 条 2–3 小时;分片约一半。

**先确认 server 的 `max_batch_size` 不小于最长的 prompt。** 第一次跑 150 条时,两条链各有 11 条
失败,全是 `Task size greater than max_batch_size (256)`:prefill 一步要把整个 prompt(含 chat template)
送进第一台 server,而 `run_qwen_server.sh` 从部署起就写死 `--max_batch_size 256`。第 7 节的合成
prompt 很短,所以从没碰到。现在默认 2048(可用 `MAX_BATCH_SIZE` 覆盖),改完要 `deploy` 再
`service restart`;server 没有固定 identity,重启后 peer ID 会变,`$A` `$B` 要重新取。
失败的那几条不用整轮重跑:`--only-failed <旧日志>` 只重发它们,报告时把旧日志和重跑日志一起传进去,
同一 (tag, id) 以最后一条为准。

两点要记着:client 跑在 N01 / N11 上,而它们本身也是 server,第 7 节量过这会多出约 25% 的开销,
这是"从 layer 0 节点发起"本身的代价,不是脚本的;两条链互不共享 server,所以同时跑不互相排队。

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

真机(15 节点):

- 2026-10-01:48 层全覆盖,15 台 ONLINE,客户端生成中文正常;`cache_tokens_left` 在 8 层和
  7 层两种 span 上都与离线公式逐位吻合
- 同一天下午约 14:08,时钟漂移的 8 台在一次重启后再也没起来(第 5 节"15 台只剩 7 台"),
  10/03 才发现。**在那之前的所有压测都只有 7 台在服务**
- 2026-10-03:时钟永久修复(N01 对内授时),15 台恢复;钉死成两条链
- 2026-10-04:吞吐基准完成,见第 7 节

还没做的:

- 第 8 节的真实负载测试:脚本已写好并用假模型离线验证过(计时、EOS/长度截断、失败与超时、报告合并),
  还没在集群上跑过;抽样脚本连不上 HF 的环境里没法测,第一次跑时先看它的输出
- `PETALS_INFERENCE_ROUTING=max_throughput` 的真实 A/B(上次那轮补丁没到节点,作废)
- 每台 12 / 16 层、组更多条链的布局(第 7 节"下一步")
- 另外 6 台(N02 N06 N08 N10 N11 N13)仍跟公网 NTP,和 N01 是两套来源。短期都是毫秒级,
  想彻底统一就把第 5 节那段 chrony 配置在这 6 台上也跑一遍
