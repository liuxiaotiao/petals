# Qwen3.6-35B-A3B 私有 swarm 运行手册

控制节点 `~/petals`,15 个 GPU 节点,SSH 走 22 端口,Petals 服务端口 9101。
所有命令都在**控制节点**执行,不需要登到各个节点上去。

本手册配套 `docs/qwen3.6-deployment.md`(讲原理和取舍),这里只讲怎么按顺序敲。

---

## 0. 这套集群的既定事实

写在最前面,因为下面每一步的写法都由它决定:

| 事实 | 后果 |
|---|---|
| 只有 **N06 / N07 / N08** 能访问 HuggingFace 的内容 CDN,其余 12 台被防火墙静默丢包 | 必须先起代理,否则 12 台永远下不动 |
| N08 磁盘 44G 且有出口 | 它当代理节点(`PROXY_NODE=N08`) |
| N06 磁盘 26G、N07 磁盘 25G | 这两台层数被磁盘卡住,比别人少;但它们有出口,不用走代理 |
| 15 台时钟都由 chrony 对齐公网 NTP | 不要跑 `synctime --yes`,会把好钟弄坏 |
| 这些节点只有自己在用 | 允许 `cleanup --gpu --yes` 杀掉占卡的外来进程 |
| `.qwen-cluster/` 是运行状态,不是代码 | 同步代码时**不要**覆盖它,否则代理注册和 bootstrap 地址都会丢 |

---

## 1. 控制节点环境变量

把这段存成 `~/petals/env.sh`,每次开新终端 `source ~/petals/env.sh`:

```bash
cd ~/petals

export NODE_PY=/home/ubuntu/anaconda3/envs/moe/bin/python  # 各节点已有的 conda 环境
export HF_HUB_DISABLE_XET=1        # 走普通 CDN,不走 xethub
export MAX_DISK_SPACE=30GB         # 每台 HF 缓存上限
export PROXY_NODE=N08              # 借出口的那台
export PROXY_SKIP="N06 N07"        # 这两台自己有出口,别绕道
export CLEANUP_ON_START=gpu        # start 前先清掉占卡的外来进程
# export ATTN_CACHE_TOKENS=65536   # 每台的 KV/状态预算,决定并发上限。见 4.5
# export INFERENCE_MAX_LENGTH=4096 # 单个会话 prompt+生成 的上限。见 4.5
```

不设 `HF_ENDPOINT` 就走 `huggingface.co`。hf-mirror 也验证过可用,
要换就 `export HF_ENDPOINT=https://hf-mirror.com`,两边都行。

---

## 2. 从零到可用:完整顺序

```bash
source ~/petals/env.sh

# ---- 第 1 步:清场 ----
bash examples/qwen_cluster.sh cleanup              # 只列不杀,先看看有什么
bash examples/qwen_cluster.sh cleanup --ours --yes # 杀本部署的遗留
bash examples/qwen_cluster.sh cleanup --gpu  --yes # 连占卡的外来进程一起清

# ---- 第 2 步:体检(此时 12 台的 cdn= 会失败,正常)----
bash examples/qwen_cluster.sh preflight

# ---- 第 3 步:推代码、建 venv ----
bash examples/qwen_cluster.sh deploy

# ---- 第 4 步:起代理(必须在 deploy 之后,N08 上要有代理脚本)----
bash examples/qwen_cluster.sh proxy start

# ---- 第 5 步:再体检,这次应该 15/15 全绿 ----
bash examples/qwen_cluster.sh preflight

# ---- 第 6 步:按实测显存/磁盘定每台层数 ----
bash examples/qwen_cluster.sh plan --cap 8          # 先看
bash examples/qwen_cluster.sh plan --cap 8 --write  # 合理再写回 task/hosts.txt

# ---- 第 7 步:起服务 ----
bash examples/qwen_cluster.sh start

# ---- 第 8 步:盯着下载 ----
bash examples/qwen_cluster.sh status --watch
```

第 4 步和第 5 步的顺序不能换:代理脚本随 `deploy` 一起 rsync 过去,
没 deploy 就没有 `repo/examples/qwen_http_proxy.py`,`proxy start` 会失败。

`--cap 8` 的理由:不设上限的话 A30 会被算成 13 层,显存只剩不到 1 GiB
余量——首轮 OOM 就是这么来的。8 层仍有约 2.9 倍冗余,且每台少下约 6.5 GB,
在代理那条唯一的上行链路上总共少走约 78 GB。

### 分批起(可选,但推荐)

12 台的下载全挤 N08 一条上行。分两批能更快拿到一个可用的 swarm:

```bash
# 第一批:6 台 A30 × 8 层 = 48 个层位,够盖满 40 层
awk '$1 ~ /^N(0[1-5]|08)$/ || /^#/' task/hosts.txt > task/hosts.wave1
HOSTS_FILE=task/hosts.wave1 bash examples/qwen_cluster.sh start
bash examples/qwen_cluster.sh status --watch
#   等到 "Every layer is online"

# 第二批:补上剩下 9 台。已在跑的会自动跳过(already-running)
bash examples/qwen_cluster.sh start
```

分批只影响 `start`。**`proxy start` 始终用完整的 `task/hosts.txt` 跑**——
代理的客户端白名单是从 `HOSTS_FILE` 生成的,拿分批文件去起代理,
第二批那 9 台会被自己的代理拒之门外。

分批文件**不需要**包含 `BOOTSTRAP_NODE`。缺了的话,地址从 `.qwen-cluster/bootstrap_peer`
里取,并且假定那个 DHT 已经在跑(缓存存在本身就说明它起过)。

`stop` 配分批文件用是安全的:它默认不动 bootstrap DHT。想连 DHT 一起停,要显式
`stop --dht`——那会把整个 swarm 停掉,包括这个文件里没列的节点。

---

## 3. 客户端

等 `status` 报 "Every layer is online" 之后:

```bash
bash examples/qwen_cluster.sh client \
  --prompt '请解释一下分布式推理的工作原理。' \
  --max-new-tokens 128
```

**控制节点上没有装 petals,也不需要装。** `client` 在一个节点上跑生成——
那里 venv、仓库、tokenizer 缓存都是现成的。默认用 `CLIENT_NODE`(默认等于
`PROXY_NODE`,因为它一定能访问 Hub),`--node N03` 可以指定别的。
`--initial-peers` 自动填,`--prompt` 之后的参数原样转给 `qwen_generate.py`。

它默认带 `PETALS_MAX_RETRIES=3`。客户端原本的重试预算在这条路径上等价于**无限重试**,
真出问题时你看到的是永远转圈而不是报错。

服务端如果固定了 `MODEL_REVISION`,客户端要加同样的 `--revision`。
两边都不指定前缀时,会从仓库名推出同一个 DHT 前缀
`Qwen3-6-35B-A3B-petals-qwen-v1`。

这是 Petals 的 Python 调用,**不是 HTTP / OpenAI 接口**,本部署没有做 HTTP 网关。

---

### 某一段层始终没人认领

`status` 里所有 span 的起点都大于 0(比如最小是 6),说明 layer 0–5 没有任何服务端认领,
即使其余全部 ONLINE,客户端仍然不能生成。再平衡靠各节点上报的吞吐权衡,吞吐在一台正忙着
下载的机器上测出来会低得离谱(见过 11 tokens/sec 对 610),判断因此失真。

把一台钉死在那一段:

```
N01  192.168.1.2:9101   blocks=0:8   # 钉死 0–7 层,该台不再参与再平衡
```

`blocks=N` 是"服务 N 层,位置交给 swarm 决定";`blocks=起:止` 是"就服务这一段"。
`plan` 不会覆盖带冒号的行——那是人为决定。改完照常 `service install` + `service restart`。

## 3.5 开机自启与崩溃自愈

`start` 起的服务端是裸进程:机器重启就没了,进程崩了也不会回来。要长期跑,装 systemd
用户单元:

```bash
bash examples/qwen_cluster.sh service install   # 写单元、开 lingering,不启动
bash examples/qwen_cluster.sh service start     # 把运行中的 swarm 交给 systemd 接管
bash examples/qwen_cluster.sh service status
```

装完之后**用 `service start/stop/restart`,不要再用 `start`/`stop`**。两个管理者抢同一个
端口,正是这套集群已经付过一次学费的故障(见下面 `peer id mismatch` 那节)。`service start`
会先把裸进程停掉再交给 systemd,避免重演。

单元里几个值得知道的设定:

- `Restart=always` + `RestartSec=20`,崩了自动拉起。
- `StartLimitIntervalSec=900` / `StartLimitBurst=5`:15 分钟内崩 5 次就停手,不然一个
  起不来的服务端会不停地捶 Hub 和显卡。
- `TimeoutStopSec=180`:Petals 关停要注销层、释放显存和端口。给得太短正是两代进程
  共用一个端口的成因。
- `ExecStartPost` 仍然写 `run/server.pid`,所以 `status`、`diag`、`cleanup --stale`
  照常可用。
- 环境从 `run/server.env` 读,里面有每台自己的 `NUM_BLOCKS`、`ANNOUNCE_IP`,以及
  **按主机决定的 `HTTPS_PROXY`**——代理节点自己那台不会被写入代理变量。

**`linger=NOT-ENABLED` 必须处理。** 没有 lingering,用户单元在 SSH 会话结束时就停,
开机也不会起,等于白装。`service install` 会尝试自动开(先试无 sudo,再试 `sudo -n`),
开不了就明说。那种情况去那台机器上跑一次:

```bash
sudo loginctl enable-linger $(id -un)
```

改了 `task/hosts.txt` 的层数、换了 `HF_ENDPOINT`、或者代理地址变了,重跑
`service install` 刷新 env 文件,再 `service restart`。

`service logs <节点>` 看 journal,`service uninstall` 卸掉单元(lingering 保留)。

## 4. 子命令速查

| 命令 | 作用 |
|---|---|
| `preflight` | 只读体检:python/torch/GPU/venv/git/rsync/github/pypi/HF 可达性/显存/磁盘/时钟。不改任何东西 |
| `plan [--cap N] [--write]` | 用 preflight 量到的真实空闲显存和磁盘算每台层数,写回 `task/hosts.txt`(留 `.bak`) |
| `deploy` | rsync 本仓库到各节点 `~/petals-qwen/repo`,基于 `NODE_PY` 建 venv。幂等 |
| `proxy start\|stop\|status\|logs` | 在 `PROXY_NODE` 上起 CONNECT 代理,并让其余节点经它访问 HF |
| `start [--restart]` | 起 bootstrap DHT,再起所有 GPU 服务端。默认跳过已在跑的;`--restart` 先停后起,**改环境变量的唯一办法** |
| `client [--node N] [...]` | 在某个节点上跑生成,参数转给 `qwen_generate.py`。控制节点不需要装 petals |
| `bench [...]` | 在某个节点上跑吞吐基准,参数转给 `bench_qwen.py`(`--concurrency`/`--new-tokens`/`--timeout`/`--inline`) |
| `status [--watch]` | 每台的进程状态 + 缓存大小 + 下载速率,加上 DHT 里的层覆盖 |
| `diag` | 没上线时用:进程死活、缓存大小、日志最后一条错误 + 该错误有多旧、bootstrap DHT 状态 |
| `logs <节点> [行数]` | 看某一台的服务端日志 |
| `cleanup [--ours\|--gpu\|--stale] [--yes]` | 列出/清理残留进程。默认只列不杀。`--stale` 只杀**上一代**服务端,保留当前那个 |
| `synctime [--yes]` | 看时钟偏差。全部 NTP 同步时会拒绝 `--yes` |
| `hosts` | 打印解析出来的节点表 |
| `service {install\|start\|stop\|restart\|status\|logs\|uninstall}` | systemd 用户单元:崩溃自愈 + 开机自启。装了之后用它代替 `start`/`stop` |
| `stop [--dht]` | 停 `HOSTS_FILE` 里那些服务端。**默认不碰 bootstrap DHT**——停它等于停掉整个 swarm,包括当前 hosts 文件里没列的那些。`--dht` 才一并停 |

---

## 4.5 会话长度与并发上限

两个参数,作用不一样,别混:

**`INFERENCE_MAX_LENGTH`(默认 4096)** 卡的是**单个会话** prompt + 生成的**总和**,
即 2048 进 + 2048 出。客户端每次申请的是 `prompt + max_new_tokens`
(`remote_generation.py:110`),服务端就按这个数记账——所以这个上限只决定
**一个调用者最坏能要多少**,并不会让短请求变便宜。超了服务端直接拒:
`Cannot allocate KV cache for N tokens, max = 4096`。

**`ATTN_CACHE_TOKENS`(默认 65536)** 是缓存**预算**,不是预留。`MemoryCache` 只记字节数,
张量在会话进来时才真分配。所以预算开得比显卡余量大**不会**在启动时报错,而是让服务端
一直收会话直到 CUDA 自己 OOM——这比干净的 `AllocationFailed`(客户端会重试并绕路)糟得多。
启动时预算超过本机层数留下的余量,服务端会打一条 warning。

一个会话在 8 层 span 上的开销是**仿射的**:

```
  12,976,176 字节   6 个 linear_attention 层的 conv + recurrent state,与长度无关
+     28,672 字节 × (prompt + 生成)
```

那 28,672 里有 24,576 是 linear 层的 `history` 缓冲(`block.py:188`):linear 的递归状态
没法像 KV 那样切片,Petals 回退会话时只能拿原始块输入重放,所以必须留住全部输入。
这是容错的代价,长会话贵就贵在这里。

| `ATTN_CACHE_TOKENS` | 预算(8层 / 7层) | 并发 @4096 | 并发 @2048 | 并发 @256 |
|---|---|---|---|---|
| 16384 | 0.52 / 0.45 GiB | 4 / 3 | 7 / 7 | 27 / 24 |
| 32768 | 1.02 / 0.89 GiB | 8 / 7 | 15 / 14 | 53 / 48 |
| **65536(现默认)** | **2.02 / 1.76 GiB** | **16 / 15** | **30 / 28** | **106 / 95** |
| 131072 | 4.02 / 3.51 GiB | 33 / 30 | 60 / 55 | 212 / 190 |

改法:`export ATTN_CACHE_TOKENS=...`,`service install` + `service restart`
(没装 systemd 单元就 `start --restart`)。这是启动参数,不重启不生效。
`service install` 每行会打出生效的 `cache=预算/单会话上限`,不用猜写进去的是什么。

**预算满了不是崩溃。** 服务端回 `Could not allocate N bytes immediately: out of memory`,
客户端重试并重新路由,那台在 `status` 里仍然是 ONLINE,会话结束就把空间还回去。
不用重启任何东西。

**T4 要看一眼。** 16GB 卡放 7 层权重之后余量不多,65536 的预算是 1.76 GiB。
重启后 `service logs N12` 里如果有 "may grow to ... but ... leave only about ..." 这条
warning,就把 `ATTN_CACHE_TOKENS` 调小重装。

---

## 5. 故障对照

### preflight 的 `cdn=` 失败

`cdn=` 是真的去取一次模型索引、再对真实分片发 `Range: bytes=0-0`,
拿到那一个字节才算 `ok`。只 ping 域名不算数——典型故障恰恰是 API 通、
内容 CDN 被挡,那样服务端会抱着空缓存无限重试而不是干脆报错。

| 取值 | 含义 | 处理 |
|---|---|---|
| `UNREACHABLE-TimeoutError` | 防火墙静默丢包 | 起代理,或让网络组放行 `*.cdn.hf.co`、`cdn-lfs*.huggingface.co` |
| `UNREACHABLE-ConnectionRefusedError` | 端口被拒 | 同上 |
| `UNREACHABLE-SSLCertVerificationError` | 有 TLS 中间人,解释器不信它的 CA | 给 Python 指系统 CA(**不要**关校验) |
| `UNREACHABLE-gaierror` | DNS 解析不了 | 查这台的 resolver |
| `NO-INDEX-404` | 该端点没收录这个仓库 | 换 `HF_ENDPOINT` |
| `NO-INDEX-401/403` | 仓库是 gated | 接受条款并设 `HF_TOKEN` |

### 服务端一直 JOINING,缓存停在 100KB 上下

这不是慢,是卡死,而且原因几乎总是同一个:**那台的进程环境里没有 `HTTPS_PROXY`**。

100KB 左右正好是 `config.json` 加 `model.safetensors.index.json` 的体积。元数据走
`huggingface.co` 的 API(没被墙),一到 CDN 取真正的分片就超时重试,永远不会自己好。

```bash
# 决定性的一条:直接看进程环境
ssh -n ubuntu@<该节点IP> \
  "tr '\0' '\n' < /proc/\$(cat ~/petals-qwen/run/server.pid)/environ | grep -i proxy \
   || echo '(没有任何 proxy 变量)'"
```

环境变量只在**进程启动时**注入,所以改不了正在跑的进程。而 `start` 默认跳过已在跑的
服务端,只会打印 `already-running`——**重跑 `start` 修不好这个问题**。必须:

```bash
bash examples/qwen_cluster.sh proxy start              # 确保注册在
bash examples/qwen_cluster.sh start --restart          # 先停后起
```

`start` 现在会在启动前打印一行 "Hub access: N host(s) via ...",没有代理时会明确警告。
启动时扫一眼这行,比事后查十小时划算。

### 客户端报 `peer id mismatch`

```
failed to dial 12D3Koo...1md61N: all dials failed
  * [/ip4/192.168.1.2/tcp/9101] peer id mismatch: expected ...1md61N,
    but remote key matches ...RSAABh
```

**一台机器上跑着两代服务端。** libp2p 用 SO_REUSEPORT,所以新旧两个 p2pd 能同时绑
同一个端口,进来的连接在两者之间分配——DHT 广播的是新 peer id,而端口上有一半概率
是旧进程应答。

确认(注意:`pgrep -c` 数出来的是主进程加 fork 的子进程,**不能**用来判断有几代;
要看启动时长):

```bash
ssh -n ubuntu@<节点IP> "ps -eo pid,ppid,etimes,args | grep '[p]etals.cli.run_server' \
  | awk '{printf \"pid=%-8s ppid=%-8s age=%ss\\n\", \$1, \$2, \$3}'"
ssh -n ubuntu@<节点IP> "ss -ltn | grep -c ':9101'"   # 正常是 2(v4+v6),4 就是两代
```

`age` 分成两簇、`ppid=1` 的有两个,就是它。清掉旧的那一代:

```bash
bash examples/qwen_cluster.sh cleanup --stale          # 先看
bash examples/qwen_cluster.sh cleanup --stale --yes    # 再杀
```

`--stale` 只杀 `ppid=1` 且不等于 `run/server.pid` 的那些,当前服务端不受影响。
**不要用 `--ours`**,那会把健康的那个一起杀掉。

清完不用重启任何服务端:DHT 里指向旧代的记录会自己过期,而端口上不再有冒名顶替的
进程,拨号立刻就正常了。

### `status` 长时间停在 JOINING

正常。35B 的权重挤一条上行,慢是预期。看 `RATE` 列:

- 有数字 → 在下,等着
- 某台一直 `0.0 MB/s` 而别人在动 → `logs <那台> 80`
- 全部 `0.0 MB/s` → `proxy logs 40` 看代理是不是挂了

### `status` 说 "No bootstrap address cached"

`.qwen-cluster/bootstrap_peer` 丢了。新版会自动从 N01 的 `logs/dht.log` 捞回来;
如果控制节点上还是旧脚本:

```bash
mkdir -p .qwen-cluster
ssh -n ubuntu@192.168.1.2 \
  "grep -ao '/ip4/192.168.1.2/tcp/31337/p2p/[A-Za-z0-9]*' ~/petals-qwen/logs/dht.log | head -1" \
  > .qwen-cluster/bootstrap_peer
```

**同时一定要重跑 `proxy start`**。代理的注册信息也在 `.qwen-cluster/` 里,
丢了之后 `start` 和 `preflight` 就不再给节点注入 `HTTPS_PROXY`,12 台会退回
没有出口的状态。`proxy start` 见进程还活着会打印 `already-running` 直接返回,
不会重启它,只把注册补上——所以这条命令随时可以重复执行。

### 一个服务端都没上线

```bash
bash examples/qwen_cluster.sh diag
```

先看输出末尾的 bootstrap DHT:它要是 DEAD,所有服务端都会以同样的方式失败,
先修它。每行的 `age=` 是那份日志多久没写了——DEAD 且 age 很老,说明你看到的是
过去某次失败的残骸,不是此刻正在发生的问题。

### 时钟

`clock=` 那一列标了来源:`(ntp)` 是从该机自己的时间守护进程读的,毫秒级可信;
`(rtt)` 是 SSH 往返估的,误差可能有一两秒,判定时会先扣掉误差棒,所以它不会
因为噪声把健康节点拦下来。全部 `(ntp)` 且 `ok` 就别管时钟。

---

## 6. 停止

```bash
bash examples/qwen_cluster.sh stop --dht  # 所有服务端 + bootstrap DHT
bash examples/qwen_cluster.sh proxy stop  # 代理
```

不加 `--dht` 只停服务端,DHT 留着——想重起服务端而不打扰整个 swarm 时用这个。

bootstrap 的身份文件保留,下次 `start` 的 peer 地址不变,客户端不用改。
