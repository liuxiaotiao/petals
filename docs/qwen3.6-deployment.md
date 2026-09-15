# Qwen3.6-35B-A3B：实验性 Petals 纯文本部署

本分支新增 `qwen3_5_moe` 适配，直接读取 `Qwen/Qwen3.6-35B-A3B` 原始权重。
计算代码从 Transformers v5.5.0 的 Qwen 实现回移，保留仓库固定的 Transformers 4.43.1。
无需把全项目升级到 Transformers 5，也无需转换权重名称。

## 支持范围与验证边界

- 支持纯文本、原始 BF16 checkpoint 加载为 FP16/BF16/FP32、按层跨节点推理、贪心/采样生成和会话续写。
- 完整注意力使用 KV cache；线性注意力使用 FP32 recurrent state、卷积状态及输入历史。
  回退会话时只重放相应层的 attention mixer；正常续写复用状态。
- 单个服务进程只使用一张卡；多卡请启动多个进程，按层划分。`--tensor_parallel_devices` 多卡模式不支持。
- 不支持量化 checkpoint、NF4/INT8、视觉、MTP、beam search、LoRA 和 prompt tuning。
  此版本面向推理验收，未验收训练；启动脚本默认加 `--inference_only`，见下节。
- 使用 eager PyTorch kernels。长 prompt、MoE 路由和较旧 GPU 可能较慢，尚未接入 FLA/fused experts。
- 小尺寸模型数值对照、缓存和本机 RPC 测试可运行；没有在上述五种 GPU 上装载完整 35B 模型测速。
  这里的层分配是起始配置，不能视为显存、吞吐或稳定性验收结果。

## 层分配

**默认自动。** 不指定 `--block_indices` / `--num_blocks` 时，每台服务器用
`_choose_num_blocks()` 按本机空闲显存算自己能装几层，再用 `choose_best_blocks()`
占住当前 swarm 里最薄弱的一段连续层；运行中按 `--balance_quality`（默认 0.75）
定期检查并在必要时整段迁移。所有 GPU 主机因此跑同一条命令，不需要各自配置层范围。

所有节点统一 FP16。每层权重按实际参数形状计算，线性层约 1.570 GiB，完整注意力层约 1.558 GiB。
下表仅计算权重，运行还需要缓存、临时激活、CUDA 上下文等空间，所以实际能装的层数要少一些。

| GPU | 标称显存 | 层权重上限估算 | 实测建议值 |
| --- | --- | --- | --- |
| A30 | 24 GB | 约 14 层 | 11–13 |
| RTX 6000 | 24 GB | 约 14 层 | 11–13 |
| T4 | 16 GB | 约 9 层 | 6–8 |

想固定分层时用 `BLOCKS=start:end`（等价于 `--block_indices`），
想只限层数、让 swarm 决定位置时用 `NUM_BLOCKS=N`。
注意 `BLOCKS` 会同时关闭该服务器的再平衡：`strict_block_indices` 不为空时
`_should_choose_other_blocks()` 无条件返回 False。

客户端额外加载 embedding 和 LM head，FP32 参数约 3.79 GiB，建议至少 8GB 可用内存。
建议每个 GPU 节点至少 32GB 主存，并为 Hub 分片缓存留出约 100GB 磁盘空间。
多个进程在同一主机时要合并计算 CPU 内存需求。分片文件可能包含本节点不负责的层。

初次采用单请求、2048 tokens 上下文，验证后再提高上下文和并发。
缓存对并发按实际 tensor 大小分配，多个会话的固定状态会额外占用缓存池。
速度受慢节点和网络延迟影响；同一局域网内测试后再考虑跨网络部署。

## 只部署推理

服务端加 `--inference_only`（`examples/run_qwen_server.sh` 已默认带上）：

- 拒绝 `rpc_backward` / `rpc_backward_stream`，任何对端都无法通过这台服务器跑反向传播；
  同时不启动 backward task pool，少一个 worker 线程和对应的显存峰值。
- **仍然提供** `rpc_inference`（`generate()`、会话续写）和 `rpc_forward`。
  `rpc_forward` 是无状态、无梯度的一次前向，`model(input_ids)` 取 logits 走的就是它，属于推理。
- 训练本来就没在这个适配上验收过，关掉它等于把未验证的代码路径从对外接口上摘掉。

客户端有个坑要注意：Petals 把 `max_retries=0` 当成**无限重试**
（`sequential_autograd.py` 里的判断是 `attempt_no + 1 == max_retries`，0 永远不成立）。
所以误调 backward 时客户端会按指数退避一直重试，而不是报错退出。
连接只推理的 swarm 时，把 `max_retries` 设成正整数。

不用管的参数：`--balance_quality`（固定 `--block_indices` 时根本不检查）、
`--adapters`、`--tensor_parallel_devices`（不支持）、`--quant_type`（必须 `none`）。

建议的递进验证顺序，每一步失败都不要往下走：

### 每台要留多少磁盘

服务器整片下载 safetensors，所以成本取决于层范围覆盖到几个分片。按真实的 26 分片索引，
连续区间的最坏情况（十进制 GB）：

| 每台层数 | 最少 | 最多 | 30GB 上限 |
| --- | --- | --- | --- |
| 14（自动分层会选到这个） | 23.56 | **30.53** | 不够 |
| 12 | 20.19 | 26.25 | 够 |
| 11（建议值） | 18.51 | **25.48** | 够，余约 4.5 GB |
| 6（T4） | 10.09 | 15.37 | 够 |

`--max_disk_space 30GB` 按十进制解析，等于 27.94 GiB。超过上限时 `free_disk_space_for()`
按最久未访问顺序淘汰旧分片；真腾不出来会明确报 `Insufficient disk space to load a block`，
不会静默失败。再平衡把某台挪到别的层区间时会下新分片、淘汰旧的，缓存稳定在上限附近，不会无限增长。

## 验证路线

1. **不需要 GPU、不需要真权重**，先把链路跑通：
   `PETALS_TEST_LOCAL_SWARM=1 python -m pytest -o pythonpath=src tests/test_qwen_swarm.py -q`
   覆盖 DHT、跨节点 RPC、会话续写，以及 `--inference_only` 确实拒绝了反向。
2. **单卡**：只起一台，`--block_indices 0:11`，确认 CUDA/dtype/权重加载正常。
   此时全部 40 层没凑齐，不能生成，但能验证加载并用 `--throughput dry_run` 提前量出吞吐。
3. **五台全起**，确认日志里 40 层都 ONLINE，再跑 `examples/qwen_generate.py`。

35B 的 FP16 权重约 62.8 GiB，必须由整个 swarm 凑齐，没有"先用一台机器试全模型"这条路。
DHT 引导节点不提供模型层，可以和某台 GPU 主机共用，不必单独占一台机器。

## 安装（Linux NVIDIA）

在每个节点同步这份修改后的代码。从仓库根目录操作，使用独立 Python 3.10 环境。
旧卡需要检查所用 PyTorch wheel 的 CUDA 架构支持，不能直接假定最新 wheel 能运行 P100。
下面的 PyTorch 2.2.2 + CUDA 11.8 是待硬件验证的兼容性起始环境；安装命令来自
[PyTorch 官方历史版本说明](https://pytorch.org/get-started/previous-versions/#v222)。

```bash
python3.10 -m venv .venv-qwen
source .venv-qwen/bin/activate
pip install --upgrade pip
pip install 'setuptools<81' wheel 'grpcio-tools==1.60.0'
pip install 'torch==2.2.2' --index-url https://download.pytorch.org/whl/cu118
pip install --no-build-isolation -e .
python -c 'import torch; print(torch.__version__, torch.version.cuda); print(torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0)); print(torch.cuda.get_arch_list()); x=torch.ones(16,16,device="cuda",dtype=torch.float16); print((x@x).sum().item())'
```

Hivemind 的旧构建脚本使用 `pkg_resources`，因此这里固定 `setuptools<81` 并预装构建工具。
请保留 `numpy<2` 和 Transformers 4.43.1 的仓库约束。Pydantic 版本遵循 Hivemind 的依赖要求。
如果出现 `no kernel image`，先解决 CUDA wheel/驱动/架构兼容性，再加载模型。

## 从控制节点一键驱动（推荐）

`task/hosts.txt` 里每行是 `<节点id> <IP>:<Petals端口>`，SSH 走另一个端口（`SSH_PORT`，默认 22）。
`examples/qwen_cluster.sh` 用它把 15 台机器当一个集群管：

```bash
export SSH_USER=ubuntu                 # 各节点的登录用户，需已配好免密公钥
export MODEL_NAME=Qwen/Qwen3.6-35B-A3B
export MAX_DISK_SPACE=30GB             # 每台的 Hub 分片缓存上限，见下文磁盘一节
export NUM_BLOCKS=11                   # 24GB 卡；不设会自动选到 14 层并 OOM
# 各节点已有的解释器（conda 环境等）。设了它就不再另装 torch。
export NODE_PY=/home/ubuntu/anaconda3/envs/moe/bin/python

bash examples/qwen_cluster.sh preflight # 只读检查 15 台是否具备部署条件
bash examples/qwen_cluster.sh deploy   # rsync 本仓库到 15 台，各自建 venv 装依赖
bash examples/qwen_cluster.sh start    # 起 DHT，抓引导地址，再并发起 15 个服务端
bash examples/qwen_cluster.sh status --watch   # 轮询到 40 层全覆盖为止
bash examples/qwen_cluster.sh diag             # 没有服务端上线时先跑这个
bash examples/qwen_cluster.sh logs N07 100     # 看某台的日志
bash examples/qwen_cluster.sh stop             # 停服务端，再停 DHT
```

几个设计要点：

- **引导地址自动获取**。`start` 在 `BOOTSTRAP_NODE`（默认 N01）上起 `run_dht`，
  从它的日志里 grep 出 `/ip4/<IP>/tcp/<端口>/p2p/<PeerID>`，写进 `.qwen-cluster/bootstrap_peer`，
  再分发给所有服务端。因为带 `--identity_path`，这个地址跨重启不变。
- **ANNOUNCE_IP 自动按 hosts.txt 逐机设置**，这是跨站点部署的必填项。
- **进程用 pidfile 管理**，`nohup` 启动，SSH 断开不影响；`status` 逐台 `kill -0` 检查。
- **每台的输出留在控制节点** `.qwen-cluster/out/<节点id>.<阶段>`，哪台失败直接看那个文件。
- 需要覆盖默认值时，设 `DEVICE`、`TORCH_DTYPE`、`NUM_BLOCKS`、`BLOCKS`、
  `BALANCE_QUALITY`、`DHT_PREFIX`、`MODEL_REVISION` 即可，脚本只透传已设置的那些。

### 复用已有的 conda 环境

各节点已经有装好 torch 的环境时，设 `NODE_PY` 指向那个解释器。`deploy` 会用
`$NODE_PY -m venv --system-site-packages venv` 在它之上建一层 venv：

- **torch 从底层环境继承**，不重复下载（15 台省掉约 40 GB 和十几分钟）。
- **Petals 自己的 pin 落在 venv 里**，不动底层环境。这点很重要：`setup.cfg` 把
  transformers 钉死在 4.43.1，还要求 `numpy<2`、`peft==0.8.2`、`bitsandbytes==0.41.1`。
  直接装进 conda 环境会把这些版本按 Petals 的要求改掉，那个环境里的其他工作可能就跑不了了。
- 底层环境缺 torch 时 `deploy` 会明确报 `no torch in <解释器路径>` 并把该节点标为 FAIL，
  不会装到一半留个半残的环境。

`deploy` 结束时每台会打印 `节点 / petals 版本 / torch 版本 / GPU 型号`，
GPU 不可见的节点显示 `NO-CUDA`——这一步同时充当上真机前的预检。

需要在这层 venv 里另装或覆盖 torch，显式设 `TORCH_SPEC`（和 `TORCH_INDEX_URL`）即可；
不设 `NODE_PY` 时脚本回退到自建 venv 并按文档的 pin 装 `torch==2.2.2 + cu118`。

### deploy 负责什么、不负责什么

`deploy` 在控制节点一条命令，并行在所有节点上完成：建目录、rsync 代码、
`$NODE_PY -m venv --system-site-packages venv`、装构建工具、`pip install -e repo`、打印版本与 GPU。
重复执行是幂等的（venv 已存在就复用）。

它**不负责**的部分，必须事先在各节点就位：

| 前置条件 | 为什么 |
| --- | --- |
| `NODE_PY` 指向的解释器存在且能 `import torch` | deploy 只在它之上叠 venv，不会去装 conda 或 torch |
| 各节点装有 `rsync` | rsync over ssh 要求**两端**都有 |
| 各节点装有 `git` | `setup.cfg` 里 hivemind 是 `git+https://github.com/...`，pip 要 clone |
| 各节点能访问 GitHub 和 PyPI | 同上；离线网段会卡在这一步 |
| 免密 SSH（脚本用 `BatchMode=yes`，不会交互输密码） | 15 台并行时没有输密码的机会 |

`preflight` 把这些逐台查一遍，只读、不改任何东西：

```
  N01 python=3.10.14 torch=2.4.1+cu121 gpu=NVIDIA A30 venv=ok git=ok rsync=ok github=ok pypi=ok free=210G
  N12 python=3.10.14 torch=MISSING gpu=NO-CUDA venv=ok git=ok rsync=ok github=UNREACHABLE pypi=ok free=88G
```

任何一项是 `MISSING` / `UNREACHABLE` 就返回非零并指出有几台不合格，先修好再 `deploy`。

### 一个服务端都没上线时

`check_qwen_swarm.py` 会把 ONLINE 和 JOINING 分开报。服务端在**加载权重完成之前**
一直是 JOINING，35B 走 Hub 要很久，所以 `0 server(s) online, 12 still joining` 是正常的等待中状态。
`0 online, 0 joining` 才是出了问题。

```bash
bash examples/qwen_cluster.sh diag
```

```
  N01 running cache=18G lines=214 | no error lines          ← 正常，正在下载
  N02 DEAD cache=4.0K lines=31 | RuntimeError: CUDA error: no kernel image is available
  N03 NO-LOG cache=0 | server was never started on this host
```

按 `cache=` 是否在增长判断下载进度。若全部 `running`、无报错、但 `cache` 不涨，
多半是节点连不上 Hub，或者公告进不了 DHT——检查各网段到引导节点 `31337` 端口的连通性，
以及每台的 `ANNOUNCE_IP` 是否是别的网段能拨通的地址。

控制节点需要 `rsync` 和到各节点的免密 SSH；不需要装 Petals（覆盖检查是在 N01 上远程跑的）。

下面两节是手工分步的做法，排查问题时用得上。

## 建立私有网络

在可被各节点访问的 CPU 主机或任一 GPU 主机运行：

```bash
python -m petals.cli.run_dht \
  --host_maddrs /ip4/0.0.0.0/tcp/31337 \
  --identity_path ./qwen-dht.identity
```

从日志复制包含 Peer ID 的完整地址，例如 `/ip4/10.0.0.10/tcp/31337/p2p/12D3KooW...`。
`0.0.0.0` 是监听地址，不能用作其他机器连接的目标。
`--identity_path` 让 Peer ID 跨重启保持不变，所以这个地址只需要抄一次，
之后可以写进各节点的环境变量或 systemd 单元里。
节点之间需要能直接访问 DHT 端口和各 GPU 服务端口，且公告 IP 必须互相可达。
这个引导网络不提供身份认证；部署在受控局域网或 VPN 内，不将 GPU 服务端口开放给不可信网络。

每台 GPU 主机跑同一条命令，只有 `ANNOUNCE_IP` 不同：

```bash
export BOOTSTRAP_PEER='/ip4/10.0.0.10/tcp/31337/p2p/替换为日志中的PeerID'
export CUDA_VISIBLE_DEVICES=0
export ANNOUNCE_IP=10.0.0.11  # 当前 GPU 主机在该网络内可达的 IP
bash examples/run_qwen_server.sh
```

跨站点部署时 `ANNOUNCE_IP` 是必填的：节点不知道自己在别人眼里的地址，不设就只有同网段能连上。
若多张卡在同一主机，为每个进程分别指定 `CUDA_VISIBLE_DEVICES` 和 `PORT`，各进程内部仍是 `cuda:0`。
所有 40 层都就绪后才能生成，DHT 节点本身不提供模型层。

节点之间的组网是自动的：每台机器只需要知道引导地址，之后通过 DHT 互相发现并直连，
不需要把各机器的地址两两配置。但"都启动了"不等于"可以用了"——加载十几层权重要几分钟，
首次还要测吞吐；自动分层的结果还依赖各节点的加入顺序，不是确定的。
用下面这个脚本确认层覆盖，不要靠客户端报错来判断：

```bash
python examples/check_qwen_swarm.py --initial-peers "$BOOTSTRAP_PEER" --watch
```

它连上 DHT 后列出每台服务器负责的层区间，并检查 0–39 是否全被 ONLINE 服务器覆盖；
齐了退出码 0，缺层会打印缺哪一段并返回 1，适合放进启动脚本里等待。
缺层时客户端不会快速失败，而是按指数退避一直重试（见上面的 `max_retries` 说明），
所以务必先确认覆盖完整再启动客户端。

可选变量：`MODEL_NAME`（模型路径）、`MODEL_REVISION`（Hub commit）、`PORT`、`ANNOUNCE_IP`、
`DHT_PREFIX`、`DEVICE`、`BLOCKS`、`NUM_BLOCKS`、`BALANCE_QUALITY`。
验收/长期运行时请为所有服务端和客户端使用同一个 Hub commit，避免不同 revision 混用。
首次自动 throughput 测量可能耗时，测试期间不要把未测量的吞吐估计当成实际速度。

## 运行客户端

客户端安装同一份 Petals 代码，然后运行：

```bash
python examples/qwen_generate.py \
  --initial-peers "$BOOTSTRAP_PEER" \
  --prompt '请解释一下分布式推理的工作原理。' \
  --max-new-tokens 128
```

如服务端固定了 revision，加上 `--revision 同一个commit`。
服务端和客户端都不指定前缀时，双方从仓库名推导出同一个 DHT 前缀
`Qwen3-6-35B-A3B-petals-qwen-v1`（去掉账号名，点号换成连字符）。
要把不同精度或不同 revision 的 swarm 隔开，给服务端设 `DHT_PREFIX`、
给客户端加 `--dht-prefix`，两边必须一致。
这是 Petals Python 调用，不是 HTTP/OpenAI API；该任务没有添加 HTTP 网关。

## 验证与后续优化

```bash
pip install pytest pytest-asyncio pytest-forked
python -m pytest -o pythonpath=src tests/test_qwen3_5_moe.py -q
PETALS_TEST_LOCAL_SWARM=1 python -m pytest -o pythonpath=src tests/test_qwen_swarm.py -q
```

`tests/data/qwen3_5_moe_reference.npz` 来自官方 v5.5.0 方程，附有源文件 SHA256 和重建脚本。
对照覆盖两种层、跨 64-token chunk 边界的前向和增量推理。网络测试只创建本机私有 CPU 网络。

本次验证（macOS ARM64、Python 3.12.2、PyTorch 2.14.0、Transformers 4.43.1）：

- Qwen 小模型测试 24 项及已有缓存测试 3 项通过。
- 单节点、两节点、只推理模式及自动分层的私有网络测试 4 项通过，包括服务端直连及会话续写。
- 官方 tokenizer.json 的中文分词和聊天模板通过；真实权重索引中两种层的全部参数名称匹配。
- Python 编译检查和启动脚本 Bash 语法检查通过。

上述测试另在 Linux x86_64、Python 3.11.15、PyTorch 2.14.0、Transformers 4.43.1 上复跑通过。

这些是 CPU 验证结果，不替代 Linux CUDA 11.8 / PyTorch 2.2.2 或完整 35B 权重验收。
原有测试清理逻辑在此 macOS 环境退出时产生 multiprocessing resource_tracker 警告；测试退出码为 0。

GPU 验收顺序：检查 FP16 CUDA 运算 → 每卡加载分配的层 → 确认全部 40 层覆盖 →
短文本生成 → 连续对话 → 记录每卡显存、首 token 延迟和输出 token/s → 调整层分配。
之后再考虑融合算子和量化支持。当前不能把 `--quant_type` 改为 NF4 来压缩 MoE 权重：
专家权重是打包的三维参数，旧版 Linear 量化路径并不覆盖它们。
