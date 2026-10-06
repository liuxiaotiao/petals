# 真实负载测试:完整流程手册

Qwen3-30B-A3B 在 15 台机器的 Petals 集群上,用 GSM8K / MBPP / No Robots 共 150 条真实请求,
从两条链的 layer 0 节点逐条发起,测 TTFT、生成延迟、p99、吞吐、显存和计算。
这份手册从"集群已关停、权重已删除"的状态讲起,到拿到结果、再收尾释放空间为止。

所有命令都在**控制节点 prin3** 上执行,repo 在 `~/petals`。长时间的步骤放在 tmux 里。

---

## 0. 一页速查

```bash
tmux new -s wl                                  # 或 tmux attach -t wl
cd ~/petals

bash examples/workload_repro.sh status          # 看现状
bash examples/workload_repro.sh all             # 拉起 → 等权重 → 核对 → 跑 150 条(下载视网速 + 测试约 2.3 h)
#   或分步:
bash examples/workload_repro.sh up              # 拉起 15 台 server 并等全部就绪
bash examples/workload_repro.sh check           # 只核对放置和题目
bash examples/workload_repro.sh smoke           # 试跑 6 条 × 64 token
bash examples/workload_repro.sh run r6_shared   # 正式跑,结果在 ~/wl-results/<日期>/r6_shared/

bash examples/workload_repro.sh down            # 用完:停服务、删权重(先列清单再确认)
```

默认复现的是 **r5**(`~/wl-results/2026-10-06/r5_shared_2048`)。换一轮:
`RUN_DIR=~/wl-results/2026-10-05/r4_2048_norobots bash examples/workload_repro.sh all`。

---

## 1. 这套东西由什么组成

### 1.1 机器

| 节点 | GPU | 角色 |
|---|---|---|
| N01–N08 | A30 24 GB | server |
| N09 N10 N11 N13 | Quadro RTX 6000 24 GB | server |
| N12 N14 N15 | T4 16 GB | server,不在测试链上 |
| prin3 | — | 控制节点:跑脚本、调度、出报告 |

N08 能直接上网,其他节点经 N08 的代理出网。N01 是集群内部的 NTP 时间源。

### 1.2 标准设置(r5)

| 项 | 值 |
|---|---|
| 模型 | `Qwen/Qwen3-30B-A3B`,fp16,48 层,版本 `ad44e777bcd18fa416d9da3bd8f70d33ebb85d39` |
| 链 A(layer 0→47) | N01 0:8 · N05 8:16 · N03 16:24 · N07 24:32 · N02 32:40 · N06 40:48 |
| 链 B | N11 0:8 · N08 8:16 · N09 16:24 · N13 24:32 · N04 32:40 · N10 40:48 |
| T4 | 各 7 层,自动放置;测试时被排除 |
| server 参数 | `max_batch_size 2048`、`inference_max_length 4096`、`attn_cache_tokens 65536` |
| client | 在每条链的 layer 0 节点(N01、N11)上;只走本链 6 台;一次一个请求 |
| 题 | gsm8k / mbpp / No Robots(单轮)各 50,seed 0 |
| 生成 | 贪心解码,关闭 thinking,`max_new_tokens 2048` |
| 分配 | 共享队列:哪条链空了就取下一条,每条题只跑一次,整体三类各 50 |

### 1.3 脚本(都在 `examples/`)

| 脚本 | 作用 |
|---|---|
| `workload_repro.sh` | **一键入口**:up / check / smoke / run / all / down / status |
| `workload_run.sh` | 跑一轮:检查 → 记录设置 → GPU 采样 → 两个 client → 报告。`--replay` / `--restore-from` / `--shared` / `--limit` |
| `workload_dispatch.py` | 共享队列调度,跑在 prin3,通过 ssh 给两条链逐条发题 |
| `workload_bench.py` | client:逐 token 计时,每条请求打印一行 `REC {json}`;`--report` 出报告 |
| `workload_snapshot.sh` | 记录集群状态:放置、peer ID、每台 server 参数和版本、代码、题目 |
| `workload_sample.py` | 从 Hugging Face 抽题(只在换题集时用) |
| `workload_split.py` | 把题平分给各链(只在固定平分模式用) |
| `cluster_gpumon.sh` | 每 5 s 采 15 台 GPU 显存和利用率 |
| `qwen_cluster.sh` | 集群管理:deploy / service / status / peers / purge / client … |
| `cluster_disk.sh` | 磁盘缓存清理(pip / conda / HF 缓存) |

### 1.4 文件放在哪

| 位置 | 内容 | 会不会被更新/删除 |
|---|---|---|
| prin3 `~/petals` | repo(脚本、`task/hosts.txt`、`task/workload/*.jsonl`) | 更新 repo 时注意保留 `task/` |
| prin3 `~/petals-env.sh` | 环境变量(MODEL_NAME、DHT_PREFIX、代理等) | 不在 repo 里,不受影响 |
| prin3 `~/wl-results/` | 每轮的结果和设置快照 | r5 已设为只读 |
| 各节点 `~/petals-qwen/repo` | deploy 过去的 repo | `deploy` 覆盖 |
| 各节点 `~/petals-qwen/venv` | Python 环境 | 保留,不要删 |
| 各节点 `~/petals-qwen/cache` | 模型权重(每台约 28 GB) | `down` / `purge` 删除 |
| 各节点 `~/petals-qwen/run/server.env` | 该节点 server 的层号和参数 | `service install` 重写 |

---

## 2. 一次性准备(只在更新脚本后需要)

prin3 上的 repo 需要包含上面那些脚本。两种方式任选:

**a. 用安装包**(在 Mac 上下载后传到 prin3):
```bash
cd ~/petals && bash install_repro_cmd.sh
```

**b. 从 Mac 同步整个 repo**(在 Mac 终端执行,保留 prin3 的 `task/`):
```bash
rsync -av --exclude .git --exclude task/ --exclude '__pycache__' \
  "$HOME/Desktop/cc workspace/petals-main/" ubuntu@<prin3地址>:~/petals/
```

之后若节点上的 client 代码和 prin3 不同,要同步到节点:
```bash
md5sum examples/workload_bench.py
ssh ubuntu@192.168.1.2 md5sum petals-qwen/repo/examples/workload_bench.py   # 不一样就:
bash examples/qwen_cluster.sh deploy
```
`workload_run.sh` 开跑前会自己核对,不一致会直接拒绝并提示 deploy。

---

## 3. 复现一轮

### 3.1 拉起集群:`up`

```bash
bash examples/workload_repro.sh up
```

它按顺序做:

1. 加载 `~/petals-env.sh`,从 `RUN_DIR/settings/MODEL_REVISION` 导出模型版本,保证下载的是当时那一版
2. 把那一轮的 `hosts.txt` 放回 `task/hosts.txt`(原文件备份为 `hosts.txt.bak-<时间>`)
3. `service install`(把层号和模型版本写进每台的 `server.env`)+ `service restart`
4. 每分钟打印一次每台的状态和缓存大小,直到 15 台都在线、没有正在加入的、swarm 可用

权重全删过的话每台要下约 28 GB,多数走 N08 代理,耗时取决于网速;默认最多等 6 小时
(`WAIT_HOURS=10 bash examples/workload_repro.sh up` 可加长)。

想更快:先只让一台下完,再按 `docs/qwen3-30b-a3b.md` 第 5 节的 `seed_span` 用 rsync 铺到其他节点。

### 3.2 核对:`check`

```bash
bash examples/workload_repro.sh check
```

逐台比对 12 台链上节点的实际层号和记录是否一致,并恢复那一轮的题目文件。正常输出:
`placement already matches the snapshot (12 pinned nodes); no restart`。
不一致时它会自己重装服务、重启并等待(最多 20 分钟),仍不一致就报出哪台不对并停下。

### 3.3 试跑:`smoke`

```bash
bash examples/workload_repro.sh smoke
```

6 条、每条 64 token,加上加载模型约几分钟。看到 `6 ok`、`datasets run: gsm8k 2/2 mbpp 2/2 norobots 2/2` 即流程正常。

### 3.4 正式跑:`run`

```bash
bash examples/workload_repro.sh run r6_shared_2048
```

约 2.3 小时。屏幕上每完成一条打一行,例如
`[37/150] (A:21 B:16) B mbpp/test/12: 410 tokens in 152s`。
离开 tmux(`Ctrl-b d`)后看进度:`tail -3 ~/wl-results/<日期>/r6_shared_2048/dispatch.log`。

开跑前它会自动拒绝这些情况:没加载 env、已有 client 在跑、某台节点层号不对、节点上的 client 代码和 prin3 不同。

### 3.5 判断这一轮是否有效

报告末尾应满足:

- `150 ok, 0 skipped, 0 failed`
- `datasets run, all chains together: gsm8k 50/50  mbpp 50/50  norobots 50/50`
- `finish spread between chains` 几分钟以内
- 每 token 时间、TTFT 和 r5 相差在约 3% 以内(r5:326 ms、1.90 s);超出要查原因

---

## 4. 结果目录里有什么

`~/wl-results/<日期>/<名字>/`:

| 文件 | 内容 |
|---|---|
| `wl.A.log` `wl.B.log` | **原始数据**。每条请求一行 `REC {json}`:`id` `dataset` `tag` `prompt_tokens` `submitted` `ttft` `latency` `output_tokens` `gaps`(每个 token 间隔)`stop`(eos/length)`client_cpu_s`,失败时有 `error`。开头 `RUN {json}`(设置),结尾 `END {json}`(总时长、client 内存峰值) |
| `gpu.csv` | 每 5 s、每台:时间戳、节点、已用显存、总显存、利用率 |
| `REPORT.txt` | 汇总:TTFT / 端到端延迟 / 每 token 时间 / token 间隔的 mean·p50·p90·p99;吞吐;按数据集和按链;KV cache、client 内存、每台 GPU 显存;FLOPs、client CPU、每台 GPU 利用率 |
| `DISPATCH.txt` `dispatch.log` | 共享队列:每条链跑了哪些题、结束时间、整体数据集比例、两链结束时间差 |
| `RUN.txt` | 本轮参数:模式、max_new_tokens、链、client 参数、12 台 server 的完整 peer ID、起止时间 |
| `settings/` | 开跑前的快照:`config/hosts.txt`、`cluster/peers.txt`(每台层号)、`cluster/nodes.txt`(每台 GPU、版本、server 启动命令)、`code/`(脚本副本 + md5)、`prompts/`(题目 + md5)、`MODEL_REVISION`、`pip-freeze.*.txt` |

报告可以随时从日志重新生成(用当时那一版脚本):
```bash
D=~/wl-results/<日期>/<名字>
python3 $D/settings/code/workload_bench.py --report $D/wl.A.log $D/wl.B.log --gpu-csv $D/gpu.csv
```

指标口径:

| 指标 | 算法 |
|---|---|
| TTFT | 发出请求到第一个 token,含整条链的 prefill |
| 生成延迟 | 端到端,到 EOS 或 max_new_tokens |
| p99 | nearest-rank;150 条时是第二大的值 |
| 每 token 时间 | (延迟 − TTFT) / (输出 token − 1) |
| 吞吐 | Σ输出 token / Σ请求时间(单用户、单链) |
| KV cache | (prompt + 输出) × 48 层 × 2048 B = 每 token 96 KiB,整条链合计 |
| FLOPs | 2 × 3.3B 激活参数 × token 数(估算,不含 attention) |

没有保存的:生成的文本内容;每一跳在 server 上的耗时。

---

## 5. 用完收尾:`down`

```bash
bash examples/workload_repro.sh down
```

1. `service stop`(必须停 systemd 服务;`qwen_cluster.sh stop` 只杀进程,systemd 会立刻拉起并重新下载)
2. 确认 15 台都是 `DOWN`,否则什么都不删
3. 列出每台要删的权重(约 406 GiB),输入 `yes` 才删

可选,再腾一些空间(都是缓存,删了只是以后重下):
```bash
KEEP="" examples/cluster_disk.sh clean hf conda          # 空跑
KEEP="" examples/cluster_disk.sh clean hf conda --yes    # 约 70 GB
```

不要删:各节点 `~/petals-qwen/venv`(删了要重建,包版本可能变)、prin3 的 `~/petals`、`~/petals-env.sh`、`~/wl-results`。

服务停了但开机自启还在;节点重启会自己拉起 server 并开始下载。要彻底关掉:
`bash examples/qwen_cluster.sh service uninstall`(以后 `up` 会重新安装)。

---

## 6. 做新的实验

改参数,直接加在 `run` 后面(会覆盖记录里的值):
```bash
bash examples/workload_repro.sh run r7_1024 --max-new-tokens 1024
bash examples/workload_repro.sh run r7_split --split            # 固定平分,如 r4
bash examples/workload_repro.sh run r7_try --limit 30           # 只跑 30 条
```

或直接用底层脚本(此时需要先 `source ~/petals-env.sh`):
```bash
bash examples/workload_run.sh --shared --name <名字> --max-new-tokens 2048
```

**换题集**(需要能访问 huggingface.co):
```bash
export HF_TOKEN=hf_...
python3 examples/workload_sample.py --out task/workload/prompts.jsonl --per-dataset 50 --seed 1
python3 examples/workload_split.py task/workload/prompts.jsonl      # 只有固定平分才需要
bash examples/workload_run.sh --shared --name <名字> --max-new-tokens 2048
```
数据集定义在 `workload_sample.py` 的 `SOURCES`;lmsys-chat-1m 仍可用 `--datasets gsm8k mbpp lmsys`(gated,要先在 HF 上接受条款)。

**换放置**:改 `task/hosts.txt` 的 `blocks=起:止`,然后
`bash examples/qwen_cluster.sh service install && bash examples/qwen_cluster.sh service restart`,
等 `status` 显示可用;链的组成用环境变量 `CHAINS="A:N01,N05,... B:N11,..."` 传给 `workload_run.sh`。

每一轮的设置都会自动存进它自己的 `settings/`,之后都能用 `RUN_DIR=<那个目录> … all` 复现。

---

## 7. 常见问题

| 现象 | 原因 / 处理 |
|---|---|
| `MODEL_NAME is not set … Qwen3.6` | 没加载 env。`workload_repro.sh` 会自动加载;直接用其他脚本前先 `source ~/petals-env.sh` |
| `a workload client is already running` | 上一轮的 client 还在。`jobs`、`pgrep -af workload_bench`,以及 `ssh ubuntu@192.168.1.2 pgrep -af workload_bench`,确认后结束它。同一条链两个 client 会让数据作废(r2 就是这样) |
| `placement does not match` / 某台 `is not serving` | 层号漂移或 server 没起来。`bash examples/workload_repro.sh check` 会恢复;仍不行看 `qwen_cluster.sh logs N05`、`diag` |
| `Task size greater than max_batch_size (256)` | server 用的是旧的 `run_qwen_server.sh`。确认 `max_batch_size` 默认 2048,`deploy` 后 `service restart` |
| `status` 里 server 数比机器多 | 重启后旧的 peer ID 记录还没过期,几分钟后消失;不影响(脚本按节点名现取 peer ID) |
| `up` 等了很久 | 在下权重。看每行的缓存大小是否在涨;不涨就 `qwen_cluster.sh diag` |
| 报告里没有 GPU 一节 | `gpu.csv` 没生成;看结果目录里的 `gpumon.log` |
| 抽题时 HTTP 429 / 404 | 429 是限流,脚本会等;404 多半是 HF_TOKEN 没有该数据集权限 |
| 归档目录写不进去 | r5 是只读的;确需修改:`chmod -R u+w <目录>`,改完再 `chmod -R a-w` |
| 文件传不到 prin3 | Mac 上 `pbcopy < 文件`,prin3 上 `cat > 文件` 粘贴后 `Ctrl-D` |

---

## 8. 历次运行

| 目录(prin3 `~/wl-results/`) | 设置 | 状态 |
|---|---|---|
| `2026-10-05/r1_256_lmsys` | 256 token,第三类为 lmsys,每链各跑 150 | 有效;22 条曾因 max_batch_size 256 失败后补跑 |
| `2026-10-05/r2_1024_INVALID_two_clients` | 1024 token | **作废**:每条链两个 client |
| `2026-10-05/r3_1024_norobots` | 1024 token,固定平分 | 有效;2 条截断 |
| `2026-10-05/r4_2048_norobots` | 2048 token,固定平分 | 有效;1 条截断;记录见 `docs/experiments/2026-10-05-sequential-workload.md` |
| `2026-10-06/r5_shared_2048` | 2048 token,**共享队列** | **标准设置**;0 条截断;记录见 `docs/experiments/2026-10-06-shared-queue.md` |

r5 结果:TTFT 平均 1.90 s(p99 4.43 s);生成延迟平均 108.7 s(p99 420.8 s);每 token 326 ms;
每条链 3.02 tok/s;两链结束相差 1.7 分钟,总 137 分钟;GPU 利用率 2–4%,显存约 11 GB / 24 GB。
