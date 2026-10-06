# 实验记录:真实负载逐条回放(2026-10-05)

Qwen3-30B-A3B 跑在 15 台机器的 Petals 集群上,两条钉死的完整链各从 layer 0 节点发请求,
一次只有一个请求,上一条结束立刻发下一条。这份记录写清楚当时的全部设置,下次照着能复现。
机器状态的原始快照(每台的 server 参数、版本、peer ID、题目文件)在归档目录的 `settings/` 里,
由 `examples/workload_snapshot.sh` 生成;这份文档是给人读的版本。

最终采用的是 **r4**(生成上限 2048)。

---

## 1. 集群

| 节点 | GPU | 显存 | 角色 |
|---|---|---|---|
| N01–N08 | NVIDIA A30 | 24 GB | server |
| N09 N10 N11 N13 | Quadro RTX 6000 | 24 GB | server |
| N12 N14 N15 | Tesla T4 | 16 GB | server(不在两条测试链上) |
| prin3 | — | — | 控制节点:跑脚本、汇总报告,不承载模型 |

软件:petals 2.3.0.dev2(本仓库)、torch 2.2.2+cu118,15 台相同。
时钟:N01 做内部 NTP(第 5 节),节点间偏差毫秒级。

## 2. 模型与 server 配置

- 模型 `Qwen/Qwen3-30B-A3B`,48 层,fp16,不量化;DHT prefix `Qwen3-30B-A3B-petals-qwen3-moe-v1`
- 由 systemd 用户服务 `petals-qwen` 运行 `examples/run_qwen_server.sh`,环境在各节点
  `~/petals-qwen/run/server.env`(`qwen_cluster.sh service install` 按 `task/hosts.txt` 写入)
- 关键参数:

| 参数 | 值 | 备注 |
|---|---|---|
| `--inference_max_length` | 4096 | 单个请求 prompt + 生成的上限 |
| `--attn_cache_tokens` | 65536 | 默认值 |
| `--max_batch_size` | **2048** | 原来是 256,超过 256 token 的 prompt 会在第一台 server 被拒;r1 之后改的 |
| `--max_chunk_size_bytes` | 16 MiB | |
| `--num_handlers` | 2 | |
| 其他 | `--inference_only --no_auto_relay --quant_type none` | |

## 3. 模型放置

12 台钉死层号(`hosts.txt` 里 `blocks=起:止`,不参与自动均衡),组成两条互不共享 server 的完整链;
3 台 T4 各 7 层、自动放置,只作为多余副本,测试时被 `--allowed-servers` 排除在外。

| 层 | 链 A | 链 B |
|---|---|---|
| 0:8 | **N01**(client 也在这台) | **N11**(client 也在这台) |
| 8:16 | N05 | N08 |
| 16:24 | N03 | N09 |
| 24:32 | N07 | N13 |
| 32:40 | N02 | N04 |
| 40:48 | N06 | N10 |
| 硬件 | 6 × A30 | 2 × A30 + 4 × RTX 6000 |

T4 的位置每次重启可能变(当时 N12 32:39,N14、N15 在 16:24 一带)。
server 没有固定 identity,**每次重启 peer ID 都会变**;脚本按节点名现取 peer ID,所以复现不依赖旧 ID。

## 4. Client 配置

- 用 `qwen_cluster.sh client --node <layer-0 节点>` 在 N01 / N11 上运行 `examples/workload_bench.py`
- `--allowed-servers` = 本链 6 台 server 的 peer ID;路由 `min_latency`;`PETALS_MAX_RETRIES=3`
- client 侧(embedding、lm_head)fp32 跑在 CPU 上
- 每条请求一个 inference session,每次 `generate()` 只生成 1 个 token,从而量到 TTFT 和每个 token 的间隔
- 贪心解码(`do_sample=False`);Qwen3 chat template,**关闭 thinking**(`enable_thinking=False`)
- `--max-new-tokens 2048`(r4),`--max-prompt-tokens 2048`,单条超时 1800 s
- 正式计时前先用 "Hi" 生成 2 个 token 预热,不计入结果

## 5. 题集

用 `examples/workload_sample.py` 生成,`--seed 0 --per-dataset 50`,每个数据集一个独立随机数发生器:

| 数据集 | 来源 | 取法 |
|---|---|---|
| gsm8k | `openai/gsm8k` main/test(1319) | 题目原文 |
| mbpp | `google-research-datasets/mbpp` full/test(500) | 原论文提示格式:任务描述 + 测试用例 |
| norobots | `HuggingFaceH4/no_robots` test(500) | 只取单轮会话(1 条 user + 1 条 assistant);有 system 的保留 system |

再用 `examples/workload_split.py` 分到两条链:每个数据集按文件顺序交替分给 A、B,
每条链里三类题轮流排(gsm8k、mbpp、norobots、gsm8k…)。结果:**每条链 75 条,每类 25 条,两条链不重复,合计 150 条**。

题目文件在归档的 `settings/prompts/`,带 `MD5SUMS`。复现时**用归档里的文件,不要重新抽样**:
数据集在 Hub 上可能更新,重抽不保证一样。

## 6. 运行方式

- 两条链同时跑,各自一次只有一个请求(两条链不共享 server,互不排队)
- `examples/cluster_gpumon.sh` 每 5 s 采一次 15 台的 GPU 显存和利用率
- 分配方式(r4):固定平分,每条链跑自己的 `chain.<TAG>.jsonl`。链 B 慢约 20–26%,所以它比链 A 晚结束
- 之后新增 `--shared`(`examples/workload_dispatch.py`):两条链共用一个队列,哪条链空了就从
  `prompts.jsonl` 取下一条,每条题仍只跑一次,两条链几乎同时结束。**整体**仍是三类各 50 条
  (`DISPATCH.txt` 的 `datasets run` 一行核对);每条链分到多少、比例如何取决于它的速度。
  试跑用 `--limit N`(共享模式下是总共 N 条)。**复现 r4 时不要加 `--shared`**
- 一条命令完成:`examples/workload_run.sh`(检查有没有残留 client、题目是否已同步、取 peer ID、
  记录设置、启动采样和两个 client、出报告)

## 7. 结果(r4,max_new_tokens 2048)

150 条全部成功;149 条自然结束,1 条(`mbpp/test/179`,链 B)写满 2048 token,单条 717 s。

| 指标 | mean | p50 | p90 | p99 |
|---|---|---|---|---|
| TTFT | 1.92 s | 1.76 s | 2.66 s | 4.58 s |
| 生成延迟(端到端) | 112.4 s | 96.7 s | 192.3 s | 326.8 s |
| 每 token 时间 | 333 ms | 317 ms | 384 ms | 389 ms |

| | 值 |
|---|---|
| 吞吐(每条链,单用户) | 2.96 tok/s 输出,3.94 tok/s 含 prompt,0.53 条/分钟;两条链合计约 5.9 tok/s |
| server KV cache / 请求 | 平均 41.5 MiB,最大 202 MiB(整条链合计,每 token 96 KiB) |
| server GPU 显存 | A30 约 11.0 GB / 24 GB,RTX 6000 约 10.9 GB / 23 GB,全程不变 |
| server GPU 利用率 | 平均 A30 2.6%、RTX 6000 1.8%,峰值 17–51%;T4 为 0%(未参与) |
| 计算 | 每条约 2.9 TFLOP(2 × 3.3B 激活参数 × token);实际 0.026 TFLOP/s |
| client | 峰值 RSS 4.0 GB;CPU 每条 177 s,约 157% 个核 |

按数据集:gsm8k 输出 267 token / 延迟 91 s,mbpp 417 / 139 s,norobots 316 / 107 s;
每 token 时间三类都在 331–335 ms,延迟差别几乎全来自输出长度。
按链:A 每 token 295 ms、TTFT 1.79 s;B 372 ms、2.04 s(B 慢约 26%:4 台 RTX 6000,且 client 在 RTX 6000 的 N11 上)。

结论:单用户逐条请求时 GPU 基本空闲,每 token 约 0.33 s 里大部分是 6 跳网络往返和 client 在 CPU 上的计算。

**可重复性**:贪心解码下 r3(1024)和 r4(2048)输出的 token 基本相同,每 token 时间差 2.5%、TTFT 差 1%。
以后改配置,差别要明显大于约 3% 才算真实效果。

## 8. 历次运行与归档(prin3 `~/wl-results/<日期>/`)

| 目录 | 设置 | 状态 |
|---|---|---|
| `r1_256_lmsys` | 256 token,第三类是 lmsys-chat-1m 第一条用户消息,每条链跑全部 150 条 | 有效;22 条因 max_batch_size 256 失败,改 2048 后补跑 |
| `r2_1024_INVALID_two_clients` | 1024 token | **作废**:每条链同时有两个 client,日志互相覆盖 |
| `r3_1024_norobots` | 1024 token,本文的题集和分链 | 有效;2 条写满 1024 |
| `r4_2048_norobots` | 2048 token,本文的题集和分链 | 固定平分的最终结果 |
| `r5_shared_2048`(2026-10-06) | 同上,但两条链共用一个队列 | **之后的标准设置**,见 [2026-10-06-shared-queue.md](2026-10-06-shared-queue.md) |

## 9. 复现

在 prin3 上,repo 根目录,tmux 里:

```bash
source ~/petals-env.sh                      # 一定要先加载;否则 MODEL_NAME 回落到 Qwen3.6
R=~/wl-results/2026-10-05/r4_2048_norobots   # 要复现的那一轮

# 一条命令:恢复放置和题目、核对,然后照原样跑(约 2.5 小时)
bash examples/workload_run.sh --restore-from $R --name r5_2048 --max-new-tokens 2048
```

`--restore-from` 做的事:

1. 把那一轮的 `hosts.txt` 放回 `task/hosts.txt`(原文件备份成 `hosts.txt.bak-<时间>`)
2. 把那一轮的 `prompts.jsonl`、`chain.A/B.jsonl` 放回 `task/workload/`,有变化就 `deploy`
   (用归档的文件,不重新抽样:Hub 上的数据集可能更新)
3. 放置和快照不一致时:`service install` + `service restart`,然后每 30 s 检查一次,
   直到每个钉死的节点都以**新的 peer ID** 服务**记录的层号**、并且 swarm 可用(最多等 20 分钟)
4. 已经一致就不重启,直接往下跑

加 `--pin-all`:3 台 T4 也钉到快照里它们当时的层号,15 台和那一轮完全相同(逐台核对);
不加则和当时一样只钉 12 台,T4 自动放置(不在测试链上,不影响结果)。
只想恢复、不跑:加 `--restore-only`。

不带 `--restore-from` 时,`workload_run.sh` 也会先核对 `hosts.txt` 里每个钉死节点的实际层号,
对不上就拒绝运行。

先想试一下流程:`bash examples/workload_run.sh --restore-from $R --name smoke --max-new-tokens 64 -- --limit 3`。

复现时对照 `$R/settings/cluster/nodes.txt` 里每台的 `server command` 和 `versions`:
server 参数或软件版本不一样,结果就不可比。

## 10. 踩过的坑

- **没加载 env**:`MODEL_NAME is not set` 时 client 会去找 Qwen3.6。`workload_run.sh` 会直接拒绝运行。
- **同一条链上两个 client**:r2 就是这样作废的。`workload_run.sh` 启动前会检查控制节点和两个 layer-0 节点。
- **`max_batch_size` 256**:超过 256 token 的 prompt 在第一台 server 报 `Task size greater than max_batch_size`。
- **重启后 peer ID 变**:旧的 `$A` `$B` 失效;`status` 里几分钟内会同时看到新旧两套记录,旧的会过期。
- **GPU 采样没起来**:直接执行 `examples/cluster_gpumon.sh` 可能没有执行权限,用 `bash` 调用。
- **截断**:256 时 56% 的回答被截断;1024 时 2 条,2048 时 1 条(同一道 mbpp,模型停不下来)。
