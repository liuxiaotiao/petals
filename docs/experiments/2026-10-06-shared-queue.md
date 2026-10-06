# 实验记录:共享队列逐条回放(2026-10-06,r5)

和 [2026-10-05 的 r4](2026-10-05-sequential-workload.md) 是同一套集群、同一放置、同一份 150 条题、
同样的 client 设置;**唯一的区别是分配方式**:r4 固定平分(每链 75 条),r5 两条链共用一个队列,
哪条链空了就取下一条,每条题只跑一次。目的:两条链同时结束,不让快的链干等。

**这是目前的标准设置。** 一键使用(`examples/workload_repro.sh`,默认就是 r5;在 tmux 里跑):

```bash
bash examples/workload_repro.sh all      # 拉起 server(同一放置、同一模型版本 ad44e777…)→ 等权重就绪 → 核对 → 复现
bash examples/workload_repro.sh smoke    # 先试 6 条、64 token
bash examples/workload_repro.sh down     # 用完:停服务(systemd)并删各节点权重,会先列出要删的再确认
```

分步:`up`(拉起并等所有节点就绪,默认最多等 6 小时)、`check`(只核对)、`run [名字]`、`status`。
换一次运行:`RUN_DIR=~/wl-results/2026-10-05/r4_2048_norobots bash examples/workload_repro.sh all`。

2026-10-06 r5 结束后权重已删除(406 GiB);当时的模型版本记在 `settings/MODEL_REVISION`
(`ad44e777bcd18fa416d9da3bd8f70d33ebb85d39`),包清单在 `settings/pip-freeze.N01.txt`,venv 保留未动。

底层命令(`workload_repro.sh` 就是按这个顺序调用的):

```bash
cd ~/petals && source ~/petals-env.sh
bash examples/workload_run.sh --replay ~/wl-results/2026-10-06/r5_shared_2048 --name <新名字>
```

`--replay` 从该目录的 `RUN.txt` 读出模式(共享队列)、`max_new_tokens 2048`、两条链、client 参数,
并像 `--restore-from` 一样先恢复并核对放置和题目文件,再开跑。命令行上另给的参数优先。

---

## 设置(与 r4 相同的部分见 10-05 文档第 1–5 节)

| 项 | 值 |
|---|---|
| 模型 / 精度 | Qwen/Qwen3-30B-A3B,fp16,48 层;DHT prefix `Qwen3-30B-A3B-petals-qwen3-moe-v1` |
| server | `max_batch_size 2048`、`inference_max_length 4096`、`attn_cache_tokens 65536` |
| 链 A(layer 0 → 47) | N01 N05 N03 N07 N02 N06(6 × A30),client 在 N01 |
| 链 B | N11 N08 N09 N13 N04 N10(2 × A30 + 4 × RTX 6000),client 在 N11 |
| T4(N12 N14 N15) | 自动放置,被 `--allowed-servers` 排除 |
| 题 | `prompts.jsonl`:gsm8k、mbpp、No Robots(单轮)各 50,seed 0 |
| 生成 | 贪心,关闭 thinking,`max_new_tokens 2048`,每条链一次一个请求 |
| **分配** | **共享队列**(`workload_dispatch.py`):队列按 gsm8k、mbpp、norobots 轮流排;client 以 `--worker` 模式从 stdin 逐条接题 |

## 结果

150 条全部成功、**全部自然结束(0 条截断)**;整体三类各 50 条。

| 指标 | mean | p50 | p90 | p99 |
|---|---|---|---|---|
| TTFT | 1.90 s | 1.68 s | 2.56 s | 4.43 s |
| 生成延迟(端到端) | 108.7 s | 91.9 s | 184.4 s | 420.8 s |
| 每 token 时间 | 326 ms | 306 ms | 387 ms | 396 ms |

| | 值 |
|---|---|
| 分配 | 链 A 87 条,链 B 63 条 |
| 结束时间差 | **1.7 分钟**;总时长 136.9 分钟 |
| 吞吐(每条链,单用户) | 3.02 tok/s,0.55 条/分钟 |
| 按链 | A:每 token 293 ms、TTFT 1.59 s;B:371 ms、2.33 s |
| 按数据集 | gsm8k 264 token / 86 s;mbpp 403 / 137 s;norobots 317 / 103 s |
| KV cache / 请求 | 平均 41 MiB,最大 198 MiB |
| GPU | 利用率平均 A30 3.3–4.2%、RTX 6000 约 2.1%;显存约 11 GB / 24 GB 不变;T4 0% |
| client | 峰值 RSS 4.0 GB;CPU 约 156% 个核 |

和 r4 比:每 token 时间(326 vs 333 ms)、TTFT(1.90 vs 1.92 s)在约 3% 的波动范围内,单请求性能没变。
总时长 137 分钟,r4 最晚的链 B 光请求时间就约 158 分钟,快了约 13%。
r4 里写满 2048 的 `mbpp/test/179` 这次自然结束了;贪心解码在不同 GPU 上数值略有差别,可能走出不同的结果。
没有截断,所以 p99 420 s 是真实最长回答的时间。

## 归档

prin3 `~/wl-results/2026-10-06/r5_shared_2048/`:

| 文件 | 内容 |
|---|---|
| `wl.A.log` `wl.B.log` | 每条请求一行 `REC {json}`:题号、数据集、prompt/输出 token 数、TTFT、端到端、每个 token 间隔、停止原因、client CPU;开头 `RUN`、结尾 `END` |
| `gpu.csv` | 15 台 GPU 每 5 s 的显存和利用率 |
| `DISPATCH.txt` `dispatch.log` | 每条链跑了哪些题(按顺序)、结束时间、整体数据集比例、结束时间差 |
| `REPORT.txt` | 汇总报告(可用 `settings/code/workload_bench.py --report` 从日志重新生成) |
| `RUN.txt` | 本次参数:模式、max_new_tokens、链、client 参数、12 台 server 的完整 peer ID、起止时间 |
| `settings/` | 开跑前的快照:hosts.txt、各节点层号与 peer ID、每台 server 的参数和软件版本、代码副本、题目文件,均带 md5 |
