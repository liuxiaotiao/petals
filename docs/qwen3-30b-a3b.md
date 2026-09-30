# 在同一套集群上跑 Qwen3-30B-A3B

`qwen3_moe` 适配器已经加进来了。它和 Qwen3.6 那个 (`qwen3_5_moe`) 是两套独立的适配器,
共用同一批脚本、同一套 systemd 单元、同一个控制节点流程。

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
8 层 span 上一个会话只要 `2048 字节 × 层数 × (prompt + 生成)`:

```
并发会话数 = ATTN_CACHE_TOKENS ÷ (prompt + 生成)
```

按现在的默认值 65536:2048+2048 的会话能并发 16 个,1024+1024 能 32 个,128+128 能 256 个。
8 层 span 的预算是 1.0 GiB,只有 Qwen3.6 的一半。回退会话也不用重放——K/V 切一刀就行。

**head_dim 是 128,不是 hidden_size ÷ 头数(那是 64)。** Petals 通用的缓存路径按后者算,
会只分配一半。所以这个适配器和 Qwen3.6 一样走 `petals_custom_cache`,用自己的
`cache_specs`。`tests/test_qwen3_moe.py::test_cache_is_sized_from_head_dim_not_hidden_size`
盯着这一条。

**专家在 checkpoint 里是分开存的。** Hub 上是
`model.layers.N.mlp.experts.{0..127}.{gate,up,down}_proj.weight`,每层 384 个张量。
Petals 按参数名精确匹配加载,没有转换步骤,所以适配器保留了每个专家一个 `nn.Linear`,
没有用新版 Transformers 的打包写法。改成打包的话,每一个名字都会对不上分片索引,
一个 block 都加载不了。

## 怎么跑

DHT 前缀带 `-petals-qwen3-moe-v1` 后缀,和 Qwen3.6 的 swarm 天然隔离,
两个模型可以先后跑、也可以分别用不同的 `HOSTS_FILE` 同时跑在不同机器上。

先把旧模型的权重清掉,否则它会一直占着 `MAX_DISK_SPACE` 的额度,
新模型每下一个分片都得先挤掉它一个:

```bash
bash examples/qwen_cluster.sh stop
bash examples/qwen_cluster.sh purge              # 先看一眼要删多少
bash examples/qwen_cluster.sh purge --yes
```

```bash
source ~/petals/env.sh
export MODEL_NAME=Qwen/Qwen3-30B-A3B
export MAX_DISK_SPACE=30GB          # 8 层的最坏分片占用约 15.3 GB

bash examples/qwen_cluster.sh preflight       # 会用新的 MODEL_NAME 探 Hub
bash examples/qwen_cluster.sh plan --cap 8    # 自动按 48 层和这个模型的块大小算
```

`plan` 现在知道自己在给哪个模型排:它按 `--model`(默认取 `$MODEL_NAME`)和
`--attn-cache-tokens`(默认取 `$ATTN_CACHE_TOKENS`)算每块占多少显存,不再用写死的常数。
24 GB 的卡按 65536 的缓存预算能放 15 层,T4 能放 9 层;15 台每台 8 层就是 120 个层位盖 48 层,
**2.5 份副本**。

```bash
bash examples/qwen_cluster.sh deploy
bash examples/qwen_cluster.sh service install    # 每行确认 cache=65536/4096
bash examples/qwen_cluster.sh service restart
bash examples/qwen_cluster.sh status --watch     # 等 0-48 全覆盖
bash examples/qwen_cluster.sh client --prompt '你好'
```

**副本多了,路由才真正有的选。** Qwen3.6 是 1 份多一点的覆盖,每层基本只有一个服务端,
所以并发全压在同一条路由上。这个模型 2.5 份覆盖,配合修好的 `cache_tokens_left`
(见 runbook 4.5),客户端能在副本之间摊开。

## 不支持的

配置里出现这些会在启动时直接报错,而不是跑出错结果:

- `rope_scaling` / `rope_parameters` 里非 `default` 的 RoPE(YaRN 等长上下文扩展)
- 滑动窗口注意力(`use_sliding_window`)
- `mlp_only_layers` 非空,或 `decoder_sparse_step != 1`——这种 checkpoint 有两种块大小,
  Petals 会按先看到的那种给整个 swarm 排层,宁可拒绝也不要排错
- 预量化的 checkpoint、prompt tuning、LoRA、beam search、投机解码

`Qwen3-235B-A22B` 的结构参数同属这一类(`qwen3_moe`,94 层),配置检查会放行,
但每层更大、层数更多,得重新 `plan`。

## 验证到哪一步了

- 逐层数值对拍:和未经修改的 Transformers v4.51.0 `Qwen3MoeDecoderLayer` 比,
  长度 7 和 67 两组**逐位相同**(fixture 在 `tests/data/`,用
  `tests/make_qwen3_moe_reference.py` 重新生成)
- 另外和 Transformers 5.17 的打包实现独立对过一次,同样逐位相同
- 增量解码、会话回退、RoPE 表增长:和一次性 forward 相差 2.4e-7 以内
- 参数名与 Hub 分片索引的键集合逐一比对
- 缓存记账、配置注册与拒绝路径:`tests/test_qwen3_moe.py` 共 27 个用例全过

**没验证的:真权重端到端没跑过。** 这里没有 GPU,也没下那 61 GB。
第一次真机部署请先 `client --prompt` 看一句中文对不对,再上 `bench`。
