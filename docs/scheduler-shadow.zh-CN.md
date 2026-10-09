# 调度器影子：骨架

调度器是 `v1/core` 迁移 spike 的第 3 步。它是移植中最大的一块（`vllm/v1/core/sched/scheduler.py`
有 3,474 行），也是唯一无法只靠状态推断正确性的一块：它决定的是**什么时候**发生什么。因此必须先
以影子模式运行——记录 Python 调度器的决策，再与移植版比较——之后才谈切换。

本文记录这套影子的骨架：记录器、轨迹格式，以及一条被记录下来的决策必须满足的不变式。决策逻辑本身
**尚未**移植，因此目前没有可跑的对比：现在有的是 oracle 与契约。

## 记录了哪些东西

`tools/gen_scheduler_trace.py` 用带种子的合成工作负载驱动**真实的** `Scheduler`（经由仓库自带的
夹具 `tests/v1/core/utils.py`，因此用的就是 CPU 测试那套配置），并写出两个文件：

| 文件 | 内容 |
| --- | --- |
| `trace.txt` | 喂给调度器的工作负载与模型回复 |
| `expected.txt` | 调度器给出的决策：每个调度步一行、每个回复一行 |

`trace.txt` 是将来要交给移植版的输入：

```
request r0 48 1
request r1 48 1
...
step r0=1 r1=1 r2=1
step r2=1 r3=1 r4=1
```

每个 `request` 行是一个被调度器接受的请求（prompt 长度与 `max_tokens`）；每个 `step` 行是一次
合成的模型回复，列出模型 runner 为哪些请求上报了 token。这个文件里没有任何“调度器该怎么做”的信息。

`expected.txt` 则是移植版必须产出的内容：

```
schedule 0 -> ok new=r0,r1,r2 cached=- resumed=- tokens=r0:48,r1:16,r2:16 total=80 preempted=- prefix_blocks=2 free=18
update 0 -> ok finished=r0,r1 waiting=r3,r4,r5,r6,r7 running=r2 free=20
schedule 1 -> ok new=r3,r4 cached=r2 resumed=- tokens=r2:1,r3:48,r4:48 total=97 preempted=- prefix_blocks=0 free=13
```

`schedule` 行是要被复现的 `SchedulerOutput`：哪些请求是首次调度、哪些是缓存命中（其中哪些是被抢占后
重新恢复的）、每个请求拿到多少 token、总数、抢占、各 KV cache 组共享的公共前缀块数，以及该决策之后
池的空闲块数。`update` 行是调度器对模型回复的反应：哪些请求结束，以及之后的 waiting 与 running 队列。

工作负载刻意往边界上走而不是停在中间：prompt 长度落在块边界两侧（于是 chunked prefill 会产生多步
prefill 与未填满的块）、`max_tokens` 取 1 让请求立刻结束、一半的 prompt 组共享同一段 prompt（于是前缀
缓存会命中），而驱动脚本既跑一个装不下这一批的池、也跑一个宽裕的池（于是调度器既会抢占、也会恢复）。

## 不变式

只有当一条轨迹里的每个决策都通过下列检查时，它才会被写出。这些检查就是移植版首先要对照的契约——决策
写错的移植版通常会先违反其中一条，而且能直接看出错在哪。

| 检查 | 排除了什么 |
| --- | --- |
| `total-matches-sum` | 总数与各请求 token 数之和不一致 |
| `scheduled-matches-tokens` | 调度了某个请求却没给它 token，或相反 |
| `no-duplicate-request` | 同一步里重复出现同一请求 |
| `positive-token-count` | 空的调度条目 |
| `bounded-token-count` | 给请求调度的 token 超过它还需计算的量 |
| `free-blocks-in-range` | 空闲块数落在 `[0, num_blocks]` 之外 |
| `finished-once` | 已结束的请求被再次调度 |
| `queue-drains` | 工作负载结束时仍有请求在排队 |
| `finished-count` | 请求既没结束也没留下痕迹地消失 |
| `blocks-released` | 块泄漏：结束时池里必须只剩 null 块在用 |

最后一条最强：无论调度器怎么决策，工作负载结束时它取走的每个块都必须回到池里——而且是在前缀缓存开启、
中途发生过驱逐的情况下。`bounded-token-count` 用的是 `schedule()` 调用**之前**的状态，因为
`schedule()` 自己会推进 `num_computed_tokens`。

## 运行方式

```bash
bash tools/check_scheduler_shadow.sh                     # 3 个种子 × 3 种配置
bash tools/check_scheduler_shadow.sh --seeds 5 --num-requests 16
SCHED_SHADOW_MODEL=/path/to/model bash tools/check_scheduler_shadow.sh
```

已验证：3 个种子 × 3 种配置（**共 9 次运行、150 个调度步、90 个请求，零不变式违规**），其中 136 步包含
缓存命中、3 步发生抢占（另有 3 步是被抢占请求恢复）、6 步存在公共前缀块。整个扫描在 CPU 上运行、不碰
GPU：调度器是直接驱动的，没有引擎也没有权重。

## 这还不是什么

* **不是对比**。目前没有任何东西重新推导 `expected.txt`。下一步是给 `trace.txt` 写一个 Rust 读取端，
  它必须逐字节复现该文件——块记账复用已经通过影子校验的 `rust/src/kv-pool`，并以上述不变式作为第一道
  防线。
* **不是吞吐或延迟测量**。工作负载是合成的，模型 runner 是每个请求固定返回一个 token 的桩；采样、
  logprobs、结构化输出与投机解码都不在范围内。
* **不是前缀缓存基准**。共享 prompt 的作用是让调度器走到缓存命中与公共前缀这两条路径，而不是衡量缓存
  效率。

## 运行环境

调度器需要一个真实的模型 **config** 目录：架构与 `max_model_len` 来自它，不加载权重也不使用 GPU。默认
用参考路由的模型（`/data/models/qwen3.8-27b/Qwen3.8-27B-FP8`）；可用 `--model` 或 `SCHED_SHADOW_MODEL`
覆盖，且当该目录不存在时 `tools/validate_all.sh --with-scheduler` 会跳过而不是失败。
