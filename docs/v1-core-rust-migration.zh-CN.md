# `v1/core` Rust 迁移可行性验证

状态：**范围内两条 CPU 路径均已验证完成、一致性已证明，但不建议把它当作性能项目推进。**
范围：`vllm/v1/core` 中调度器每步执行的两条 CPU 路径——前缀缓存哈希链与 KV 缓存块簿记。本文记录实测数据，避免以后重复论证。

## 已迁移的内容

`rust/src/kv-hash`（`vllm-kv-hash`）重新实现了：

- `vllm.v1.core.kv_cache_utils.hash_block_tokens(hash_fn, parent_hash, token_ids, extra_keys)`
- `vllm.v1.core.kv_cache_utils.init_none_hash(hash_fn)` / `NONE_HASH`
- `benchmarks/benchmark_prefix_block_hash.py` 使用的链式驱动循环

Python 侧自带的两种算法都被**逐字节**复现，包括序列化步骤——因为哈希的输入是一个被序列化的 Python 元组：

| 算法 | 序列化方式 | 摘要 |
| --- | --- | --- |
| `sha256` | CPython pickle 协议 5（vLLM 默认） | SHA-256 |
| `sha256_cbor` | 规范 CBOR，`cbor2.dumps(..., canonical=True)` | SHA-256 |

`xxhash` / `xxhash_cbor` **没有**移植：它们依赖 `xxhash.xxh3_128_digest`，而本仓库的离线 cargo 源中没有提供 XXH3-128 的 crate（`twox-hash` 只实现 XXH32/XXH64）。对于已支持的算法，Python 的种子规则被完整保留：设置了 `PYTHONHASHSEED` 时使用它，否则使用加密类算法固定的 `vllm-none-hash` 种子。

## 一致性证据

`tools/check_prefix_hash_parity.sh` 以**真实 Python 实现**为准绳：`tools/gen_prefix_hash_corpus.py` 生成 token 块语料，向 vLLM 询问期望的块哈希，然后 Rust 二进制必须复现它们，脚本对两侧结果做 diff。

```bash
bash tools/check_prefix_hash_parity.sh --num-blocks 200
# -> prefix_hash_parity_ok total=18
```

18/18 全部通过：2 种算法 × 3 种 `extra_keys` 形态（`None`、多模态哈希使用的整数元组、字符串元组）× {序列化 payload 字节、完整 200 块的链式哈希}，另有带/不带固定 `PYTHONHASHSEED` 的 `NONE_HASH` 推导。

逐字节复现 CPython pickle 输出是本次最大的不确定项；编码器既由单元测试中的黄金字节锁定（`cargo test -p vllm-kv-hash` 共 7 个测试），也由端到端一致性运行锁定。

## 实测成本

同一语料（10,000 块 × 32 token，token id 取自 248,320 词表），5 次计时取最优：

| 算法 | Python（真实 vLLM） | Rust | 加速比 |
| --- | --- | --- | --- |
| `sha256` | 16.06 M token/s | 43.67 M token/s | 2.7 倍 |
| `sha256_cbor` | 2.77 M token/s | 45.41 M token/s | 16.4 倍 |

单块（32 token）：Python 1.99 µs，Rust 0.73 µs。

两个值得记录的发现：

- 在 Python 中**默认的 pickle 路径比 CBOR 路径快 5.8 倍**，因为 `pickle.dumps` 是 C 实现而 `cbor2.dumps(canonical=True)` 不是。把 Python 部署切到 `--prefix-caching-hash-algo sha256_cbor` 会付出真实 CPU 成本；而在 Rust 中两种序列化都是约 45 M token/s。
- 若将来需要跨语言一致，应统一到 `sha256_cbor`（规范 CBOR 有明确定义）；但要匹配既有部署的缓存键，必须复现默认的 `sha256` 路径。

## 块簿记：基于状态的那一半移植

哈希是无状态的，块池不是。`rust/src/kv-pool`（`vllm-kv-pool`）移植了调度器每步都会改动的
那部分状态：

* `FreeKVCacheBlockQueue` —— 空闲块的双向链表，包括伪造的头尾哨兵。Python 把链表指针穿在
  `KVCacheBlock` 对象上；移植版把同样的指针放进按下标寻址的表，这是 Rust 下更自然的形状。
  失败模式也是契约的一部分，且被如实复现：空表 `popleft()` 报 `ValueError`、`popleft_n()`
  超出空闲数报 `AssertionError`、`remove()` 一个不在链表里的块报 `RuntimeError`。
* `BlockHashToBlockMap` —— 一个哈希键对应单个块或多个块（按插入顺序）。移植版与可观测行为
  一致，包括：键只在其最后一个块被弹出时消失；失败的 `pop()` 保持键存在（且移到键序末尾，
  与 Python 删除再插回的效果一致）。

`tools/check_block_pool_parity.sh` 把哈希那套方法变成了生成器：

```bash
bash tools/check_block_pool_parity.sh
bash tools/check_block_pool_parity.sh --seeds 6 --num-ops 1500 --num-blocks 32
# -> block_pool_parity_ok traces=6 operations=9030
```

已验证：6 条轨迹、每条 1,505 步（**共比较 9,030 步，其中 522 步被 Python 拒绝**），日志逐行
一致，包括每次拒绝所报的异常类型。crate 自带 9 个单测锁住 diff 看不到的部分——被弹出的块确实不再
持有任何指针、以及映射只在最后一个块被弹出时才删键。

`tools/gen_block_pool_trace.py` 用带种子的随机操作序列驱动**真实的** Python 类，同时写出操作
轨迹与 Python 产生的日志（每步一行）——包括被 Python 拒绝时的 `err <异常类名>`。Rust 二进制
重放同一轨迹，两份日志做 diff。扫多个种子是它区别于单元测试的关键：真正考验重写是否漂移的
地方，恰恰是那些**拒绝路径**也能一一对应。

尚未覆盖的部分：`BlockPool` 本身（块事件、部分块哈希、引用计数、`BlockHashListWithBlockSize`）
以及其上的调度器决策。已移植的是这些上层逻辑所依赖的数据结构。

## 为什么这不是一个性能项目

哈希只是 `vllm/v1/core` 中 Python 记账工作的一部分。另一个每步都会用到的原语是块池；用仓库自带基准实测（`benchmarks/benchmark_block_pool.py`，100,000 块，1,000 次迭代）：

| 操作 | Python |
| --- | --- |
| `get_new_blocks`（10 块） | 3.85 µs |
| `get_new_blocks`（100 块） | 32.8 µs |
| `free_blocks`（100 块） | 15.4 µs |

即每个已分配块约 0.32 µs、每个已释放块约 0.15 µs。即使批次很大，一个解码步的块池 + 哈希开销也只有**数百微秒**，而一次 GPU 步进是数十毫秒。解码步在请求新生成的 token 填满一个新块之前根本不哈希，而且 Python 只哈希此前未哈希过的块。

因此诚实的结论是：把 `v1/core` 迁到 Rust 能让 Python 退出引擎控制环、缩小运行时 Python 面积、让 Rust 前端直接掌握调度决策——但**不会**体现在吞吐、步进延迟或启动时间上。启动时间由引擎初始化和 CUDA 图捕获主导，与这些无关。

## 如果仍要推进

建议顺序如下：

1. **块池**（`block_pool.py`、`FreeKVCacheBlockQueue`、`KVCacheBlock`、`BlockHashListWithBlockSize`）：调度器每步都会改动的可变状态。其中 `FreeKVCacheBlockQueue` 与 `BlockHashToBlockMap` 已移植并通过影子校验（见上文）；剩下的是 `BlockPool` 本身，它的事件、部分块哈希与引用计数都建在这两个结构之上。真正的语义风险在这里，不在哈希。
2. **调度决策**（`sched/scheduler.py`，自身 3,474 行）：必须先以影子模式运行，在同一条流量上把决策序列与 Python 调度器对比，之后才谈切换。
3. 把 Rust 侧调度器放到 `managed-engine` / M3–M4 的进程工作之后，最后才把 Python 收缩为执行已决定批次的 worker。

注意 `v1/core` 中只有 `sched/output.py` 导入 torch，因此该包其余部分没有计算依赖，可以在不触碰 torch 路径的前提下迁移。

## 复现上述数字

```bash
export PATH="$PWD/.venv/bin:$PATH"
bash tools/check_prefix_hash_parity.sh --num-blocks 200   # 同时会跑 crate 单测

python3 tools/gen_prefix_hash_corpus.py --out-dir /tmp/hashbench \
  --num-blocks 10000 --block-size 32 --vocab-size 248320 --seed 0 --bench-trials 5
./rust/target/release/vllm-kv-hash bench --algo sha256 --corpus /tmp/hashbench/corpus.txt --trials 5
./rust/target/release/vllm-kv-hash bench --algo sha256_cbor --corpus /tmp/hashbench/corpus.txt --trials 5

cd benchmarks && python3 benchmark_block_pool.py --num-gpu-blocks 100000 --num-iteration 1000
```

`vllm-kv-hash` 二进制是开发工具：不会与 `vllm/vllm-rs` 一起安装，运行时路径也不依赖它。
