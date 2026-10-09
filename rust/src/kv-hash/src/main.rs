// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//! Command line tool for the `vllm-kv-hash` prefix-cache hashing spike.
//!
//! It exists to prove (and to keep proving) byte-exact parity between the Rust
//! re-implementation and the Python original in `vllm/v1/core/kv_cache_utils.py`, and
//! to measure the CPU cost of the hashing chain that the scheduler runs once per
//! step.
//!
//! Corpus format (produced by `tools/gen_prefix_hash_corpus.py`):
//!
//! ```text
//! <block_size>
//! <token_id> <token_id> ...
//! <token_id> <token_id> ...
//! ```

use std::process::ExitCode;
use std::time::Instant;

use vllm_kv_hash::{
    DEFAULT_NONE_HASH_SEED, HashAlgo, HashValue, hash_block_tokens, hash_blocks, none_hash,
};

const USAGE: &str = "\
usage: vllm-kv-hash <command> [options]

commands:
  none-hash --algo <sha256|sha256_cbor> [--seed <seed>]
        Print NONE_HASH (hex) exactly as init_none_hash() would derive it.

  hash --algo <algo> --corpus <file> [--extra <keys>] [--limit <n>]
        Print the chained block hash (hex, one per line) for every block.

  payload --algo <algo> --corpus <file> [--extra <keys>] [--limit <n>]
        Print the serialized hash payload (hex) for the first blocks, for
        debugging serialization mismatches.

  bench --algo <algo> --corpus <file> [--trials <n>] [--extra <keys>]
        Time the hashing chain and report tokens/s.

options:
  --extra <keys>  comma separated extra_keys, e.g. `1234` or `s:mmhash`;
                  defaults to None (matches the upstream benchmark)
";

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match run(&args) {
        Ok(()) => ExitCode::SUCCESS,
        Err(message) => {
            eprintln!("error: {message}");
            eprintln!("{USAGE}");
            ExitCode::FAILURE
        }
    }
}

struct Args {
    algo: HashAlgo,
    corpus: Option<String>,
    extra: Option<HashValue>,
    seed: String,
    limit: Option<usize>,
    trials: usize,
}

fn run(args: &[String]) -> Result<(), String> {
    let (command, rest) = args.split_first().ok_or("missing command")?;
    if command == "-h" || command == "--help" || command == "help" {
        print!("{USAGE}");
        return Ok(());
    }
    let parsed = parse_args(rest)?;

    match command.as_str() {
        "none-hash" => {
            let hash = none_hash(parsed.algo, &parsed.seed);
            println!("{}", hex::encode(hash));
            Ok(())
        }
        "hash" | "payload" | "bench" => {
            let path = parsed.corpus.as_deref().ok_or("--corpus is required")?;
            let blocks = read_corpus(path)?;
            match command.as_str() {
                "hash" => {
                    let hashes = hash_blocks(
                        parsed.algo,
                        &none_hash(parsed.algo, &parsed.seed),
                        &blocks,
                        parsed.extra.as_ref(),
                    );
                    let limit = parsed.limit.unwrap_or(hashes.len()).min(hashes.len());
                    for hash in &hashes[..limit] {
                        println!("{}", hex::encode(hash));
                    }
                    Ok(())
                }
                "payload" => {
                    let none = none_hash(parsed.algo, &parsed.seed);
                    let limit = parsed.limit.unwrap_or(4).min(blocks.len());
                    let mut parent: Option<Vec<u8>> = None;
                    for block in &blocks[..limit] {
                        let parent_ref = parent.as_deref();
                        let value =
                            HashValue::block(&none, parent_ref, block, parsed.extra.as_ref());
                        println!("{}", hex::encode(parsed.algo.serialize(&value)));
                        // Chain like hash_block_tokens() does, so the payload printed for
                        // the next block carries the real parent hash.
                        parent = Some(
                            hash_block_tokens(
                                parsed.algo,
                                &none,
                                parent_ref,
                                block,
                                parsed.extra.as_ref(),
                            )
                            .to_vec(),
                        );
                    }
                    Ok(())
                }
                _ => {
                    let none = none_hash(parsed.algo, &parsed.seed);
                    let tokens: usize = blocks.iter().map(Vec::len).sum();
                    let mut best = f64::MAX;
                    let mut total = 0.0;
                    for _ in 0..parsed.trials {
                        let start = Instant::now();
                        let hashes =
                            hash_blocks(parsed.algo, &none, &blocks, parsed.extra.as_ref());
                        let elapsed = start.elapsed().as_secs_f64();
                        std::hint::black_box(&hashes);
                        best = best.min(elapsed);
                        total += elapsed;
                    }
                    let trials = parsed.trials as f64;
                    println!(
                        "algo={} blocks={} tokens={} best={:.6}s avg={:.6}s tokens_per_second={:.0}",
                        parsed.algo.name(),
                        blocks.len(),
                        tokens,
                        best,
                        total / trials,
                        tokens as f64 / best
                    );
                    Ok(())
                }
            }
        }
        other => Err(format!("unknown command `{other}`")),
    }
}

fn parse_args(args: &[String]) -> Result<Args, String> {
    let mut algo = HashAlgo::Sha256;
    let mut algo_set = false;
    let mut corpus = None;
    let mut extra = None;
    let mut seed = DEFAULT_NONE_HASH_SEED.to_string();
    let mut limit = None;
    let mut trials = 5;
    let mut iter = args.iter();
    while let Some(arg) = iter.next() {
        match arg.as_str() {
            "--algo" => {
                let name = iter.next().ok_or("--algo needs a value")?;
                algo = HashAlgo::from_name(name).ok_or(format!(
                    "unsupported algorithm `{name}` (spike covers sha256 and sha256_cbor)"
                ))?;
                algo_set = true;
            }
            "--corpus" => corpus = Some(iter.next().ok_or("--corpus needs a value")?.clone()),
            "--extra" => extra = Some(parse_extra(iter.next().ok_or("--extra needs a value")?)?),
            "--seed" => seed = iter.next().ok_or("--seed needs a value")?.clone(),
            "--limit" => {
                limit = Some(
                    iter.next()
                        .ok_or("--limit needs a value")?
                        .parse::<usize>()
                        .map_err(|e| format!("bad --limit: {e}"))?,
                );
            }
            "--trials" => {
                trials = iter
                    .next()
                    .ok_or("--trials needs a value")?
                    .parse::<usize>()
                    .map_err(|e| format!("bad --trials: {e}"))?;
            }
            other => return Err(format!("unknown option `{other}`")),
        }
    }
    if !algo_set {
        return Err("--algo is required".to_string());
    }
    Ok(Args {
        algo,
        corpus,
        extra,
        seed,
        limit,
        trials: trials.max(1),
    })
}

/// Parse `--extra` into the `tuple[Any, ...]` vLLM passes as `extra_keys`.
fn parse_extra(spec: &str) -> Result<HashValue, String> {
    let mut keys = Vec::new();
    for raw in spec.split(',') {
        let raw = raw.trim();
        if raw.is_empty() {
            return Err("empty --extra key".to_string());
        }
        if let Some(text) = raw.strip_prefix("s:") {
            keys.push(HashValue::Str(text.to_string()));
        } else if let Some(text) = raw.strip_prefix("b:") {
            keys.push(HashValue::Bytes(text.as_bytes().to_vec()));
        } else {
            keys.push(HashValue::Int(
                raw.parse::<i64>().map_err(|e| format!("bad --extra key `{raw}`: {e}"))?,
            ));
        }
    }
    Ok(HashValue::Tuple(keys))
}

fn read_corpus(path: &str) -> Result<Vec<Vec<u32>>, String> {
    let content = std::fs::read_to_string(path).map_err(|e| format!("{path}: {e}"))?;
    let mut lines = content.lines();
    let block_size: usize = lines
        .next()
        .ok_or("empty corpus")?
        .trim()
        .parse()
        .map_err(|e| format!("bad block size header: {e}"))?;

    let mut blocks = Vec::new();
    for (index, line) in lines.enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        let mut tokens = Vec::with_capacity(block_size);
        for token in line.split_whitespace() {
            tokens.push(
                token
                    .parse::<u32>()
                    .map_err(|e| format!("line {}: bad token id `{token}`: {e}", index + 2))?,
            );
        }
        if tokens.len() != block_size {
            return Err(format!(
                "line {}: expected {block_size} token ids, found {}",
                index + 2,
                tokens.len()
            ));
        }
        blocks.push(tokens);
    }
    if blocks.is_empty() {
        return Err("corpus has no blocks".to_string());
    }
    Ok(blocks)
}
