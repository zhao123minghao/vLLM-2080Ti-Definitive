// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//! Command line tool for the `vllm-kv-pool` block bookkeeping spike.
//!
//! It replays an operation trace through the Rust port of `FreeKVCacheBlockQueue` and
//! `BlockHashToBlockMap` and prints one log line per operation. The Python recorder
//! (`tools/gen_block_pool_trace.py`) writes the same log for the real vLLM classes, so
//! `diff` between the two is the correctness argument — see
//! `tools/check_block_pool_parity.sh`.
//!
//! Trace format (one operation per line, `#` starts a comment):
//!
//! ```text
//! init 16
//! popleft
//! popleft_n 3
//! append 7
//! prepend_n 4 1
//! remove 5
//! peek
//! num_free
//! map_insert a1 3
//! map_pop a1 3
//! map_contain a1 3
//! map_get_one a1
//! map_len
//! map_dump
//! ```

use std::fs;
use std::process::ExitCode;

use vllm_kv_pool::replay;

const USAGE: &str = "\
usage: vllm-kv-pool <command> [options]

commands:
  replay --trace <file>
        Replay an operation trace and print the resulting log on stdout.

  selftest
        Replay a short built-in trace; handy when wiring the tool up.
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

fn run(args: &[String]) -> Result<(), String> {
    let (command, rest) = args.split_first().ok_or("missing command")?;
    if command == "-h" || command == "--help" || command == "help" {
        print!("{USAGE}");
        return Ok(());
    }

    match command.as_str() {
        "replay" => {
            let mut trace_path: Option<&str> = None;
            let mut index = 0;
            while index < rest.len() {
                match rest[index].as_str() {
                    "--trace" => {
                        trace_path = Some(rest.get(index + 1).ok_or("--trace needs a value")?);
                        index += 2;
                    }
                    other => return Err(format!("unknown option `{other}`")),
                }
            }
            let path = trace_path.ok_or("--trace is required")?;
            let trace = fs::read_to_string(path).map_err(|e| format!("{path}: {e}"))?;
            print!("{}", replay(&trace)?);
            Ok(())
        }
        "selftest" => {
            let trace = "init 3\nnum_free\npopleft\npeek\nmap_insert a1 5\nmap_dump\n";
            print!("{}", replay(trace)?);
            Ok(())
        }
        other => Err(format!("unknown command `{other}`")),
    }
}
