# AGENTS.md

This file governs the whole `vLLM 2080 Ti Definitive Edition` repository.

## Project Identity And Credit

This repository is a hardware-focused fork for dual RTX 2080 Ti / SM75 vLLM
serving. It builds on upstream vLLM and preserves the local runtime work,
profiles, documentation, and benchmark evidence needed to reproduce the 2080 Ti
stack.

If you publish, redistribute, repackage, benchmark, or build a derivative from
this repository, keep clear credit to:

- Upstream vLLM and its original license.
- `vLLM 2080 Ti Definitive Edition`.
- The repository author: `github.com/weicj`.

Do not remove existing attribution, license notices, benchmark provenance, or
project identity text. If you maintain a public derivative, state that it is
based on this project unless the code has been independently replaced.

## Upstream Compatibility

This project remains a fork of upstream vLLM. When changing source files that
come from upstream vLLM:

- Preserve upstream license and copyright notices.
- Prefer small, reviewable patches over broad rewrites.
- Keep SM75/Turing-specific behavior guarded or clearly documented.
- Do not present fork-specific behavior as upstream vLLM behavior.
- If an upstream `AGENTS.md` or contribution instruction applies in a copied
  upstream subtree, follow it as well.

## Runtime And Profile Rules

This repository is organized around validated runtime routes, not generic
benchmark guesses.

- Do not invent context-size, throughput, or support claims without evidence.
- Keep profile files focused on route parameters only. Do not store global
  service settings such as GPU selection, port, chat template, or reasoning
  defaults inside route profiles.
- Use the hardware-specific README files under `profiles/` as the source of
  truth for shipped profiles.
- If adding or promoting a profile, include capacity evidence and throughput
  evidence using the repository's documented benchmark口径.
- Do not keep tiny smoke-only profiles as recommended deployment presets.

## Validation Before Publishing

Before committing or publishing changes, run the relevant subset of:

```bash
bash -n build.sh launcher.sh tools/validate_profiles.sh
bash tools/validate_profiles.sh
python3 -m py_compile <changed-python-files>
git diff --check
```

`tools/validate_all.sh` runs exactly those four steps in one command and can pull in the two
optional checks below:

```bash
bash tools/validate_all.sh                 # the four static steps above
bash tools/validate_all.sh --with-parity   # + the cross-language parity checks
bash tools/validate_all.sh --with-chat     # + the chat regression
bash tools/validate_all.sh --full          # all of them
```

Extra non-option arguments are compiled as Python, which is how the
`py_compile <changed-python-files>` step is meant to be used:

```bash
bash tools/validate_all.sh --with-chat vllm/entrypoints/chat_utils.py
```

For changes under `tools/` that touch prefix-cache hashing, or for changes under
`rust/src/kv-hash`, also run the cross-language parity check. It builds the crate, runs the
crate's own unit tests, and then diffs the Rust implementation against Python:

```bash
bash tools/check_prefix_hash_parity.sh
```

For changes under `rust/src/kv-pool`, or to the block bookkeeping in
`vllm/v1/core/kv_cache_utils.py` and `vllm/v1/core/block_pool.py`, run the shadow
verification the same way: Python records a random operation trace and the Rust port has to
replay it into an identical log, including the operations Python rejects.

```bash
bash tools/check_block_pool_parity.sh
```

For changes to model code, kernels, quantization, or the parser/serving path, also run the
chat regression against a live route (it starts and stops the reference route itself):

```bash
bash tools/chat_smoke.sh --launch --quick
```

For dependency, requirement or environment changes (adding, pruning or replacing packages),
re-run the audit and check the measured pin set against it; `check_minimal_set.py` fails when
a distribution the run loaded is missing from the pin file:

```bash
bash tools/dep_audit.sh --out-dir /tmp/dep-audit -- <launcher args...>
python3 tools/dep_audit/check_minimal_set.py --report /tmp/dep-audit/report.json
```

For launcher/profile changes, also verify `launcher.sh --print-config` for the
affected route and mode. For runtime kernel or graph-policy changes, include a
real benchmark or smoke result that proves the changed path still works.

## Documentation Discipline

- Keep English and Simplified Chinese documentation consistent when both exist.
- Keep benchmark numbers tied to the exact model, KV precision, MTP setting,
  context, and benchmark method.
- Restore or update linked assets when moving documentation. Broken benchmark
  figures are treated as documentation regressions.
- Avoid overstating support. Use precise wording such as `validated`,
  `supported`, `experimental`, or `not promoted` according to the evidence.

## Repository Hygiene

- Do not commit local caches, model weights, logs, temporary workspace state,
  run outputs, or generated native build artifacts.
- Keep `README.md`, `README.zh-CN.md`, `CHANGELOG.md`, `VERSION`, and
  `pyproject.toml` version fallback aligned for releases.
- Release tags and GitHub Releases are separate. Pushing a tag is not enough to
  update the GitHub Release page.
