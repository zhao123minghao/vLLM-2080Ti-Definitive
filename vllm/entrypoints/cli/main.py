# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The CLI entrypoints of vLLM

Note that all future modules must be lazily loaded within main
to avoid certain eager import breakage."""

import importlib
import importlib.metadata
import sys
from importlib.util import find_spec

from vllm.logger import init_logger

logger = init_logger(__name__)

# All subcommand modules, in the order they are registered for help output.
_SUBCOMMAND_MODULES = (
    "vllm.entrypoints.cli.benchmark.main",
    "vllm.entrypoints.cli.collect_env",
    "vllm.entrypoints.cli.launch",
    "vllm.entrypoints.cli.openai",
    "vllm.entrypoints.cli.run_batch",
    "vllm.entrypoints.cli.serve",
)

# Subcommand name -> the single module that provides it. Importing only that
# module keeps unrelated commands (and their heavy serving imports) out of the
# process; the headless engine path only ever imports `serve`.
_SUBCOMMAND_PROVIDER = {
    "serve": "vllm.entrypoints.cli.serve",
    "bench": "vllm.entrypoints.cli.benchmark.main",
    "launch": "vllm.entrypoints.cli.launch",
    "render": "vllm.entrypoints.cli.launch",
    "collect-env": "vllm.entrypoints.cli.collect_env",
    "run-batch": "vllm.entrypoints.cli.run_batch",
    "chat": "vllm.entrypoints.cli.openai",
    "complete": "vllm.entrypoints.cli.openai",
}


def maybe_run_omni() -> bool:
    # If `--omni` arg is passed to the CLI, delegate to vLLM Omni's entrypoint handling
    if "--omni" not in sys.argv:
        return False

    # NOTE: Check the spec instead of importing directly here, since things could
    # fail with ImportError due to mismatched versions if things are moved around.
    spec = find_spec("vllm_omni")
    if spec is None:
        logger.error(
            "--omni flag requires a valid instance of vllm-omni to be installed."
        )
        sys.exit(1)

    from vllm_omni.entrypoints.cli.main import main as omni_main

    logger.info("Delegating entrypoint handling to vllm-omni")
    omni_main()

    return True


def _subcommand_provider() -> str | None:
    """Return the module providing the requested subcommand, if identifiable.

    `None` means "unknown or not requested", in which case every subcommand
    module is imported so that help output and argument validation stay
    complete.
    """
    for arg in sys.argv[1:]:
        if arg.startswith("-"):
            continue
        return _SUBCOMMAND_PROVIDER.get(arg)
    return None


def main():
    if maybe_run_omni():
        return

    from vllm.entrypoints.cli.types import (
        VLLM_SUBCMD_PARSER_EPILOG,
        cli_env_setup,
    )
    from vllm.utils.argparse_utils import FlexibleArgumentParser

    provider = _subcommand_provider()
    if provider is None:
        cmd_modules = [importlib.import_module(name) for name in _SUBCOMMAND_MODULES]
    else:
        cmd_modules = [importlib.import_module(provider)]

    cli_env_setup()

    benchmark_main = sys.modules.get("vllm.entrypoints.cli.benchmark.main")
    if benchmark_main is not None:
        benchmark_main.maybe_exec_rust_bench()

    # For 'vllm bench *': use CPU instead of UnspecifiedPlatform by default
    if len(sys.argv) > 1 and sys.argv[1] == "bench":
        logger.debug(
            "Bench command detected, must ensure current platform is not "
            "UnspecifiedPlatform to avoid device type inference error"
        )
        from vllm import platforms

        if platforms.current_platform.is_unspecified():
            from vllm.platforms.cpu import CpuPlatform

            platforms.current_platform = CpuPlatform()
            logger.info(
                "Unspecified platform detected, switching to CPU Platform instead."
            )

    parser = FlexibleArgumentParser(
        description="vLLM CLI",
        epilog=VLLM_SUBCMD_PARSER_EPILOG.format(subcmd="[subcommand]"),
    )
    parser.add_argument(
        "-v",
        "--version",
        action="version",
        version=importlib.metadata.version("vllm"),
    )
    subparsers = parser.add_subparsers(required=False, dest="subparser")
    cmds = {}
    for cmd_module in cmd_modules:
        new_cmds = cmd_module.cmd_init()
        for cmd in new_cmds:
            cmd.subparser_init(subparsers).set_defaults(dispatch_function=cmd.cmd)
            cmds[cmd.name] = cmd
    args = parser.parse_args()
    if args.subparser in cmds:
        cmds[args.subparser].validate(args)

    if hasattr(args, "dispatch_function"):
        args.dispatch_function(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
