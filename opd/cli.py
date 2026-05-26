from __future__ import annotations

import argparse
from pathlib import Path

from opd.config import load_config, validate_config
from opd.runtime.local_driver import LocalDriver


def _export_explorer(run: Path, out: Path) -> None:
    try:
        from opd.export.explorer import export_run
    except ImportError as exc:
        raise NotImplementedError("export-explorer is not implemented yet") from exc
    export_run(run, out)


def main() -> None:
    parser = argparse.ArgumentParser(prog="opd")
    sub = parser.add_subparsers(dest="cmd", required=True)

    train = sub.add_parser("train")
    train.add_argument("--config", type=Path, required=True)

    exp = sub.add_parser("export-explorer")
    exp.add_argument("--run", type=Path, required=True)
    exp.add_argument("--out", type=Path, required=True)

    args = parser.parse_args()

    if args.cmd == "train":
        cfg = load_config(args.config)
        validate_config(cfg)
        if cfg.runtime == "local":
            LocalDriver(cfg).run()
        else:
            raise NotImplementedError("ray runtime is M2")
    elif args.cmd == "export-explorer":
        _export_explorer(args.run, args.out)
    else:
        raise AssertionError(f"unknown command: {args.cmd}")


if __name__ == "__main__":
    main()
