## \file check_deployment.py
# \brief Operator-side import, configuration and shared-path deployment checks.
"""This check starts no worker, opens no INDI connection and changes no controller state."""

import argparse
import importlib
import json
import os
from pathlib import Path
import sys
import tempfile
from .app import TrainerConfig
from .configuration import load_training, revision


def main():
    """Validate the installed package/configuration as the intended runtime account."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="/opt/MagAOX/config/po4aoTrainer.conf")
    args = parser.parse_args()
    config = TrainerConfig.from_config(config_path_or_paths=args.config)
    resolved = load_training(config.training_config, config.runtime_config)
    for module in (
        "magaox.indi.device",
        "purepyindi2",
        "xconf",
        "torch",
        "onnx",
        "ImageStreamIOWrap",
        "CacaoProcessTools",
    ):
        imported = importlib.import_module(module)
        print(module, getattr(imported, "__file__", "built-in"))
    root = Path(config.artifact_root)
    if not root.is_absolute() or not root.is_dir():
        raise ValueError("Provision artifact_root first")
    if not os.access(config.trtexec, os.X_OK):
        raise ValueError("trtexec is not executable")
    from configparser import ConfigParser

    runtime = ConfigParser(inline_comment_prefixes=("#", ";"))
    runtime.read(config.runtime_config)
    if (
        Path(runtime["parameters"]["engineDirs"]) / runtime["parameters"]["engineName"]
        != root / "current" / "policy.trt"
    ):
        raise ValueError("Runtime engine path does not match atomic publication link")
    with tempfile.TemporaryDirectory(
        prefix=".deployment-check-", dir=root
    ) as temporary:
        directory = Path(temporary)
        (directory / "bundle").mkdir()
        (directory / "next").symlink_to(directory / "bundle", target_is_directory=True)
        os.replace(directory / "next", directory / "current")
        assert (directory / "current").resolve() == directory / "bundle"
    print(
        json.dumps(
            dict(
                interpreter=sys.executable,
                config_revision=revision(resolved),
                artifact_root=str(root),
                status="imports/config/atomic-link checks passed",
            ),
            indent=2,
        )
    )
    print(
        "GPU export, full C++ build, writer identity and live INDI acceptance still require operator verification."
    )


if __name__ == "__main__":
    main()
