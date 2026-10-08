# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Run the model server on its own: ``python -m model_serving``.

From ``smart-classroom/``::

    python -m model_serving                 # host/port from models.text_gen.serving
    python -m model_serving --port 8000     # model-only install on the usual port
    python -m model_serving --config ../utils/flutter/config.yaml

The model is selected exactly as for the app: ``models.text_gen.vlm_name``,
``weight_format``, ``device`` (and ``speculative``) in config.yaml.
"""

import argparse
import os
import sys
from pathlib import Path

_SC_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description="edu-ai-suite VLM/LLM model server")
    parser.add_argument("--host", help="bind address (default: serving.host)")
    parser.add_argument("--port", type=int, help="port (default: serving.port)")
    parser.add_argument("--config", help="config.yaml to read (default: SC_CONFIG_PATH or ./config.yaml)")
    parser.add_argument("--parent-pid", type=int, help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.config:
        os.environ["SC_CONFIG_PATH"] = str(Path(args.config).resolve())
    # config.yaml and the models/ tree resolve from the app root.
    os.chdir(_SC_ROOT)
    if str(_SC_ROOT) not in sys.path:
        sys.path.insert(0, str(_SC_ROOT))

    from utils.logger_config import setup_logger

    setup_logger()

    import uvicorn

    from model_serving.server import create_app
    from model_serving.settings import serving_settings

    settings = serving_settings()
    uvicorn.run(
        create_app(parent_pid=args.parent_pid),
        host=args.host or settings.host,
        port=args.port or settings.port,
        timeout_graceful_shutdown=5,
    )


if __name__ == "__main__":
    main()
