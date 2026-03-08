#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))

from inference import MissingLLMCredentials, query_model  # type: ignore


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompt-file", required=True)
    parser.add_argument("--system-file", required=True)
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--tries", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=20.0)
    parser.add_argument("--version", default="1.5")
    parser.add_argument("--print-cost", action="store_true")
    parser.add_argument("--temp", default=None)
    args = parser.parse_args()

    os.environ["AGENTLAB_REMOTE_SUBPROCESS"] = "0"

    prompt = Path(args.prompt_file).read_text(encoding="utf-8")
    system_prompt = Path(args.system_file).read_text(encoding="utf-8")
    out_path = Path(args.out_json)

    payload = {"ok": False, "answer": "", "error": "", "error_type": ""}
    try:
        temp = None
        if args.temp not in (None, "", "None"):
            temp = float(args.temp)
        answer = query_model(
            args.model,
            prompt,
            system_prompt,
            tries=args.tries,
            timeout=args.timeout,
            temp=temp,
            print_cost=args.print_cost,
            version=args.version,
        )
        payload["ok"] = True
        payload["answer"] = answer if isinstance(answer, str) else ""
    except MissingLLMCredentials as e:
        payload["error"] = str(e)
        payload["error_type"] = "MissingLLMCredentials"
    except Exception as e:
        payload["error"] = str(e)
        payload["error_type"] = type(e).__name__

    out_path.write_text(json.dumps(payload), encoding="utf-8")
    return 0 if payload["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
