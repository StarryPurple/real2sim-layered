"""Resumable downloader for Lijiaxin0111/MVOIK-4D dataset into PFS."""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

from huggingface_hub import snapshot_download

from real2sim_layered.schemas import clear_proxy_env, load_hf_token


def main() -> None:
    parser = argparse.ArgumentParser(description="Download MVOIK-4D dataset from HuggingFace.")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(os.environ.get("REAL2SIM_ROOT", ".")) / "data" / "MVOIK-4D",
        help="Destination directory for MVOIK-4D.",
    )
    parser.add_argument(
        "--env-file",
        type=Path,
        default=Path(os.environ.get("REAL2SIM_ROOT", ".")) / ".env",
        help="Optional .env file providing HF_TOKEN.",
    )
    parser.add_argument("--workers", type=int, default=16, help="Parallel download workers.")
    parser.add_argument("--retries", type=int, default=5, help="Retry count on transient errors.")
    parser.add_argument(
        "--allow-patterns",
        action="append",
        default=None,
        help="Repeatable glob pattern filter (e.g., 'with_gt/**'). Omit for full dataset.",
    )
    parser.add_argument(
        "--no-proxy",
        action="store_true",
        help="Unset HTTP(S)_PROXY environment variables to connect directly.",
    )
    args = parser.parse_args()

    if args.no_proxy:
        clear_proxy_env()

    token = load_hf_token(args.env_file)
    auth_state = "authenticated" if token else "unauthenticated"
    route_state = "direct (--no-proxy)" if args.no_proxy else "env-proxy"
    scope = ", ".join(args.allow_patterns) if args.allow_patterns else "FULL"

    args.out.mkdir(parents=True, exist_ok=True)
    print(
        f"Downloading Lijiaxin0111/MVOIK-4D [{scope}] to {args.out} "
        f"(workers={args.workers}, {auth_state}, {route_state}) ...",
        flush=True,
    )

    last_err: Exception | None = None
    for attempt in range(1, args.retries + 1):
        try:
            t0 = time.time()
            snapshot_download(
                repo_id="Lijiaxin0111/MVOIK-4D",
                repo_type="dataset",
                local_dir=str(args.out),
                max_workers=args.workers,
                allow_patterns=args.allow_patterns,
                token=token,
            )
            elapsed = time.time() - t0
            print(f"Download completed on attempt {attempt} in {elapsed:.1f}s.", flush=True)
            last_err = None
            break
        except Exception as exc:
            last_err = exc
            print(
                f"[attempt {attempt}/{args.retries}] error: {exc}; retrying in 10s...", flush=True
            )
            time.sleep(10)

    if last_err is not None:
        raise last_err

    for sub in [
        "with_gt/input",
        "with_gt/eval_gt",
        "with_gt_memory/input",
        "with_gt_memory/eval_gt",
        "with_gt_memory_mask/input",
        "with_gt_gaussian_ls",
    ]:
        p = args.out / sub
        count = len(list(p.iterdir())) if p.is_dir() else 0
        print(f"  {sub}: {count} entries", flush=True)


if __name__ == "__main__":
    main()
