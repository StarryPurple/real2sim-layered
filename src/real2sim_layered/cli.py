"""CLI entrypoints for MVOIK-4D probe inspection, smoke evaluation, and IKG extraction."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from real2sim_layered.eval import evaluate_case_dual_axis, format_reports_markdown
from real2sim_layered.schemas import DualAxisReport, RuntimeConfig
from real2sim_layered.tools import (
    OpenAICompatVlmClient,
    load_probe_cases,
    load_rgba_frame,
    sample_frame_indices,
)

_SMOKE_MJCF = """
<mujoco>
  <option gravity="0 0 -9.81" timestep="0.005"/>
  <worldbody>
    <light pos="0 0 2"/>
    <geom name="floor" type="plane" size="2 2 0.1" rgba="0.8 0.8 0.8 1"/>
    <body name="block" pos="0 0 0.06">
      <freejoint/>
      <geom type="box" size="0.05 0.05 0.05" mass="0.3" rgba="0.2 0.6 0.8 1"/>
    </body>
  </worldbody>
</mujoco>
"""


def _cmd_probe_inspect(cfg: RuntimeConfig) -> None:
    cases = load_probe_cases(cfg.dataset_root)
    for case in cases:
        views = sorted(case.eval_views.keys())
        print(  # noqa: T201  # CLI subcommand prints tabular report to stdout
            f"{case.case_id:32s} regime={case.regime:20s} "
            f"frames={case.num_frames:3d} size={case.width}x{case.height} views={views}"
        )


def _cmd_eval_smoke(cfg: RuntimeConfig, out_path: Path | None) -> None:
    """Run an identity/upper-bound visual + MuJoCo physics smoke check on the 5 probe cases."""
    cases = load_probe_cases(cfg.dataset_root)
    reports: list[DualAxisReport] = []
    for case in cases:
        t0 = time.perf_counter()
        indices = sample_frame_indices(case.num_frames, max_frames=4)
        gt_by_view = {
            view_name: [
                load_rgba_frame(frames[idx], target_size=(160, 120))
                for idx in indices
                if idx < len(frames)
            ]
            for view_name, frames in case.eval_views.items()
        }
        # Self-consistency smoke baseline: compares GT against itself plus a stable MuJoCo block.
        report = evaluate_case_dual_axis(
            case=case,
            method_tag="smoke_oracle_check",
            pred_by_view=gt_by_view,
            gt_by_view=gt_by_view,
            mjcf_xml=_SMOKE_MJCF,
            physics_steps=120,
            wall_time_s=time.perf_counter() - t0,
        )
        reports.append(report)

    md = format_reports_markdown(reports)
    print(md)  # noqa: T201  # CLI subcommand prints evaluation table to stdout
    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            json.dumps([r.model_dump(mode="json") for r in reports], indent=2) + "\n",
            encoding="utf-8",
        )


def _cmd_extract_ikg(
    cfg: RuntimeConfig,
    case_ids: list[str] | None,
    max_frames: int,
    out_dir: Path | None,
) -> None:
    cases = load_probe_cases(cfg.dataset_root)
    if case_ids:
        wanted = set(case_ids)
        cases = [c for c in cases if c.case_id in wanted]
    client = OpenAICompatVlmClient(cfg.vlm)
    dest = out_dir or (cfg.outputs_root / "ikg_probe")
    dest.mkdir(parents=True, exist_ok=True)

    for case in cases:
        t0 = time.perf_counter()
        result = client.extract_ikg(case, max_frames=max_frames)
        elapsed = time.perf_counter() - t0
        out_file = dest / f"{case.case_id}.json"
        out_file.write_text(result.graph.model_dump_json(indent=2) + "\n", encoding="utf-8")
        print(  # noqa: T201  # CLI subcommand prints extraction status to stdout
            f"[OK] {case.case_id} -> regime={result.graph.dominant_regime} "
            f"(gt={case.regime}) objects={len(result.graph.objects)} "
            f"intervals={len(result.graph.intervals)} tokens={result.total_tokens} "
            f"time={elapsed:.1f}s saved={out_file}"
        )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="real2sim-layered")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("probe-inspect", help="Inspect the 5 cross-regime MVOIK-4D probe cases.")

    p_smoke = sub.add_parser("eval-smoke", help="Run dual-axis smoke evaluation on probe cases.")
    p_smoke.add_argument("--out", type=Path, default=None, help="Optional JSON output path.")

    p_ikg = sub.add_parser("extract-ikg", help="Extract structured G_int via configured VLM.")
    p_ikg.add_argument(
        "--case",
        action="append",
        default=None,
        help="Optional probe case ID filter (repeatable).",
    )
    p_ikg.add_argument("--max-frames", type=int, default=6, help="Sampled frames per clip.")
    p_ikg.add_argument("--out-dir", type=Path, default=None, help="Output directory for JSONs.")

    args = parser.parse_args(argv)
    cfg = RuntimeConfig.from_env()

    if args.command == "probe-inspect":
        _cmd_probe_inspect(cfg)
    elif args.command == "eval-smoke":
        _cmd_eval_smoke(cfg, out_path=args.out)
    elif args.command == "extract-ikg":
        _cmd_extract_ikg(
            cfg,
            case_ids=args.case,
            max_frames=args.max_frames,
            out_dir=args.out_dir,
        )


if __name__ == "__main__":
    main()
