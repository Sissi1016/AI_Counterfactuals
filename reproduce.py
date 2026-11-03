"""Entry point for checking and reproducing the final-appendix review package."""
from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.dont_write_bytecode = True
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
sys.path.insert(0, str(ROOT / "tools"))
from appendix_support import read_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("verify", help="Check final tables, graphics, source bindings, code, and checksums")
    sub.add_parser("assemble", help="Assemble final tables and an HTML exhibit index from supplied source cells")
    sub.add_parser("refresh", help="Refresh source cells from regenerated LaTeX (requires Pandoc), then assemble")
    sub.add_parser("figures", help="Redraw all final figure panels from aggregate CSVs (main environment)")
    sub.add_parser("plan", help="Print the analysis stages and their commands without executing them")
    stages = read_json(ROOT / "specification/workflow.json")
    run = sub.add_parser("run", help="Execute one raw-data analysis stage")
    run.add_argument("stage", choices=list(stages))
    run.add_argument("--dry-run", action="store_true")
    run.add_argument("--llm-python", default=".venv-llm/bin/python")
    args = parser.parse_args()
    if args.command == "verify":
        from verify_release import verify
        verify()
        return
    if args.command in ("assemble", "refresh"):
        command = [sys.executable, "tools/assemble_appendix.py"]
        if args.command == "refresh":
            command.append("--refresh-sources")
        subprocess.run(command, cwd=ROOT, check=True)
        subprocess.run([sys.executable, "tools/assemble_proposal.py"], cwd=ROOT, check=True)
        return
    if args.command == "figures":
        subprocess.run([sys.executable, "tools/rebuild_figures.py"], cwd=ROOT, check=True)
        return
    selected = list(stages) if args.command == "plan" else [args.stage]
    env = os.environ.copy()
    env.update(CFE_SAMPLE="all", CFE_LOSS="poisson")
    env.setdefault("CFE_THREADS", "4")
    env.setdefault("CFE_GPUS", "0,1")
    env.setdefault("CFE_MAX_SEQS", "4")
    for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        env[variable] = "1"
    for stage in selected:
        item = stages[stage]
        print(f"\n{stage}: {item['description']}", flush=True)
        print(f"Requires: {item['requires']}", flush=True)
        for task in item["tasks"]:
            python = getattr(args, "llm_python", ".venv-llm/bin/python") if task.get("gpu") else sys.executable
            command = [python] + (["-I"] if task.get("isolated") else []) + task["args"]
            task_env = env | task.get("env", {})
            overrides = {"CFE_SAMPLE": task_env["CFE_SAMPLE"], "CFE_LOSS": task_env["CFE_LOSS"]}
            overrides.update(task.get("env", {}))
            print(" ".join(f"{k}={shlex.quote(v)}" for k, v in overrides.items()) + " " + shlex.join(command), flush=True)
            if args.command == "run" and not args.dry_run:
                if task.get("gpu") and not Path(python).is_absolute():
                    command[0] = str(ROOT / python)
                subprocess.run(command, cwd=ROOT, env=task_env, check=True)


if __name__ == "__main__":
    main()
