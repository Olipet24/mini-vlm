"""Lightweight subprocess-based hyperparameter sweep driver, reusing
mini_vlm.train as-is (no optuna/wandb -- a designed grid over a handful of
discrete axes doesn't need a search library, and this keeps the whole
project's reproducibility story at `git clone && run`, no extra
dependencies or network/account requirements).

Config JSON shape:
    {
      "name": "s1_primary_reg",
      "base": {"model": "primary", "epochs": 12, "batch-size": 32, ...},
      "runs": [{"run-name": "p_d0.1_wd0.01", "dropout": 0.1, "weight-decay": 0.01}, ...]
    }
Every run = base merged with that run's overrides, translated 1:1 into
`python -m mini_vlm.train` CLI flags (hyphens in JSON keys match the flag
names directly; boolean True becomes a bare `--flag`, False/absent omits it).

Resumable by construction: a run is skipped if `{results-dir}/{run-name}_metrics.json`
already exists, so re-invoking the exact same command after a Ctrl-C, crash,
or an overnight stop picks up only the runs that didn't finish.

Usage:
    python -m mini_vlm.sweep --config sweeps/s1_primary_reg.json \
        --results-dir outputs/sweeps/s1_primary_reg --devices cuda:0,cuda:1 \
        [--dry-run] [--warmup-first]
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

_BOOL_FLAGS = {"skip-test", "ablate-vision", "save-best", "bridge-global-token", "bridge-question-cond"}


def build_argv(run_args: dict, results_dir: str) -> list:
    argv = [sys.executable, "-m", "mini_vlm.train", "--results-dir", results_dir]
    for key, value in run_args.items():
        if key == "results-dir":
            continue  # the driver's --results-dir always wins, for consistent sweep layout
        flag = f"--{key}"
        if key in _BOOL_FLAGS:
            if value:
                argv.append(flag)
            continue
        argv += [flag, str(value)]
    return argv


def load_config(config_path: Path):
    cfg = json.load(open(config_path))
    base = cfg.get("base", {})
    resolved = []
    for run in cfg["runs"]:
        merged = {**base, **run}
        run_name = merged.get("run-name") or merged.get("model")
        merged["run-name"] = run_name
        resolved.append(merged)
    return cfg.get("name", config_path.stem), resolved


def already_done(run_name: str, results_dir: Path) -> bool:
    return (results_dir / f"{run_name}_metrics.json").exists()


def run_one(run_args: dict, results_dir: Path, device: str, log_dir: Path):
    run_name = run_args["run-name"]
    args = {**run_args, "device": device}
    argv = build_argv(args, str(results_dir))
    log_path = log_dir / f"{run_name}.log"
    start = time.time()
    with open(log_path, "w") as logf:
        logf.write(f"$ {' '.join(argv)}\n\n")
        logf.flush()
        proc = subprocess.run(argv, stdout=logf, stderr=subprocess.STDOUT)
    duration = time.time() - start
    return {
        "run_name": run_name,
        "device": device,
        "returncode": proc.returncode,
        "duration_sec": round(duration, 1),
        "log": str(log_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--results-dir", required=True)
    parser.add_argument("--devices", default="cuda:0,cuda:1")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--warmup-first", action="store_true",
        help="run the first config alone, to completion, before parallelizing the rest -- "
             "avoids two cold-cache primary runs racing on the WKV CUDA kernel's ninja "
             "JIT-compile lock in ~/.cache/torch_extensions.",
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    log_dir = results_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    status_path = results_dir / "sweep_status.json"

    devices = args.devices.split(",")
    sweep_name, runs = load_config(config_path)

    pending = [r for r in runs if not already_done(r["run-name"], results_dir)]
    skipped = [r["run-name"] for r in runs if r not in pending]
    print(f"[{sweep_name}] {len(runs)} configs total, {len(skipped)} already done "
          f"(skipping), {len(pending)} to run across devices={devices}")
    if skipped:
        print(f"  already done: {skipped}")

    if args.dry_run:
        for r in pending:
            print("  " + " ".join(build_argv({**r, "device": devices[0]}, str(results_dir))))
        return

    if not pending:
        print("nothing to do.")
        return

    status = json.load(open(status_path)) if status_path.exists() else {"runs": []}

    if args.warmup_first:
        first = pending.pop(0)
        print(f"[warmup] running {first['run-name']} alone on {devices[0]} before parallelizing...")
        result = run_one(first, results_dir, devices[0], log_dir)
        status["runs"].append(result)
        json.dump(status, open(status_path, "w"), indent=2)
        outcome = "OK" if result["returncode"] == 0 else f"FAILED (rc={result['returncode']})"
        print(f"[warmup] {first['run-name']}: {outcome} in {result['duration_sec']:.0f}s")

    # simple polling scheduler: one subprocess per device slot at a time
    active = {}  # device -> (subprocess.Popen, run_name, start_time, log_file_handle)
    queue = list(pending)
    free_devices = list(devices)

    def launch(run_args, device):
        run_name = run_args["run-name"]
        run_full = {**run_args, "device": device}
        argv = build_argv(run_full, str(results_dir))
        log_path = log_dir / f"{run_name}.log"
        logf = open(log_path, "w")
        logf.write(f"$ {' '.join(argv)}\n\n")
        logf.flush()
        proc = subprocess.Popen(argv, stdout=logf, stderr=subprocess.STDOUT)
        print(f"[launch] {run_name} on {device} (pid {proc.pid})")
        active[device] = (proc, run_name, time.time(), logf)
        if device in free_devices:
            free_devices.remove(device)

    while queue or active:
        while queue and free_devices:
            device = free_devices.pop(0)
            launch(queue.pop(0), device)

        time.sleep(5)
        finished_devices = []
        for device, (proc, run_name, start, logf) in list(active.items()):
            rc = proc.poll()
            if rc is not None:
                logf.close()
                duration = time.time() - start
                outcome = "OK" if rc == 0 else f"FAILED (rc={rc})"
                print(f"[done]   {run_name} on {device}: {outcome} in {duration:.0f}s")
                status["runs"].append({
                    "run_name": run_name, "device": device, "returncode": rc,
                    "duration_sec": round(duration, 1), "log": str(log_dir / f"{run_name}.log"),
                })
                json.dump(status, open(status_path, "w"), indent=2)
                finished_devices.append(device)
        for device in finished_devices:
            del active[device]
            free_devices.append(device)

    n_ok = sum(1 for r in status["runs"] if r["returncode"] == 0)
    n_fail = sum(1 for r in status["runs"] if r["returncode"] != 0)
    print(f"[{sweep_name}] finished this invocation: {n_ok} succeeded, {n_fail} failed "
          f"(see {status_path} and {log_dir}/*.log for details)")


if __name__ == "__main__":
    main()
