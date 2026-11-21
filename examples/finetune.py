"""Learn, the last step: harvested successes -> one checked dataset -> Dream-1 fine-tune -> its model id (from DreamGym).

`dreamkit harvest export` gathers and checks the data and uploads nothing; this script takes the next steps with
Dream Labs' own client calls, and writes what it did to <dataset>/.dreamkit_finetune.json (a hidden file: dataset
checks and uploads skip it), so `--status` can be asked any time later.

    python examples/finetune.py --runs <run-id> <run-id> --check      # gather + validate only (no key, no credits)
    python examples/finetune.py --runs <run-id> --suffix apple         # + upload + start the job (fine-tuning credits)
    python examples/finetune.py --status                               # poll the job; prints the model id when done

A job takes hours (Dream Labs quotes 7-8 h for three cameras). Then compare it with the base model on the same scenes:
    python examples/ab_test.py --models dreamlabs/dream-1-beta,<model id> --episodes 20
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from dreamkit import config as dk_config, harvest  # noqa: E402
from dreamkit.plugins.harvester.plugin import YAM_CONFIG  # noqa: E402

EMBODIMENT = "dreamkit-yam"
## The sim's YAM body as the harvester writes it, plus the camera the model treats as the head.
CONFIG = {**YAM_CONFIG, "head_camera": "overhead"}
STATE = ".dreamkit_finetune.json"


def save(folder: Path, data: dict) -> None:
    (folder / STATE).write_text(json.dumps({**data, "updated": time.time()}, indent=2, default=str))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", nargs="*", help="run ids whose successful attempts to use (default: all runs)")
    parser.add_argument("--out", default="harvest", help="the dataset folder to build")
    parser.add_argument("--name", help="dataset name on the platform (default: the folder's name)")
    parser.add_argument("--suffix", default="dreamkit", help="the fine-tune is named dream-1-ft:<id>-<suffix>")
    parser.add_argument("--steps", type=int, help="training steps (multiple of 100, 500-10000; service default 1000)")
    parser.add_argument("--check", action="store_true", help="gather and validate only")
    parser.add_argument("--status", action="store_true", help="poll the job started from --out")
    parser.add_argument("--config", default="dreamkit.yaml")
    args = parser.parse_args(argv)
    out = Path(args.out)

    if args.status:
        state = json.loads((out / STATE).read_text()) if (out / STATE).is_file() else {}
        if not state.get("job_id"):
            print(f"no fine-tune started from {out}")
            return 1
        import dream

        job = dream.DreamClient().get_finetuning_status(state["job_id"])
        job = job if isinstance(job, dict) else vars(job)
        state.update(job=job, model=job.get("fine_tuned_model"))
        save(out, state)
        print(json.dumps(job, indent=2, default=str))
        if state["model"]:
            print(f"\nmodel ready: {state['model']}")
        return 0

    cfg, base = dk_config.load(args.config if Path(args.config).is_file() else None)
    root = Path(cfg.runs_dir) if Path(cfg.runs_dir).is_absolute() else base / cfg.runs_dir
    runs = [root / r for r in args.runs] if args.runs else sorted(p for p in root.iterdir() if p.is_dir())
    copied = harvest.export(runs, out)
    if not copied:
        print("no successful attempts in these runs (is the harvester plugin on?)")
        return 1
    report = harvest.validate(out, CONFIG)
    print(f"{len(copied)} episodes -> {out}; validate_dataset: "
          + ("OK" if report.get("ok") else f"FAILED: {report.get('error')}"))
    if not report.get("ok") or args.check:
        return 0 if report.get("ok") else 1

    import dream

    client = dream.DreamClient()
    mine = {e.name.split("/", 1)[-1]: e for e in client.get_embodiments()}
    embodiment = mine.get(EMBODIMENT) or client.add_embodiment(EMBODIMENT, CONFIG)
    name = (args.name or out.name).replace("_", "-").lower()
    dataset = client.upload_dataset(str(out), name, embodiment.name)
    save(out, {"stage": "processing", "dataset_id": dataset.id, "dataset_name": dataset.name})
    print(f"uploaded {dataset.id}; waiting for Dream Labs to process it ...", flush=True)
    while (status := client.get_dataset_status(dataset.id)).status not in ("processed", "failed"):
        time.sleep(20)
    if status.status == "failed":
        save(out, {"stage": "failed", "dataset_id": dataset.id, "error": status.error})
        print(f"processing failed: {status.error}")
        return 1
    job = client.finetune(dataset.name, args.suffix, steps=args.steps)
    save(out, {"stage": "training", "dataset_id": dataset.id, "dataset_name": dataset.name, "job": job,
               "job_id": job.get("id") or job.get("job_id")})
    print(json.dumps(job, indent=2, default=str))
    print(f"\ntraining; ask later: python examples/finetune.py --out {out} --status")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
