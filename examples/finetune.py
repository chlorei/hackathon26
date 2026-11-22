"""Learn, the last step: a dataset in <data_dir>/datasets/ -> checked -> Dream-1 fine-tune -> its model id.

The harvester appends every successful attempt to datasets/harvest-<task>/ as it ends, so the data is already there
(`dreamkit datasets` lists them). This script takes the next steps with Dream Labs' own client calls and writes what
it did to the dataset's dataset.json (upload, finetune), where the Data tab and `--status` read it.

    python examples/finetune.py harvest-put-apple-into-the-bowl --check     # validate only (no key, no credits)
    python examples/finetune.py harvest-put-apple-into-the-bowl --suffix apple   # + upload + start the job (credits)
    python examples/finetune.py harvest-put-apple-into-the-bowl --status    # poll the job; prints the model id when done

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
from dreamkit import config as dk_config, datasets  # noqa: E402
from dreamkit.plugins.harvester.plugin import YAM_CONFIG  # noqa: E402

EMBODIMENT = "dreamkit-yam"
## The sim's YAM body as the harvester writes it, plus the camera the model treats as the head.
CONFIG = {**YAM_CONFIG, "head_camera": "overhead"}



def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("dataset", help="a dataset in <data_dir>/datasets/ (dreamkit datasets lists them)")
    parser.add_argument("--name", help="dataset name on the platform (default: the folder's name)")
    parser.add_argument("--suffix", default="dreamkit", help="the fine-tune is named dream-1-ft:<id>-<suffix>")
    parser.add_argument("--steps", type=int, help="training steps (multiple of 100, 500-10000; service default 1000)")
    parser.add_argument("--check", action="store_true", help="validate only")
    parser.add_argument("--status", action="store_true", help="poll the job started for this dataset")
    parser.add_argument("--config", default="dreamkit.yaml")
    args = parser.parse_args(argv)
    cfg, base = dk_config.load(args.config if Path(args.config).is_file() else None)
    root = cfg.datasets_path(base) / args.dataset
    if not root.is_dir():
        print(f"no dataset {args.dataset!r} in {root.parent} (dreamkit datasets)")
        return 1

    if args.status:
        job = datasets.record(root).get("finetune") or {}
        if not job.get("job_id"):
            print(f"no fine-tune started for {args.dataset}")
            return 1
        import dream

        status = dream.DreamClient().get_finetuning_status(job["job_id"])
        status = status if isinstance(status, dict) else vars(status)
        model = status.get("fine_tuned_model")
        datasets.update(root, finetune={"status": status.get("status"), "model": model, "updated": time.time()})
        print(json.dumps(status, indent=2, default=str))
        if model:
            print(f"\nmodel ready: {model}")
        return 0

    report = datasets.check(root, CONFIG)
    print(f"{datasets.describe(root)['episodes']} episodes in {args.dataset}; validate_dataset: "
          + ("OK" if report.get("ok") else f"FAILED: {report.get('error')}"))
    if not report.get("ok") or args.check:
        return 0 if report.get("ok") else 1

    import dream

    client = dream.DreamClient()
    mine = {e.name.split("/", 1)[-1]: e for e in client.get_embodiments()}
    embodiment = mine.get(EMBODIMENT) or client.add_embodiment(EMBODIMENT, CONFIG)
    name = (args.name or root.name).replace("_", "-").lower()
    with datasets.lerobot_view(root) as view:  # meta/, data/, videos/ only
        dataset = client.upload_dataset(str(view), name, embodiment.name)
    datasets.update(root, upload={"dataset_id": dataset.id, "dataset_name": dataset.name, "status": "processing"})
    print(f"uploaded {dataset.id}; waiting for Dream Labs to process it ...", flush=True)
    while (status := client.get_dataset_status(dataset.id)).status not in ("processed", "failed"):
        time.sleep(20)
    if status.status == "failed":
        datasets.update(root, upload={"status": "failed", "error": str(status.error)})
        print(f"processing failed: {status.error}")
        return 1
    job = client.finetune(dataset.name, args.suffix, steps=args.steps)
    datasets.update(root, upload={"status": "processed"},
                    finetune={"job_id": job.get("id") or job.get("job_id"), "status": "training", "stage": "training",
                              "updated": time.time()})
    print(json.dumps(job, indent=2, default=str))
    print(f"\ntraining; ask later: python examples/finetune.py {args.dataset} --status")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
