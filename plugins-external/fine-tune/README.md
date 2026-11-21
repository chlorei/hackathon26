# dreamkit-fine-tune

A [dreamkit](https://github.com/chlorei/hackathon26) plugin that closes **Learn**: the harvester's successful attempts
become one dataset checked by Dream Labs' own `validate_dataset`, and, when you switch it on, a Dream-1 fine-tune.

```bash
dreamkit install https://github.com/chlorei/hackathon26/tree/main/plugins-external/fine-tune
```

```yaml
plugins:
  - success-check
  - failure-monitor
  - harvester
  - fine-tune                                     # build + check the dataset after every run (free)
  # - fine-tune: {auto_train: true, min_episodes: 30, suffix: apple}   # + upload and train (fine-tuning credits)
```

## What it does

At the end of every run, on the `ctx.defer` worker (the loop never waits for it):

1. gathers the successful attempts of this run and, with `include_previous_runs`, of earlier runs of the same task
   into `runs/<id>/fine-tune/dataset/` (one LeRobot v2.1 dataset per subfolder, YAM embodiment);
2. runs `validate_dataset` on it (local: no key, no credits) and writes `status.json`;
3. with `auto_train` on and at least `min_episodes`: registers the `dreamkit-yam` embodiment once, uploads the dataset,
   waits up to `processing_wait_s` for Dream Labs to process it and starts the fine-tune; `job.json` keeps the job.

At the start of every run it asks Dream Labs about the newest earlier job: a processed upload starts training, a
finished job announces `fine-tune ready: dream-1-ft:…` in the feed. Training takes hours, so the model id usually
arrives a few runs later.

The plugin's page (› on its lines in the feed) shows the attempts collected, minutes of demonstrations, the
`validate_dataset` result, the job's stage and model id, and the command to compare the fine-tune with the base model:

```bash
python examples/ab_test.py --models dreamlabs/dream-1-beta,<model id> --episodes 20
```

`examples/finetune.py` does the same steps by hand from the terminal.

Example event:

```json
{"plugin": "fine-tune", "level": "success", "message": "dataset 34 attempts, 6.8 min, validate_dataset ok",
 "data": {"episodes": 34, "duration_s": 408.0}}
```

Notes: data from `--mock` runs are scripted successes, fine for a demo dataset, not Dream-1's own behaviour. The
dataset is copied into each run's folder (the standard allows writes only there), so it grows with the runs.

## Settings

<!-- params -->
| param | default | unit | what |
|---|---|---|---|
| `include_previous_runs` | `True` |  | Also gather successful attempts of earlier runs |
| `same_task_only` | `True` |  | Only attempts of this run's task |
| `min_episodes` | `20` |  | Successful attempts needed before training starts |
| `auto_train` | `False` |  | Upload and start a Dream-1 fine-tune when the dataset is ready (uses fine-tuning credits) |
| `train_again` | `False` |  | Start a new job even when an earlier one has finished |
| `suffix` | `dreamkit` |  | The fine-tune is named dream-1-ft:<id>-<suffix> |
| `train_steps` | `0` |  | Training steps, a multiple of 100; 0 for the service default |
| `processing_wait_s` | `60.0` | s | How long to wait for Dream Labs to process an upload before leaving training to the next run |
| `check_status` | `True` |  | At run start, ask Dream Labs about the last job |
<!-- /params -->
