# dreamkit-monitor-stats

A [dreamkit](https://github.com/chlorei/hackathon26) plugin that **judges every attempt from what the camera saw**
and keeps the statistics across runs: how often the robot really succeeds, at which task, and why it fails.

- **A vision judge.** At the end of each attempt, 8 frames taken evenly from one camera and the task go to Gemini
  (or Claude). It answers: success or not, a 0–100 score, the reason in words and as a code from `dreamkit/reasons.py`,
  and a status with a comment for every frame. success-check decides from the simulator's object positions; the judge
  needs only frames, so it works on a real robot too.
- **Statistics across runs.** At the end of a run, every judged attempt in `runs/` is counted: success rate with a
  Wilson interval, the same by task, the top failure reasons, and which task needs more demonstrations.
- **Can you trust the judge?** Where success-check (or failure-monitor) also decided, the plugin counts how often
  they agree: `judge agrees with success-check on 9 of 10 attempts`.

Nothing slow runs in the control loop: `on_step` only keeps a thinned buffer of small frames (budget well under
1 ms); the model call runs on the `ctx.defer` worker and must finish within `max_wait_s` (90 s), because dreamkit
saves the attempt after 120 s. If no model answers, the attempt gets `not judged` and the run goes on.

```bash
dreamkit install https://github.com/chlorei/hackathon26/tree/main/plugins-external/monitor-stats
export GEMINI_API_KEY=...          # or provider: claude with ANTHROPIC_API_KEY (pip install "dreamkit-monitor-stats[claude]")
```

```yaml
plugins:
  - success-check                  # optional: lets the plugin measure how often the judge agrees
  - monitor-stats                  # or: - monitor-stats: {camera: top, frames: 8}
```

Gemini models are tried in order `gemini-3.8-flash → gemini-3.6-flash → gemini-3.5-flash-lite`: a 503 ("high demand")
or a timeout moves on to the next one. A reply that is not valid JSON of the schema is asked again, twice at most.

## What it shows

- Per attempt: a badge `judge: success · 87` or `judge: failed · 25`, timeline marks on the frames the judge flagged,
  metrics `judged success` and `judge score`, and an event in the feed.
- At the end of a run, in the feed: overall rate with its interval, agreement with success-check, the recommendation.
- `run.json`, each attempt: `"monitor-stats": {"success", "score", "reason", "reason_code", "frames": [{"index",
  "status", "comment", "t_s"}]}` (or `{"error": ...}` when not judged).
- `runs/<id>/monitor-stats/summary.json`: `overall`, `by_task`, `reasons` (code, label, count, share, examples),
  `agreement`, `recommendation`.

Example events:

```json
{"plugin": "monitor-stats", "level": "warn", "message": "judge: failed, dropped the apple while carrying it (score 25)",
 "data": {"score": 25, "frames_off": 2, "reason": "dropped"}}
{"plugin": "monitor-stats", "level": "warn",
 "message": "record more demonstrations of 'put apple into the bowl': 2 of 5 judged successful, mostly dropped"}
```

## Develop

```bash
pip install -e ".[dev]"
dreamkit check .        # the standard (PLUGIN_STANDARD.md)
dreamkit test .         # replay tests with a fake model: no network, no credits, no simulator
dreamkit bench .        # ms/step on every bundled fixture
```

## Settings

<!-- params -->
| param | default | unit | what |
|---|---|---|---|
| `camera` | `top` | top / left / right | Camera the judge looks at (the first one if it is missing) |
| `frames` | `8` |  | Frames per attempt sent to the judge |
| `image_width` | `512` |  | Frames are scaled down to at most this many pixels wide |
| `buffer` | `64` |  | Frames kept per attempt; above this every other one is dropped |
| `provider` | `gemini` | gemini / claude | Vision model that judges (GEMINI_API_KEY or ANTHROPIC_API_KEY) |
| `timeout_s` | `30.0` | s | Longest wait for one model call |
| `max_wait_s` | `90.0` | s | Longest wait for a verdict, fallbacks included; dreamkit waits at most 120 s before saving the attempt |
| `confidence` | `0.95` |  | Confidence level of the success-rate interval |
<!-- /params -->
