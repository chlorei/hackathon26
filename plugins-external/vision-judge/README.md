# dreamkit-vision-judge

A [dreamkit](https://github.com/chlorei/hackathon26) plugin that **judges every attempt from what the camera saw**
and keeps the statistics across runs: how often the robot really succeeds, at which task, and why it fails.

- **A vision judge.** At the end of each attempt, 10 frames taken evenly from one camera and the task go to Gemini
  (or Claude). It answers: success or not, a 0–100 score, the reason in words and as a code from `dreamkit/reasons.py`,
  and a status with a comment for every frame. success-check decides from the simulator's object positions; the judge
  needs only frames, so it works on a real robot too.
- **Statistics across runs.** At the end of a run, every judged attempt in `runs/` is counted: success rate with a
  Wilson interval, the same by task, the top failure reasons, and which task needs more demonstrations.
- **Can you trust the judge?** Where success-check (or failure-monitor) also decided, the plugin counts how often
  they agree: `judge agrees with success-check on 9 of 10 attempts`.

Nothing slow runs in the control loop: `on_step` only keeps a thinned buffer of small frames (budget well under
1 ms); the model call runs on the `ctx.defer` worker: 10 s per call, 20 s for the verdict in all (`timeout_s`,
`max_wait_s`), so the next attempt is not held up. If no model answers, the attempt gets `not judged` and the run goes on.

```bash
dreamkit install https://github.com/chlorei/hackathon26/tree/main/plugins-external/vision-judge
export GEMINI_API_KEY=...          # or GEMINI_API_KEY=... in a .env next to your loop (or a folder above); or provider: claude with ANTHROPIC_API_KEY (pip install "dreamkit-vision-judge[claude]")
```

```yaml
plugins:
  - success-check                  # optional: lets the plugin measure how often the judge agrees
  - vision-judge                  # or: - vision-judge: {camera: top, frames: 10}
```

Gemini Flash-Lite answers first (`gemini-3.5-flash-lite`, fast and cheap); on a 503 ("high demand") or a timeout,
Flash (`gemini-3.8-flash`) takes over, then Claude Haiku (`claude-haiku-5-5`) if `ANTHROPIC_API_KEY` is set. A 429
(quota) is never waited on: without a Claude key, all-429 means `not judged: quota` right away. On Google's free tier
Flash allows only 20 requests a day.

**Recoveries.** When recovery sends the robot back to try again, the try before it is judged on its own
(`judging try 1`, then `camera verdict: … (try 1, score …)` with timeline marks), so a drop between two frames of a
long attempt is not lost. The attempt's verdict is its last try's; earlier ones are under `"tries"`. A reply that is not valid JSON of the schema is asked again, twice at most.

## What it shows

- Per attempt: `judging attempt` in the feed, then `camera verdict: success, score 87, …` (success, with `model_s`: how long the model took) or
  `camera verdict: dropped the apple while carrying it (score 25)` (error); a badge `camera: success · 87`, timeline
  marks (`ctx.ui.mark`) at the frames the judge flagged, metrics `judged success` and `judge score`.
- **Its own page** (`ui/index.html`, PLUGIN_STANDARD §7a): click a vision-judge line in Live or History. A summary
  (attempts, successes, agreement with success-check), three charts (score of each attempt, why attempts fail, in
  which frame a failed attempt first goes wrong), every try with its 10 frames and the judge's comment on each, and
  *Download JSON / CSV*.
- At the end of a run, in the feed: overall rate with its interval, agreement with success-check, the recommendation.
- `run.json`, each attempt: `"vision-judge": {"success", "score", "reason", "reason_code", "frames": [{"index",
  "status", "comment", "t_s"}], "model_s", "tries": [...]}` (or `{"error": ...}` when not judged).
- `runs/<id>/vision-judge/`: the judged frames `ep003_t1_f0.jpg …` (per try) (512 px wide) and `attempts.json`
  (this run's verdicts, what the page reads).
- `runs/<id>/vision-judge/summary.json`: `overall`, `by_task`, `reasons` (code, label, count, share, examples),
  `agreement`, `recommendation`.

Example events:

```json
{"plugin": "vision-judge", "level": "error", "message": "camera verdict: dropped the apple while carrying it (score 25)",
 "data": {"score": 25, "frames_off": 2, "reason": "dropped"}}
{"plugin": "vision-judge", "level": "warn",
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
| `frames` | `10` |  | Frames per try sent to the judge (each recovery starts a new try) |
| `image_width` | `512` |  | Frames are scaled down to at most this many pixels wide |
| `buffer` | `64` |  | Frames kept per attempt; above this every other one is dropped |
| `provider` | `gemini` | gemini / claude | Vision model that judges (GEMINI_API_KEY or ANTHROPIC_API_KEY) |
| `timeout_s` | `10.0` | s | Longest wait for one model call |
| `max_wait_s` | `20.0` | s | Longest wait for a verdict, fallbacks included; dreamkit waits at most 120 s before saving the attempt |
| `confidence` | `0.95` |  | Confidence level of the success-rate interval |
<!-- /params -->
