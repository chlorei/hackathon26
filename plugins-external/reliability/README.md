# dreamkit-reliability

A [dreamkit](https://github.com/chlorei/hackathon26) plugin that answers two questions with numbers you can defend:
**how often does it really work**, and **is B really better than A**.

- **Success rate with a confidence interval.** 3 successes out of 3 is "somewhere between 44 % and 100 %", not
  "100 %". The interval is Wilson's, which stays honest at the small counts a robot run produces.
- **Why the rest failed, how long successes took, what one success cost.** Failure reasons come from
  failure-monitor; cost = model calls × the price per robot-hour.
- **A/B against an earlier run.** Set `baseline_run` to the run you want to beat (base model vs your fine-tune,
  prompt A vs prompt B, recovery off vs on). With the same `seed`, both runs see the same scenes, so they are compared
  scene by scene with McNemar's exact test: "+22 pp, better" only when it is not luck, otherwise
  "not proven: could be luck".

Everything is computed from finished attempts after success-check and failure-monitor have written their verdicts:
nothing runs per step, so it fits any loop and any plugin order.

```bash
dreamkit install https://github.com/chlorei/hackathon26/tree/main/plugins-external/reliability
```

```yaml
plugins:
  - success-check
  - failure-monitor
  - reliability                                   # this run alone
  # - reliability: {baseline_run: 20261121-174122-0001, price_per_hour: 3}   # and compared with an earlier run
```

## What it shows

- During the run: a badge `success 4/30 · 13% (5–30%)` and a `success rate` metric, updated after each attempt.
- At the end, in the feed:
  `success 4 of 30 attempts, 13% (95% interval 5–30%)` and, with a baseline,
  `vs run 20261121-174122-0001: 43% vs 13% on 30 shared scenes, +30 pp, better`.
- `runs/<id>/reliability/summary.json`: rate, interval, failure reasons, mean time to success, model calls, cost,
  cost per success; with a baseline also its numbers, the comparison (paired or not, p-value) and the verdict.

Example event:

```json
{"plugin": "reliability", "level": "success", "message": "success 4 of 30 attempts, 13% (95% interval 5–30%)",
 "data": {"successes": 4, "attempts": 30, "rate_pct": 13.3, "low_pct": 5.3, "high_pct": 29.7, "cost_cents": 98,
          "model_calls": 738}}
```

## Settings

<!-- params -->
| param | default | unit | what |
|---|---|---|---|
| `confidence` | `0.95` |  | Confidence level of the success-rate interval |
| `price_per_hour` | `3.0` |  | Dollars per robot-hour of the model (Dream-1 3, flash 2); 0 for a mock run |
| `calls_per_hour` | `2250` |  | Model calls billed as one robot-hour (1.6 s of actions each) |
| `baseline_run` | `` |  | Run id or folder to compare with scene by scene; empty for none |
| `pair_on` | `first_attempt` | first_attempt / any_attempt | Which outcome of a scene to compare |
| `significance` | `0.05` |  | p-value below which a difference counts as real |
<!-- /params -->

Ported from DreamGym's Arena and Compare.
