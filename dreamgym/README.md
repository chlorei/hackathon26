# DreamGym

A reliability gym for Dream-1 in Dream Labs' YAM MuJoCo simulator, with its own web app. Where dreamkit lives
**inside** your control loop, DreamGym **drives** the simulator itself: many seeded episodes, a judge, statistics,
datasets and fine-tuning, all from one page.

```bash
dreamgym serve            # http://127.0.0.1:8765 (dreamkit's panel stays on :8800)
```

| tab | what it does |
|---|---|
| **Runs → Arena** | N seeded episodes of a task with scene randomisation (object positions, yaw, lighting); success judged from the physics; success rate with a Wilson 95% interval, failure reasons (`untouched`, `misplaced`, `still_held`, `fell_off_table`, …), time and cost per success, a GIF of every failure |
| **Runs → Compare** | two or more variants (models × prompts) on **the same seeds**, McNemar's exact test: "B is better, and it is not luck" or "could be luck" |
| **Runs → Prompt Tuner** | paraphrases of the task's prompt on the same seeds, ranked, the baseline first |
| **Data & fine-tune** | scripted demos (an IK expert, free), **Harvester** (Dream-1's own successes), datasets with **Check** (`validate_dataset`), **Upload & train**, job status, **Compare vs base** when the model is ready |
| **Teleop** | drive the arms from the keyboard in the browser (IK on the gripper), record demonstrations into a LeRobot v2.1 dataset |

The same from the terminal:

```bash
dreamgym tasks
dreamgym arena   --task banana_plate --policy dream-1 --n 30
dreamgym compare --task apple_bowl --policy dream-1 --prompt "put apple into the bowl" --prompt "place the apple into the bowl"
dreamgym record  --task banana_plate --n 30                       # scripted demos -> gym/data/banana_plate
dreamgym finetune gym/data/banana_plate --validate-only           # then without the flag: upload + train; --status
```

Policies: `dream-1` (= `dreamlabs/dream-1-beta`), `dream:<model id>` (flash, a fine-tune), `scripted[:noise_m]`.

**Where things go:** `gym/runs/<id>/` (episodes.jsonl, summary.json, gifs/, report.html), `gym/data/<dataset>/`
(LeRobot v2.1 at 10 fps, VP9; state `.dreamgym/` beside it). `gym/` is ignored by git.

**Simulator:** found in this order: `DREAMGYM_SIM_DIR`, `sim_path` of `./dreamkit.yaml`, the `dream_examples`
package, `../dreamlabs/...` next to the repo. It is imported, never edited.

**Rules kept:** the arm, cameras and simulator files are Dream Labs' as shipped; every Dream-1 chunk goes through
`dream_control.validate_chunk`; only objects are moved (randomisation). The scripted expert and Teleop use IK on a
scratch copy of the model data and send joint rows through `env.step` like any policy.

**Code:** `core.py` (Task / Policy contracts) · `tasks.py` (pick-and-place tasks, prompt paraphrases) ·
`policies/` (Dream-1, scripted IK expert) · `runner.py` (one episode) · `arena.py` / `compare.py` · `stats.py`
(Wilson, McNemar, Fisher) · `lerobot_writer.py` · `finetune.py` · `web/` (server, page, teleop).

Known limits: the success thresholds in `scene.py` come from the scene geometry; Teleop's key directions follow the
top camera and are not tuned on hardware; fine-tuning on simulator data helps in the simulator, not necessarily on a
real robot.
