# DreamGym — a dreamkit app plugin

The **Data** tab of dreamkit's panel (PLUGIN_STANDARD.md §13): everything needed to fine-tune Dream-1 on your
own demonstrations, and nothing else (live runs, History and the loop plugins are dreamkit's own tabs). Every
episode still goes **through `dreamkit.wrap`**, so the loop plugins of `dreamkit.yaml` run while you record.

```bash
dreamkit install ./plugins-external/dreamgym   # check + its tests, then pip install
dreamkit ui                                     # restart the panel: tab "Data"
```

The tab exists only while the package is installed. It is not listed among the loop plugins: install it from
Plugins → Catalogue → **Data tab**, remove it with **Uninstall tab** in the tab's header (pip uninstall; datasets,
runs and tasks in the data folder stay on disk).

One dataset, four steps on one page:

| step | what it does |
|---|---|
| **1 · Collect** | pick tasks and how many episodes of each; source: **scripted expert** (IK, free), **Dream-1** (its successes, uses credits) or **Teleop** (keyboard). Tasks take turns, prompts are paraphrased, objects and lighting randomised. Only successes are kept. Progress per task is live, Stop keeps what was saved. **Your own task**: type the command in your words and pick which object it moves and where (filled in from the words when it can tell); it is saved in `<data_dir>/dreamgym-tasks.json` and judged and recorded like the built-in ones |
| **2 · Review** | episodes appear the moment they are saved: balance per task (aim ≥ 20 each), warnings, every episode with its source, seed and prompt, ▶ its three camera videos, remove the ones not to learn from (the dataset is renumbered and stays valid) |
| **3 · Train** | Check (`validate_dataset`, local), then Upload & fine-tune (about $200, 7–8 h; asks first), job status, the model id |
| **4 · Evaluate** | base Dream-1 vs the fine-tune on the same scenes for every task of the dataset, McNemar: "better, and not luck" or "could be luck" |

The same from the terminal (`dreamkit dreamgym …` = `dreamgym …`):

```bash
dreamkit dreamgym arena --task banana_plate --policy scripted --n 10 --recovery-ab   # recovery off vs on, same seeds
dreamgym tasks
dreamgym arena   --task banana_plate --policy dream-1 --n 30
dreamgym compare --task apple_bowl --policy dream-1 --prompt "put apple into the bowl" --prompt "place the apple into the bowl"
dreamgym record  --task banana_plate --n 30                       # scripted demos -> gym/data/banana_plate
dreamgym record  --task banana_plate --mix banana_plate:30,apple_bowl:30,orange_bowl:30 --vary-prompts --out gym/data/fruit
dreamgym finetune gym/data/banana_plate --validate-only           # then without the flag: upload + train; --status
```

Policies: `dream-1` (= `dreamlabs/dream-1-beta`), `dream:<model id>` (flash, a fine-tune), `scripted[:noise_m]`.

**Where things go:** dreamkit's `runs/` (one dreamkit run per variant, source `gym`), and beside dreamkit.yaml `gym/runs/<id>/` (episodes.jsonl, summary.json, gifs/, report.html), `gym/data/<dataset>/`
(LeRobot v2.1 at 10 fps, VP9; state `.dreamgym/` beside it). `gym/` is ignored by git.

**Simulator:** found in this order: `DREAMGYM_SIM_DIR`, `sim_path` of `./dreamkit.yaml`, the `dream_examples`
package, `../dreamlabs/...` next to the repo. It is imported, never edited.

**Rules kept:** the arm, cameras and simulator files are Dream Labs' as shipped; every Dream-1 chunk goes through
`dream_control.validate_chunk`; only objects are moved (randomisation). The scripted expert and Teleop use IK on a
scratch copy of the model data and send joint rows through `env.step` like any policy.

**Code:** `core.py` (Task / Policy contracts) · `tasks.py` (pick-and-place tasks, prompt paraphrases) ·
`policies/` (Dream-1, scripted IK expert) · `kit.py` (dreamkit.wrap around any policy) · `runner.py` (one episode through dreamkit) · `arena.py` / `compare.py` · `stats.py`
(Wilson, McNemar, Fisher) · `lerobot_writer.py` · `dataset.py` (balance, live progress, removing episodes) · `finetune.py` · `web/` (router, page, teleop) · `app.py` (the app plugin).

Known limits: the scripted expert plans `banana_plate`, `apple_bowl` and `orange_bowl`, and custom tasks with any fruit and target (apple onto the plate: 20 of 24 scenes; it fails when the apple lies furthest right, out of the plate-side arm's reach — recording keeps successes only) (round fruit: each arm grips at the yaw its wrist reaches; carry and retreat over the bowl stay within ~0.14 m, the highest gripper-down pose the arm reaches there; if the nearer arm cannot reach every pose the other arm is used);
recovery acts on Dream-1 and on the scripted expert (a new plan from where the arms are), not on Teleop; the success thresholds in `scene.py` come from the scene geometry; Teleop's key directions follow the
top camera and are not tuned on hardware; fine-tuning on simulator data helps in the simulator, not necessarily on a
real robot.
