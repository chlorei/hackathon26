"""Evaluation tools: plugins that read the simulator's world (positions, contacts), which a real arm does not report.

dreamkit works from what the arm reports (ctx.robot). These tools exist only to measure it in the simulator, the
right answer to compare with: dreamkit doctor --sim and the replay tests load them (Kit(evaluation=True)); the
working loop refuses them (PLUGIN_STANDARD.md §6).

- success-check: whether the object rests in the target, from physics.
- world-judge: why an attempt failed, from positions and contacts (failure-monitor's former simulator branch).
"""
