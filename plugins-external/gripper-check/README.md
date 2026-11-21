# dreamkit-gripper-check

A [dreamkit](https://github.com/chlorei/hackathon26) plugin: warns when a gripper closes on nothing (the fingers
shut all the way) and when a grip loosens while the policy still holds it closed.

```bash
dreamkit install gripper-check          # from the catalogue
# or: dreamkit install https://github.com/chlorei/hackathon26/tree/main/plugins-external/gripper-check
```

```yaml
plugins:
  - gripper-check: {empty_below: 0.04}
```

## Settings

<!-- params -->
| param | default | unit | what |
|---|---|---|---|
| `closed_cmd` | `0.3` |  | Gripper command below which the policy wants it shut |
| `empty_below` | `0.04` |  | Measured opening below this while shut means nothing is between the fingers |
| `loosen_by` | `0.08` |  | The fingers opening this much while still shut means the grip is loosening |
| `settle_s` | `0.6` | s | How long the fingers get to close before judging |
<!-- /params -->
