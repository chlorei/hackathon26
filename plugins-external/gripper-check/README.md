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
