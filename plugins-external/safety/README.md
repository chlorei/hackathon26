# dreamkit-safety — an example `modify_actions` plugin

Caps every joint's speed and keeps both tool points inside a workspace box. It may only make a chunk **stricter**:
the dreamkit manager rejects any change that moves a joint faster than the model asked or leaves the joint limits.

```bash
dreamkit install plugins-external/safety     # check + tests + bench, then install
```

Note: the mock policy's `dropped` jerk is exactly what this plugin slows down, so keep it off for mock demos.

## Settings

<!-- params -->
| param | default | unit | what |
|---|---|---|---|
| `max_speed_rad_s` | `3.0` | rad/s | No joint moves faster than this |
| `box_min_m` | `[-0.6, -0.65, 0.745]` | m | Workspace box corner (x, y, z), world frame |
| `box_max_m` | `[0.6, 0.4, 1.3]` | m | Opposite workspace box corner (x, y, z) |
| `check_box` | `True` |  | Cut a chunk where a tool point would leave the box |
| `warn_every_s` | `2.0` | s | At most one warning of a kind this often |
<!-- /params -->
