"""Loaded at interpreter start by `dreamkit run`: puts dreamkit on the program's DreamClient (dreamkit/attach.py)."""

import os

if os.environ.get("DREAMKIT_ATTACH") == "1":
    try:
        from dreamkit import attach

        attach.install()
    except Exception as error:  # the program must start either way
        import sys

        print(f"dreamkit: overlay not attached: {error}", file=sys.stderr)
