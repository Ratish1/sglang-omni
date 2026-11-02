"""One SGLang dumper directory per process, so the thinker's and the talker's dumps do not overwrite each other."""

import os

base = os.environ.get("DUMPER_BASE")
if base:
    os.environ["DUMPER_DIR"] = os.path.join(base, str(os.getpid()))
