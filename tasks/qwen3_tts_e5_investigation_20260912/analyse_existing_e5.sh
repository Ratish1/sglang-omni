#!/usr/bin/env bash
set -euo pipefail

if [[ "$(uname -s)" != Linux ]]; then
    echo "Run this in the H100 Linux container. Do not run it on the Mac." >&2
    exit 1
fi

E5_INPUT=${1:?Usage: bash analyse_existing_e5.sh /absolute/path/to/e5}
E5_SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
E5_INPUT=$(cd -- "$E5_INPUT" && pwd)
E5_OUTPUT=$(mktemp -d "${E5_INPUT}/../e5-existing-trace-audit.XXXXXX")
export PYTHONDONTWRITEBYTECODE=1
printf 'Analysis directory: %s\n' "$E5_OUTPUT"

command -v nsys >/dev/null
command -v python3 >/dev/null
nsys --version | tee "$E5_OUTPUT/nsys-version.txt"
python3 -c 'import sys; assert sys.version_info >= (3, 11), "Use Python 3.11 or newer"'
python3 - "$E5_OUTPUT/nsys-version.txt" <<'PY'
import re
import sys
from pathlib import Path
s = Path(sys.argv[1]).read_text()
m = re.search(r'(20\d\d)\.(\d+)\.(\d+)', s)
if not m or tuple(map(int, m.groups())) < (2026, 2, 1):
    raise SystemExit('Use Nsight Systems 2026.2.1 or newer to read these reports. No new collection is required.')
PY

for E5_ARM in control early; do
    E5_REPORT="$E5_INPUT/nsight/$E5_ARM/window.nsys-rep"
    test -s "$E5_REPORT"
    test -d "$E5_INPUT/nsight/$E5_ARM/events_window"
    sha256sum "$E5_REPORT" >> "$E5_OUTPUT/input-sha256.txt"

    # The fresh destination preserves all existing reports/exports.
    if ! nsys export --type=sqlite --output="$E5_OUTPUT/$E5_ARM.sqlite" \
        "$E5_REPORT" > "$E5_OUTPUT/$E5_ARM-export.log" 2>&1; then
        cat "$E5_OUTPUT/$E5_ARM-export.log" >&2
        exit 1
    fi
    python3 "$E5_SCRIPT_DIR/inspect_sqlite.py" \
        --db "$E5_OUTPUT/$E5_ARM.sqlite" \
        --events "$E5_INPUT/nsight/$E5_ARM/events_window" \
        --arm "$E5_ARM" --out "$E5_OUTPUT/$E5_ARM" \
        2>&1 | tee "$E5_OUTPUT/$E5_ARM-analysis.log"
done

python3 - "$E5_OUTPUT" <<'PY'
import sys
import tarfile
from pathlib import Path
root = Path(sys.argv[1])
dest = root / 'e5-compact-audit.tar.gz'
with tarfile.open(dest, 'w:gz') as tar:
    for path in sorted(root.rglob('*')):
        if not path.is_file() or path == dest:
            continue
        if path.suffix == '.sqlite' or path.name.endswith('.jsonl.gz'):
            continue
        tar.add(path, arcname=str(path.relative_to(root)), recursive=False)
print(f'Analysis directory: {root}')
print(f'Return this compact bundle: {dest}')
print('Keep the SQLite files and detailed *.jsonl.gz files in the container for request-level inspection.')
PY
