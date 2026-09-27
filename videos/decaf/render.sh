#!/usr/bin/env bash
# Render one scene headlessly with manimgl (3b1b/manim) under a virtual X server.
#   ./render.sh scenes_a.py S02_Teacher [--hd|-m|-l] [extra manimgl args]
set -euo pipefail
cd "$(dirname "$0")"
MANIMGL="${MANIMGL:-manimgl}"
file="$1"; scene="$2"; shift 2
exec xvfb-run -a -s "-screen 0 1920x1080x24" "$MANIMGL" "$file" "$scene" -w "$@"
