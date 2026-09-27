#!/usr/bin/env bash
# Render every scene (default 1080p) with up to $JOBS parallel manimgl processes,
# then assemble the final video.   JOBS=4 QUALITY=--hd ./render_all.sh
set -uo pipefail
cd "$(dirname "$0")"
JOBS="${JOBS:-4}"
QUALITY="${QUALITY:---hd}"
mkdir -p build/logs

# Longest scenes first so the parallel slots stay busy.
SCENES=(
  "scenes_b.py S06_AlignedLoss" "scenes_a.py S05_DenoiserMap" "scenes_a.py S04_SigmaSpace"
  "scenes_a.py S02_Teacher" "scenes_b.py S09_Results" "scenes_b.py S08_Search"
  "scenes_a.py S01_Intro" "scenes_a.py S03_FlowMaps" "scenes_b.py S10_Takeaways"
  "scenes_b.py S07_Sampling"
)

python check_cues.py scenes_a.py scenes_b.py || exit 1

running=0
for entry in "${SCENES[@]}"; do
  set -- $entry
  ( ./render.sh "$1" "$2" $QUALITY > "build/logs/$2.log" 2>&1; echo "$2 exit $?" ) &
  running=$((running + 1))
  if [ "$running" -ge "$JOBS" ]; then
    wait -n
    running=$((running - 1))
  fi
done
wait

if grep -l "Traceback" build/logs/*.log; then
  echo "some scenes failed" >&2
  exit 1
fi
python assemble.py
