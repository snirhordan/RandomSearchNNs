#!/usr/bin/env bash
# Tile frames sampled every $2 seconds from a rendered scene into contact sheets.
#   ./contact.sh S02_Teacher 4
set -euo pipefail
cd "$(dirname "$0")"
scene="$1"; every="${2:-4}"
mkdir -p build/contact
rm -f build/contact/${scene}_*.png
ffmpeg -loglevel error -y -i "build/videos/${scene}.mp4" \
  -vf "fps=1/${every},scale=640:-1,drawtext=text='%{pts\:hms}':x=8:y=8:fontcolor=yellow:fontsize=18,tile=2x3:padding=4" \
  "build/contact/${scene}_%02d.png"
ls build/contact/${scene}_*.png
