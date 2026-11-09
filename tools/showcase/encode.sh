#!/bin/bash
# encode.sh IN.mp4 OUT_BASENAME [TARGET_SECONDS] [CROP_X_CENTER_FRAC]
# Speeds a recorded camera to ~TARGET_SECONDS (default 20) and writes
#   OUT_BASENAME.mp4      1280x720, the master
#   OUT_BASENAME_4x3.mp4  760x570, a drop-in for the project page's task cards
set -euo pipefail
in=$1 out=$2 target=${3:-20} cx=${4:-0.5}
dur=$(ffprobe -v error -show_entries format=duration -of csv=p=0 "$in")
speed=$(python3 -c "print(max(1.0, $dur / $target))")
pts=$(python3 -c "print(1.0 / $speed)")
common="-c:v libx264 -preset slow -crf 20 -pix_fmt yuv420p -movflags +faststart -an"
ffmpeg -v error -y -i "$in" -vf "setpts=$pts*PTS,fps=30" $common "$out.mp4"
# 4:3 crop of the 16:9 frame: 960 wide, centred at CROP_X_CENTER_FRAC of the width.
ffmpeg -v error -y -i "$in" -vf "setpts=$pts*PTS,fps=30,crop=960:720:'min(max(0,iw*$cx-480),iw-960)':0,scale=760:570" $common "${out}_4x3.mp4"
printf '%s  %.1fx  -> %s(.mp4, _4x3.mp4)\n' "$(basename "$in")" "$speed" "$out"
