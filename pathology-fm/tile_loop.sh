#!/bin/bash
# Tile slides as they land so the download is not followed by a serial tiling pass.
# The tiler skips in-flight files and already-tiled slides, so repeated runs are safe.
#
# Loop until done, not a fixed pass count: the previous version used `seq 1 80`
# and exited 0 at 37/50 simply because it ran out of passes, which looks
# identical to success in the exit code.
cd "$(dirname "$0")/src" || exit 1
TILES=${C17_ROOT:-$(cd .. && pwd)/data/camelyon17}/tiles     # same default as the Python scripts
TARGET=${TARGET:-50}
STALL_LIMIT=${STALL_LIMIT:-40}          # passes with no progress before giving up
prev=-1
stall=0
pass=0

while :; do
  pass=$((pass + 1))
  n=$(ls "$TILES"/*.npz 2>/dev/null | wc -l | tr -d ' ')
  echo "=== pass $pass : $n/$TARGET tiled at $(date +%H:%M:%S) (stall $stall/$STALL_LIMIT) ==="
  if [ "$n" -ge "$TARGET" ]; then
    echo "DONE all $TARGET slides tiled"
    exit 0
  fi
  ../.venv/bin/python -u wsi.py --max-neg 400 --workers 4 2>&1 | grep -vE "SKIP in flight"
  after=$(ls "$TILES"/*.npz 2>/dev/null | wc -l | tr -d ' ')
  if [ "$after" -eq "$prev" ]; then
    stall=$((stall + 1))
  else
    stall=0
  fi
  prev=$after
  if [ "$stall" -ge "$STALL_LIMIT" ]; then
    echo "STALLED: no new slides for $STALL_LIMIT passes, stopping at $after/$TARGET"
    exit 3
  fi
  sleep 120
done
