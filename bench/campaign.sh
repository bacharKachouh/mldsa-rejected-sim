#!/usr/bin/env bash
# Measurement campaign for bench/RESULTS.md. Usage: bash bench/campaign.sh semi|mascot
# Each line of the log: config then the driver's one-line summary (or the error).
set -u
cd "$(dirname "$0")/.."
proto=${1:-semi}
log=bench/results/campaign-$proto.log
run() { echo "== $*" >> "$log"; python bench/run.py "$@" >> "$log" 2>&1 || echo "FAILED: $*" >> "$log"; }

if [ "$proto" = semi ]; then
  for N in 2 4 8 16; do for s in 1 2 3; do run --N $N --protocol semi --K 4 --seed $s; done; done
  for N in 2 4 8 16; do for K in 8 24; do run --N $N --protocol semi --K $K --seed 1; done; done
else
  run --N 2 --protocol mascot --K 4 --seed 1
  run --N 4 --protocol mascot --K 4 --seed 1
  run --N 8 --protocol mascot --K 4 --seed 1
fi
echo "== CAMPAIGN DONE" >> "$log"
