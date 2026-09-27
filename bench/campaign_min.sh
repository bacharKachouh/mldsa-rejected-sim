#!/usr/bin/env bash
# Minimal-circuit measurements for the paper's cost section. Bytes and rounds are deterministic,
# so one seed per point.
set -u
cd "$(dirname "$0")/.."
log=bench/results/campaign-min.log
run() { echo "== $*" >> "$log"; python bench/run.py --circuit min "$@" >> "$log" 2>&1 || echo "FAILED: $*" >> "$log"; }
run --N 2 --protocol semi --K 4 --seed 1
run --N 4 --protocol semi --K 4 --seed 1
run --N 2 --protocol mascot --K 4 --seed 1
run --N 8 --protocol semi --K 4 --seed 1
run --N 16 --protocol semi --K 4 --seed 1
echo "== CAMPAIGN DONE" >> "$log"
