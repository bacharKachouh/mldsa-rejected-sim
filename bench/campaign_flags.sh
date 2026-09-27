#!/usr/bin/env bash
# Flag-level optimisations of the minimal circuit, N = 2, one job at a time.
set -u
cd "$(dirname "$0")/.."
log=bench/results/campaign-flags.log
run() { echo "== $*" >> "$log"; python bench/run.py --circuit min --seed 1 --tag=-flag "$@" >> "$log" 2>&1 || echo "FAILED: $*" >> "$log"; }
run --N 2 --protocol cowgear --K 4
run --N 2 --protocol highgear --K 4
run --N 2 --protocol lowgear --K 1
echo "== DONE" >> "$log"
