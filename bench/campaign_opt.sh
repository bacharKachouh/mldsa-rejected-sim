#!/usr/bin/env bash
# Optimised circuit, N = 2, one job at a time (starts after campaign_flags.sh finishes).
set -u
cd "$(dirname "$0")/.."
until grep -q "== DONE" bench/results/campaign-flags.log 2>/dev/null; do sleep 30; done
log=bench/results/campaign-opt.log
run() { echo "== $*" >> "$log"; python bench/run.py --circuit opt --seed 1 --tag=-opt "$@" >> "$log" 2>&1 || echo "FAILED: $*" >> "$log"; }
run --N 2 --protocol semi --K 4
run --N 2 --protocol lowgear --K 4
run --N 2 --protocol lowgear --K 1
run --N 2 --protocol cowgear --K 1
echo "== DONE" >> "$log"
