#!/usr/bin/env bash
# Our minimal circuit in every competitor's setting (paper, Section 9).
# One job at a time (memory). Usage: bash bench/campaign_settings.sh [malicious-dm|honest|all]
set -u
cd "$(dirname "$0")/.."
log=bench/results/campaign-settings.log
run() { echo "== $*" >> "$log"; python bench/run.py --circuit min --K 4 --seed 1 --tag=-set "$@" >> "$log" 2>&1 || echo "FAILED: $*" >> "$log"; }
what=${1:-all}

if [ "$what" = malicious-dm ] || [ "$what" = all ]; then
  # dishonest majority, malicious (Mithril's corruption model): the backends MP-SPDZ offers
  run --N 2 --protocol lowgear
  run --N 2 --protocol spdz2k
fi
if [ "$what" = honest ] || [ "$what" = all ]; then
  # honest majority, t < N/2 (Quorus's corruption model)
  for N in 3 4 8 16; do run --N $N --protocol mal-shamir; done
  for N in 3 8; do run --N $N --protocol shamir; done
fi
echo "== DONE $what" >> "$log"
