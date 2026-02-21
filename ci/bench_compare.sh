#!/usr/bin/env bash
# Builds two revisions and runs the benchmark scenes on each, alternating, three
# times. Compares the best time of each and fails on a slowdown larger than
# FE_BENCH_TOLERANCE percent. Running both on one machine in one job keeps
# runner-to-runner noise out of the comparison.
set -euo pipefail
base="$1"
head="$2"
tol="${FE_BENCH_TOLERANCE:-5}"
scenes=(scenes/bench/dam_break.scene scenes/bench/sand_storm.scene scenes/bench/demolition_bench.scene)
work="$(mktemp -d)"

build() {
  local rev="$1" out="$2"
  git worktree add --force "$work/$out" "$rev" >/dev/null
  (cd "$work/$out" && cargo build --release -p fe --quiet)
}
build "$base" base
build "$head" head

best() {
  local bin="$1" scene="$2" best=""
  for _ in 1 2 3; do
    t=$("$bin" bench "$scene" --json --threads 2 | python3 -c 'import json,sys; print(json.load(sys.stdin)["ms_per_tick"]["total"])')
    if [ -z "$best" ] || python3 -c "import sys; sys.exit(0 if $t < $best else 1)"; then best="$t"; fi
  done
  echo "$best"
}

status=0
for s in "${scenes[@]}"; do
  b=$(cd "$work/base" && best ./target/release/fe "$PWD/../head/$s")
  h=$(cd "$work/head" && best ./target/release/fe "$s")
  pct=$(python3 -c "print(round(($h - $b) / $b * 100, 1))")
  echo "$s: base ${b} ms/tick, head ${h} ms/tick (${pct}%)"
  if python3 -c "import sys; sys.exit(0 if $pct > $tol else 1)"; then
    if [[ "${FE_COMMIT_MESSAGE:-}" == *"perf-regression-ok:"* ]]; then
      echo "  slower than ${tol}%, accepted by commit message"
    else
      echo "  FAIL: slower than ${tol}%"
      status=1
    fi
  fi
done
exit $status
