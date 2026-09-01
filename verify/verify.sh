#!/usr/bin/env bash
# Recompute the published numbers in every language here and require agreement.
#
# Every figure in README.md and RESULTS.md is produced by one program, the
# Python under src/loraft/. The tables are rendered from summary_*.json, the
# figures are drawn from the same file, and `make report-check` compares the
# document to the generator that wrote it. Nothing in that loop ever asked
# whether the aggregation was right, because everything in it reads the same
# output. These are independent implementations from reports/preds_*.jsonl and
# reports/train_log.csv, so a mistake in the Python would have to be repeated
# identically in eight languages to survive.
#
# Each check is skipped with a clear message if its toolchain is absent, so this
# runs on a laptop with only some of them. CI has all of them.
set -uo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$root"

pass=0 fail=0 skip=0

run () {
    local name="$1" tool="$2"; shift 2
    printf '\n=== %s ===\n' "$name"
    if ! command -v "$tool" >/dev/null 2>&1; then
        printf 'skipped: %s is not installed\n' "$tool"
        skip=$((skip + 1)); return
    fi
    if "$@"; then pass=$((pass + 1)); else fail=$((fail + 1)); fi
}

# SQL has no way to set an exit status, so it prints one MISMATCH line per
# disagreement and the count is read back here. sqlite3 reads stdin, which
# inside a script is the rest of this file, so it gets /dev/null instead; and it
# writes CRLF, so the carriage returns come out before anything is matched.
check_sql () {
    local out
    out=$(sqlite3 -init verify/summary.sql :memory: "" < /dev/null 2>/dev/null | tr -d '\r')
    printf '  %s\n' "$out"
    case "$out" in
        *"0 mismatches"*) return 0 ;;
        *) return 1 ;;
    esac
}

check_c () {
    cc -std=c99 -O2 -Wall -Wextra -Wpedantic -Werror \
       -o "${TMPDIR:-/tmp}/loraft_score" verify/score.c -lm || return 1
    "${TMPDIR:-/tmp}/loraft_score" "$root"
}

check_go   () { ( cd verify/gocheck && go run . -root "$root" ); }
check_rust () { ( cd verify/paired && cargo run --release --quiet -- "$root" ); }

run "SQL, the four summaries"        sqlite3 check_sql
run "C, the scoring kernel"          cc      check_c
run "Go, file and document validation" go    check_go
run "R, intervals and exact tests"   Rscript Rscript verify/inference.R "$root"
run "Rust, exhaustive permutation"   cargo   check_rust
run "JavaScript, the training log"   node    node verify/trainlog.js "$root"
run "Ruby, the prose claims"         ruby    ruby verify/claims.rb "$root"
run "Java, rebuilding RESULTS.md"    java    java verify/Verify.java "$root"

printf '\n%s\n' "----------------------------------------"
printf '%d passed, %d failed, %d skipped\n' "$pass" "$fail" "$skip"
[ "$fail" -eq 0 ] || exit 1
[ "$pass" -gt 0 ] || { echo "nothing ran"; exit 1; }
