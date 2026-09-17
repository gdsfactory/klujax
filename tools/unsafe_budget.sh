#!/usr/bin/env bash
# Enforce the unsafe-code budget (Stage 0 of work.md).
#
# Counts `unsafe` occurrences across the Rust crates and fails if the total
# exceeds the budget in tools/unsafe_budget.txt. The budget is lowered as
# hardening stages land; it must never be raised.
#
# Usage: tools/unsafe_budget.sh [budget]
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BUDGET_FILE="$ROOT/tools/unsafe_budget.txt"

# Count unsafe *operations* (not declarations): `unsafe {` blocks and
# `unsafe impl`. Declaring an `unsafe fn` contract is not itself an operation,
# so counting those would penalise FFI/trait signatures. Strip `//` comments.
BUDGET="${1:-$(tr -d '[:space:]' < "$BUDGET_FILE")}"
COUNT="$(grep -rhE 'unsafe' "$ROOT/crates" --include='*.rs' \
  | sed 's://.*::' \
  | grep -oE 'unsafe\s*(\{|impl)' | wc -l | tr -d ' ')"

echo "unsafe occurrences: $COUNT (budget: $BUDGET)"
if [ "$COUNT" -gt "$BUDGET" ]; then
  echo "FAIL: unsafe count $COUNT exceeds budget $BUDGET." >&2
  echo "Refactor instead of adding unsafe, or lower the budget if you removed some." >&2
  exit 1
fi
echo "OK"
