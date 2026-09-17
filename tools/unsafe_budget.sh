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

# Count only code constructs: strip `//` comments first, then count `unsafe`
# (so TODO comments mentioning "unsafe" don't move the budget).
BUDGET="${1:-$(tr -d '[:space:]' < "$BUDGET_FILE")}"
COUNT="$(grep -rhE 'unsafe' "$ROOT/crates" --include='*.rs' \
  | sed 's://.*::' \
  | grep -oE '\bunsafe\b' | wc -l | tr -d ' ')"

echo "unsafe occurrences: $COUNT (budget: $BUDGET)"
if [ "$COUNT" -gt "$BUDGET" ]; then
  echo "FAIL: unsafe count $COUNT exceeds budget $BUDGET." >&2
  echo "Refactor instead of adding unsafe, or lower the budget if you removed some." >&2
  exit 1
fi
echo "OK"
