#!/usr/bin/env bash
# smoke_test.sh — Automatyczne testy bramki Etapu 1
# Użycie: bash scripts/smoke_test.sh
# Wymaga: działającej aplikacji pod http://127.0.0.1:8000

set -e
BASE="http://127.0.0.1:8000"
PASS=0
FAIL=0
ERRORS=()

green() { echo -e "\033[32m✓\033[0m $1"; }
red()   { echo -e "\033[31m✗\033[0m $1"; }

check() {
  local desc="$1"; local cmd="$2"; local expect="$3"
  result=$(eval "$cmd" 2>/dev/null || true)
  if echo "$result" | grep -q "$expect"; then
    green "$desc"
    PASS=$((PASS+1))
  else
    red "$desc (oczekiwano: '$expect', dostałem: '${result:0:120}')"
    FAIL=$((FAIL+1))
    ERRORS+=("$desc")
  fi
}

echo ""
echo "╔══════════════════════════════════════════╗"
echo "║   SMOKE TEST — EnergyForecast API        ║"
echo "╚══════════════════════════════════════════╝"
echo ""

# 1. Health
echo "--- 1. GET /health ---"
check "status ok"          "curl -sf $BASE/health"         '"status":"ok"'
check "model_loaded true"  "curl -sf $BASE/health"         '"model_loaded":true'
check "sklearn_version"    "curl -sf $BASE/health"         '"sklearn_version"'

# 2. Frontend
echo ""
echo "--- 2. GET / (HTML) ---"
check "strona HTML serwowana"  "curl -sf $BASE/"            '<!DOCTYPE html'
check "tytuł strony"           "curl -sf $BASE/"            'EnergyForecast'

# 3. POST /predict
echo ""
echo "--- 3. POST /predict ---"
PAYLOAD='{"hour":14,"day_of_week":2,"is_holiday_or_weekend":0,"temp_pl":15.3,"wind_pl":12.5,"radiation_pl":320.0,"price_lag_24h":285.50,"price_lag_168h":271.30}'
check "predykcja zwraca JSON" \
  "curl -sf -X POST $BASE/predict -H 'Content-Type: application/json' -d '$PAYLOAD'" \
  '"prediction"'
check "pole status ok" \
  "curl -sf -X POST $BASE/predict -H 'Content-Type: application/json' -d '$PAYLOAD'" \
  '"status":"ok"'

# 4. POST /predict z sample_payload.json
echo ""
echo "--- 4. POST /predict z sample_payload.json ---"
if [ -f "scripts/sample_payload.json" ]; then
  check "sample_payload daje predykcję" \
    "curl -sf -X POST $BASE/predict -H 'Content-Type: application/json' -d @scripts/sample_payload.json" \
    '"prediction"'
else
  red "sample_payload.json nie znaleziony (pomiń lub uruchom notebook)"
  FAIL=$((FAIL+1))
fi

# 5. Walidacja Pydantic (błędne dane → 422)
echo ""
echo "--- 5. Walidacja Pydantic ---"
BAD_PAYLOAD='{"hour":99,"day_of_week":2}'
STATUS=$(curl -s -o /dev/null -w "%{http_code}" -X POST $BASE/predict \
  -H "Content-Type: application/json" -d "$BAD_PAYLOAD" 2>/dev/null || echo "000")
if [ "$STATUS" = "422" ]; then
  green "Błędne dane → 422 Unprocessable Entity"
  PASS=$((PASS+1))
else
  red "Oczekiwano 422, dostałem: $STATUS"
  FAIL=$((FAIL+1))
fi

# 6. GET /predict/next-day
echo ""
echo "--- 6. GET /predict/next-day ---"
check "next-day zwraca JSON" \
  "curl -sf --max-time 60 $BASE/predict/next-day" \
  '"forecasts"'
check "next-day ma datę" \
  "curl -sf --max-time 60 $BASE/predict/next-day" \
  '"date"'

# 7. Sprawdzenie liczby godzin w prognozie
echo ""
echo "--- 7. Liczba godzin w prognozie ---"
N_HOURS=$(curl -sf --max-time 60 $BASE/predict/next-day 2>/dev/null | \
  python3 -c "import sys,json; d=json.load(sys.stdin); print(len(d.get('forecasts',[])))" 2>/dev/null || echo "0")
if [ "$N_HOURS" -ge 1 ] 2>/dev/null; then
  green "Prognoza zawiera $N_HOURS godzin"
  PASS=$((PASS+1))
else
  red "Prognoza nie zawiera godzin (N=$N_HOURS)"
  FAIL=$((FAIL+1))
fi

# Podsumowanie
echo ""
echo "════════════════════════════════════════"
echo "  WYNIK: ${PASS} passed, ${FAIL} failed"
echo "════════════════════════════════════════"
if [ ${#ERRORS[@]} -gt 0 ]; then
  echo "  Błędy:"
  for e in "${ERRORS[@]}"; do echo "    • $e"; done
  echo ""
  exit 1
fi
echo ""
exit 0
