#!/usr/bin/env bash
# End-to-end demo against real processes and the HTTP API: submit a 144-task grid to
# 4 workers, SIGKILL one mid-run, wait for recovery, run the verifier.
# Run: scripts/dev_services.sh start && eval "$(scripts/dev_services.sh env)"; scripts/demo.sh
# Output: stdout and var/demo/transcript.txt (plus per-process logs in var/demo/).
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
PY=.venv/bin/python
PORT=${QG_DEMO_PORT:-8765}
API=http://127.0.0.1:$PORT
KEY=${QG_DEMO_KEY:-dev-key}
OUT=var/demo; mkdir -p "$OUT"
export QG_QUEUE_NAME="qg.demo.$$" QG_ALLOW_FAULT_INJECTION=true QG_LOG_LEVEL=WARNING
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
exec > >(tee "$OUT/transcript.txt") 2>&1

PIDS=()
cleanup() { for p in "${PIDS[@]}"; do kill "$p" 2>/dev/null || true; done; wait "${PIDS[@]}" 2>/dev/null || true; }
trap cleanup EXIT

echo "## 1. migrate + start coordinator, API, 4 workers"
.venv/bin/alembic upgrade head >/dev/null
$PY -m quantgrid.coordinator.main >"$OUT/coordinator.log" 2>&1 & PIDS+=($!)
$PY -m uvicorn quantgrid.api.app:create_app --factory --port "$PORT" --log-level warning >"$OUT/api.log" 2>&1 & PIDS+=($!)
WORKER_PIDS=()
for i in 1 2 3 4; do
  QG_WORKER_ID="demo-w$i" $PY -m quantgrid.workers.main --worker-id "demo-w$i" >"$OUT/worker-$i.log" 2>&1 &
  WORKER_PIDS+=($!); PIDS+=($!)
done
for _ in $(seq 1 100); do curl -fs "$API/readyz" >/dev/null 2>&1 && break; sleep 0.3; done
curl -s "$API/readyz"; echo

echo; echo "## 2. dataset signatures (recorded, checksum-verified)"
SHA=$(curl -s -H "X-API-Key: $KEY" "$API/v1/datasets" | $PY -c "import sys,json; d=json.load(sys.stdin)['items']; [print(i['name'], i['sha256'][:16]+'...', i['n_bars'], 'bars', file=sys.stderr) for i in d]; print([i for i in d if i['name']=='pairs-ou-small-v1'][0]['sha256'])")

echo; echo "## 3. submit a 144-task parameter grid (Idempotency-Key makes retries safe)"
cat >"$OUT/batch.json" <<EOF
{"name": "demo-144", "dataset": {"name": "pairs-ou-small-v1", "sha256": "$SHA"},
 "strategy": {"name": "pairs_zscore", "version": "1"},
 "grid": {"lookback": [20, 30, 40, 60, 80, 120], "entry_z": [1.5, 2.0, 2.5, 3.0], "exit_z": [0.0, 0.5, 1.0]},
 "seeds": [7, 11], "workload": {"bootstrap_samples": 3000}, "cache_policy": "bypass",
 "budget": {"max_runtime_s": 120, "max_memory_mb": 1024}, "max_attempts": 4}
EOF
RESP=$(curl -s -D "$OUT/headers.txt" -X POST "$API/v1/batches" -H "X-API-Key: $KEY" \
  -H "Idempotency-Key: demo-$$" -H 'Content-Type: application/json' --data @"$OUT/batch.json")
head -1 "$OUT/headers.txt"; echo "$RESP" | $PY -m json.tool
BID=$(echo "$RESP" | $PY -c "import sys,json; print(json.load(sys.stdin)['batch_id'])")
echo; echo "re-submitting the same Idempotency-Key -> same batch, no duplicate:"
curl -s -X POST "$API/v1/batches" -H "X-API-Key: $KEY" -H "Idempotency-Key: demo-$$" \
  -H 'Content-Type: application/json' --data @"$OUT/batch.json" | $PY -c "import sys,json; d=json.load(sys.stdin); print('batch_id', d['batch_id'], 'idempotent_replay', d['idempotent_replay'])"

status() { curl -s -H "X-API-Key: $KEY" "$API/v1/batches/$BID" | $PY -c "import sys,json; d=json.load(sys.stdin); c=d['progress']['counts']; print(d['state'], {k:v for k,v in c.items() if v}, 'attempts', d['progress']['attempts'])"; }

echo; echo "## 4. wait for ~20 tasks, then SIGKILL worker demo-w2 while it is running one"
for _ in $(seq 1 200); do
  N=$(curl -s -H "X-API-Key: $KEY" "$API/v1/batches/$BID" | $PY -c "import sys,json; print(json.load(sys.stdin)['progress']['counts']['SUCCEEDED'])")
  [ "$N" -ge 20 ] && break; sleep 0.2
done
status; kill -9 "${WORKER_PIDS[1]}"; echo ">>> SIGKILL demo-w2 (pid ${WORKER_PIDS[1]})"

echo; echo "## 5. progress (leases expire -> coordinator re-queues -> survivors finish)"
for _ in $(seq 1 400); do
  S=$(status); echo "$S"; case "$S" in COMPLETED*|CANCELED*) break;; esac; sleep 2
done

echo; echo "## 6. proof: exactly one published result per expected task"
$PY -m quantgrid.cli verify "$BID"

echo; echo "## 7. failed-over task history (attempt log shows the killed worker)"
TID=$($PY - <<EOF
import os
from sqlalchemy import text
from quantgrid.storage.db import make_engine
with make_engine().connect() as c:
    print(c.execute(text("select task_id from task_attempts where batch_id=:b and status='LEASE_EXPIRED' order by id limit 1"), {"b": "$BID"}).scalar() or "")
EOF
)
if [ -n "$TID" ]; then
  curl -s -H "X-API-Key: $KEY" "$API/v1/tasks/$TID" | $PY -c "
import sys,json; d=json.load(sys.stdin)
print('task', d['task_id'], d['state'], 'workers', d['worker_ids'])
for a in d['attempts']: print('  attempt', a['attempt'], a['worker_id'], a['status'], a['error_code'], a['wall_ms'], 'ms')
print('  artifact', d['artifact']['output_uri'])"
fi

echo; echo "## 8. merged results (top 3 by Sharpe) + reproducibility manifest"
curl -s -H "X-API-Key: $KEY" "$API/v1/batches/$BID/results?limit=1000" | $PY -c "
import sys,json; d=json.load(sys.stdin)
print('complete', d['complete'], 'results', len(d['results']), 'summary', {k: d['summary'][k] for k in ('count','best_ordinal','best_value','p50')})
for r in sorted(d['results'], key=lambda r: -(r['metrics'].get('sharpe') or -9))[:3]: print(' ', r['params'], 'seed', r['seed'], 'sharpe', round(r['metrics']['sharpe'],3))
m=d['manifest']; print('manifest: code_sha', m['code']['code_sha'][:16], 'env_sha', m['environment']['env_sha'][:16], 'dataset', m['dataset']['sha256'][:16])"
echo; echo "transcript: $OUT/transcript.txt"
