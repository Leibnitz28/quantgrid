#!/usr/bin/env bash
# Start/stop a local PostgreSQL + Redis for QuantGrid development WITHOUT Docker.
#
#   scripts/dev_services.sh start    # initdb on first run, then start both services
#   scripts/dev_services.sh status
#   scripts/dev_services.sh stop
#   scripts/dev_services.sh env      # print export lines for QG_DATABASE_URL / QG_REDIS_URL
#
# Services listen on 127.0.0.1 only, on non-default ports so they never clobber a
# system PostgreSQL/Redis. Local dev only: PostgreSQL uses trust auth on loopback.
# Use docker-compose.yml for a containerised setup instead.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEV_DIR="${QG_DEV_DIR:-$ROOT/.dev}"
PG_PORT="${QG_DEV_PG_PORT:-5433}"
REDIS_PORT="${QG_DEV_REDIS_PORT:-6380}"
PG_DATA="$DEV_DIR/pgdata"
RUN_DIR="$DEV_DIR/run"
LOG_DIR="$DEV_DIR/logs"
REDIS_DIR="$DEV_DIR/redis"

find_pg_bin() {
  if command -v pg_config >/dev/null 2>&1 && [ -x "$(pg_config --bindir)/pg_ctl" ]; then
    pg_config --bindir; return
  fi
  local candidate
  candidate="$(ls -d /usr/lib/postgresql/*/bin 2>/dev/null | sort -V | tail -1 || true)"
  if [ -n "$candidate" ] && [ -x "$candidate/pg_ctl" ]; then echo "$candidate"; return; fi
  if command -v pg_ctl >/dev/null 2>&1; then dirname "$(command -v pg_ctl)"; return; fi
  echo "ERROR: PostgreSQL server binaries (pg_ctl/initdb) not found" >&2; exit 1
}

PG_BIN="$(find_pg_bin)"

# PostgreSQL refuses to run as root: drop to the 'postgres' OS user when needed.
as_pg() {
  if [ "$(id -u)" = "0" ]; then
    runuser -u postgres -- "$@"
  else
    "$@"
  fi
}

pg_running() { as_pg "$PG_BIN/pg_ctl" -D "$PG_DATA" status >/dev/null 2>&1; }
redis_running() { redis-cli -p "$REDIS_PORT" ping >/dev/null 2>&1; }

start() {
  mkdir -p "$RUN_DIR" "$LOG_DIR" "$REDIS_DIR"
  if [ "$(id -u)" = "0" ]; then chown -R postgres:postgres "$RUN_DIR" "$LOG_DIR"; fi
  if [ ! -s "$PG_DATA/PG_VERSION" ]; then
    mkdir -p "$PG_DATA"
    if [ "$(id -u)" = "0" ]; then chown postgres:postgres "$PG_DATA"; fi
    as_pg "$PG_BIN/initdb" -D "$PG_DATA" -U postgres --auth=trust -E UTF8 --locale=C.UTF-8 >/dev/null
    cat >>"$PG_DATA/postgresql.conf" <<EOF
listen_addresses = '127.0.0.1'
port = $PG_PORT
unix_socket_directories = '$RUN_DIR'
max_connections = 200
shared_buffers = 256MB
log_min_duration_statement = -1
timezone = 'UTC'
EOF
  fi
  if ! pg_running; then
    as_pg "$PG_BIN/pg_ctl" -D "$PG_DATA" -l "$LOG_DIR/postgres.log" -w start >/dev/null
  fi
  # Idempotently create the application role + databases.
  psql_admin() { psql -h 127.0.0.1 -p "$PG_PORT" -U postgres -v ON_ERROR_STOP=1 -qtA "$@"; }
  if [ "$(psql_admin -c "SELECT 1 FROM pg_roles WHERE rolname='quantgrid'")" != "1" ]; then
    psql_admin -c "CREATE ROLE quantgrid LOGIN PASSWORD 'quantgrid' CREATEDB"
  fi
  for db in quantgrid quantgrid_test; do
    if [ "$(psql_admin -c "SELECT 1 FROM pg_database WHERE datname='$db'")" != "1" ]; then
      psql_admin -c "CREATE DATABASE $db OWNER quantgrid"
    fi
  done
  if ! redis_running; then
    redis-server --port "$REDIS_PORT" --bind 127.0.0.1 --daemonize yes \
      --dir "$REDIS_DIR" --appendonly yes --appendfsync everysec --save "" \
      --pidfile "$RUN_DIR/redis.pid" --logfile "$LOG_DIR/redis.log"
    for _ in $(seq 1 50); do redis_running && break; sleep 0.1; done
  fi
  status
}

stop() {
  if redis_running; then redis-cli -p "$REDIS_PORT" shutdown nosave >/dev/null 2>&1 || true; fi
  if pg_running; then as_pg "$PG_BIN/pg_ctl" -D "$PG_DATA" -m fast -w stop >/dev/null; fi
  echo "stopped"
}

status() {
  if pg_running; then echo "postgres: running on 127.0.0.1:$PG_PORT"; else echo "postgres: stopped"; fi
  if redis_running; then echo "redis:    running on 127.0.0.1:$REDIS_PORT"; else echo "redis:    stopped"; fi
}

env_lines() {
  echo "export QG_DATABASE_URL=postgresql+psycopg://quantgrid:quantgrid@127.0.0.1:$PG_PORT/quantgrid"
  echo "export QG_REDIS_URL=redis://127.0.0.1:$REDIS_PORT/0"
  echo "export QG_TEST_DATABASE_URL=postgresql+psycopg://quantgrid:quantgrid@127.0.0.1:$PG_PORT/quantgrid_test"
  echo "export QG_TEST_REDIS_URL=redis://127.0.0.1:$REDIS_PORT/15"
}

case "${1:-}" in
  start) start ;;
  stop) stop ;;
  status) status ;;
  env) env_lines ;;
  *) echo "usage: $0 {start|stop|status|env}" >&2; exit 2 ;;
esac
