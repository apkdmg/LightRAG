#!/usr/bin/env bash
#
# pre-merge-snapshot.sh — Consistent cold snapshot of the LightRAG production
# stack (Full/GPU preset) before the upstream v1.5.0.x catch-up merge.
#
# See docs/UPSTREAM_CATCHUP_PLAN.md Phase 0.7 and docs/ROLLBACK.md.
#
# What it captures (one consistent point, stack stopped):
#   - Postgres  : logical pg_dump (portable)  + physical volume tar
#   - Neo4j     : physical volume tar (neo4j_data)
#   - Milvus    : physical volume tars (milvus_data + etcd + minio — all 3
#                 must be captured together or the vector index is useless)
#   - File dirs : ./data/{rag_storage,inputs,prompts} (bind mounts)
#   - Manifest  : image digests, artifact sha256sums, env snapshot
#
# Restore is the reverse — see docs/ROLLBACK.md Scenario C.
#
# USAGE (run on the production host, from the repo/deploy dir):
#   scripts/backup/pre-merge-snapshot.sh
#   COMPOSE_FILE=docker-compose-full.yml ENV_FILE=.env BACKUP_ROOT=/mnt/backups \
#       scripts/backup/pre-merge-snapshot.sh
#   DRY_RUN=1 scripts/backup/pre-merge-snapshot.sh     # print actions, change nothing
#
# This script STOPS the stack for the duration of the physical tars (the
# pg_dump runs first while Postgres is still up). Expect a few minutes of
# downtime proportional to data size. It restarts the stack at the end, even
# on error (trap).

set -euo pipefail

# ---------------------------------------------------------------------------
# Config (override via environment)
# ---------------------------------------------------------------------------
COMPOSE_FILE="${COMPOSE_FILE:-docker-compose-full.yml}"
ENV_FILE="${ENV_FILE:-env.docker-compose-full}"
BACKUP_ROOT="${BACKUP_ROOT:-./backups}"
DRY_RUN="${DRY_RUN:-0}"

# Service names as defined in the compose file
SVC_APP="lightrag"
SVC_PG="postgres"
SVC_NEO4J="neo4j"
SVC_MILVUS="milvus"
SVC_ETCD="milvus-etcd"
SVC_MINIO="milvus-minio"

# Bind-mounted host dirs to capture (relative to deploy dir)
FILE_DIRS=("./data/rag_storage" "./data/inputs" "./data/prompts")

# In-container mount paths (must match the compose volume targets)
PG_MOUNT="/var/lib/postgresql"
NEO4J_MOUNT="/data"
MILVUS_MOUNT="/var/lib/milvus"
ETCD_MOUNT="/etcd"          # compose: milvus-etcd_data:/etcd  (-data-dir /etcd)
MINIO_MOUNT="/minio_data"   # compose: milvus-minio_data:/minio_data

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
log()  { printf '\033[36m[snapshot]\033[0m %s\n' "$*"; }
warn() { printf '\033[33m[snapshot:warn]\033[0m %s\n' "$*" >&2; }
die()  { printf '\033[31m[snapshot:error]\033[0m %s\n' "$*" >&2; exit 1; }

run() {
  if [[ "$DRY_RUN" == "1" ]]; then
    printf '\033[90m  DRY: %s\033[0m\n' "$*"
  else
    eval "$@"
  fi
}

DC=(docker compose -f "$COMPOSE_FILE")
[[ -f "$ENV_FILE" ]] && DC+=(--env-file "$ENV_FILE")

cid() {
  # Resolve a (possibly stopped) container id for a service. Empty if absent.
  "${DC[@]}" ps -aq "$1" 2>/dev/null | head -n1
}

started_stack=0
restart_stack() {
  if [[ "$started_stack" == "1" ]]; then
    log "Restarting stack..."
    run "${DC[*]} start"
  fi
}
trap restart_stack EXIT

# ---------------------------------------------------------------------------
# Preflight
# ---------------------------------------------------------------------------
command -v docker >/dev/null || die "docker not found on PATH"
docker compose version >/dev/null 2>&1 || die "docker compose v2 not available"
[[ -f "$COMPOSE_FILE" ]] || die "compose file not found: $COMPOSE_FILE (set COMPOSE_FILE=)"

# Load PG creds from env file if present (for pg_dump). Don't fail if absent —
# pg_dump step will be skipped with a warning.
PG_USER=""; PG_DB=""
if [[ -f "$ENV_FILE" ]]; then
  # shellcheck disable=SC1090
  set -a; source "$ENV_FILE" 2>/dev/null || true; set +a
  PG_USER="${POSTGRES_USER:-}"
  PG_DB="${POSTGRES_DATABASE:-${POSTGRES_DB:-}}"
fi

TS="$(date +%Y%m%d-%H%M%S)"
BACKUP_DIR="${BACKUP_ROOT}/pre-merge-${TS}"
log "Backup target: ${BACKUP_DIR}"
log "Compose file:  ${COMPOSE_FILE}    Env file: ${ENV_FILE}"
run "mkdir -p '${BACKUP_DIR}'"

MANIFEST="${BACKUP_DIR}/MANIFEST.txt"
record() { [[ "$DRY_RUN" == "1" ]] && return 0; printf '%s\n' "$*" >> "$MANIFEST"; }

record "LightRAG pre-merge snapshot"
record "timestamp: ${TS}"
record "compose_file: ${COMPOSE_FILE}"
record "host: $(hostname)"
record "git_head: $(git rev-parse HEAD 2>/dev/null || echo 'n/a')"
record "anchor_tag: pre-1.5.0.3-merge"
record ""
record "== image digests =="

# ---------------------------------------------------------------------------
# Record image digests (so restore uses identical images)
# ---------------------------------------------------------------------------
for svc in "$SVC_APP" "$SVC_PG" "$SVC_NEO4J" "$SVC_MILVUS" "$SVC_ETCD" "$SVC_MINIO"; do
  c="$(cid "$svc" || true)"
  if [[ -n "$c" ]]; then
    img="$(docker inspect --format '{{.Config.Image}} {{index .Image}}' "$c" 2>/dev/null || echo '?')"
    log "image ${svc}: ${img}"
    record "${svc}: ${img}"
  else
    warn "service '${svc}' has no container (not running this preset?) — skipping"
    record "${svc}: ABSENT"
  fi
done
record ""

# ---------------------------------------------------------------------------
# Step 1 — Logical pg_dump (Postgres still UP, app stopped to freeze writes)
# ---------------------------------------------------------------------------
log "Stopping app container '${SVC_APP}' to freeze writes (DBs stay up for dump)..."
run "${DC[*]} stop ${SVC_APP}"
started_stack=1   # from here, EXIT trap will restart the stack

PG_CID="$(cid "$SVC_PG" || true)"
if [[ -n "$PG_CID" && -n "$PG_USER" && -n "$PG_DB" ]]; then
  log "pg_dump ${PG_DB} (user ${PG_USER}) -> postgres-${PG_DB}.sql.gz"
  run "docker exec '${PG_CID}' pg_dump -U '${PG_USER}' -d '${PG_DB}' --no-owner --clean --if-exists \
        | gzip > '${BACKUP_DIR}/postgres-${PG_DB}.sql.gz'"
else
  warn "Skipping pg_dump (no PG container or POSTGRES_USER/DATABASE unset in ${ENV_FILE})"
  record "pg_dump: SKIPPED"
fi

# ---------------------------------------------------------------------------
# Step 2 — Stop the whole stack for consistent physical tars
# ---------------------------------------------------------------------------
log "Stopping full stack for consistent physical snapshot..."
run "${DC[*]} stop"

# Tar a named volume by borrowing it from its (stopped) container.
tar_volume() {
  local svc="$1" mount="$2" out="$3"
  local c; c="$(cid "$svc" || true)"
  if [[ -z "$c" ]]; then warn "no container for ${svc}; skipping ${out}"; record "${out}: SKIPPED"; return 0; fi
  log "tar ${svc}:${mount} -> ${out}"
  run "docker run --rm --volumes-from '${c}' -v '$(cd "${BACKUP_DIR}" 2>/dev/null && pwd || echo "${BACKUP_DIR}")':/backup alpine \
        tar czf '/backup/${out}' -C '${mount}' ."
  record "${out}: $(_sha "${BACKUP_DIR}/${out}")"
}

_sha() { [[ "$DRY_RUN" == "1" ]] && { echo "(dry)"; return; }; shasum -a 256 "$1" 2>/dev/null | awk '{print $1}' || echo "?"; }

record "== artifact sha256 =="
tar_volume "$SVC_PG"     "$PG_MOUNT"     "postgres_data.tar.gz"
tar_volume "$SVC_NEO4J"  "$NEO4J_MOUNT"  "neo4j_data.tar.gz"
tar_volume "$SVC_MILVUS" "$MILVUS_MOUNT" "milvus_data.tar.gz"
tar_volume "$SVC_ETCD"   "$ETCD_MOUNT"   "milvus-etcd_data.tar.gz"
tar_volume "$SVC_MINIO"  "$MINIO_MOUNT"  "milvus-minio_data.tar.gz"

# ---------------------------------------------------------------------------
# Step 3 — Bind-mounted file dirs (host-side tar)
# ---------------------------------------------------------------------------
for d in "${FILE_DIRS[@]}"; do
  if [[ -d "$d" ]]; then
    base="$(basename "$d")"
    log "tar host dir ${d} -> files-${base}.tar.gz"
    run "tar czf '${BACKUP_DIR}/files-${base}.tar.gz' -C '$(dirname "$d")' '${base}'"
    record "files-${base}.tar.gz: $(_sha "${BACKUP_DIR}/files-${base}.tar.gz")"
  else
    warn "file dir not found: ${d} (skipping)"
    record "files-$(basename "$d"): SKIPPED (not found)"
  fi
done

# ---------------------------------------------------------------------------
# Done — trap restarts the stack
# ---------------------------------------------------------------------------
log "Snapshot complete: ${BACKUP_DIR}"
if [[ "$DRY_RUN" != "1" ]]; then
  log "Contents:"
  ls -lh "${BACKUP_DIR}"
  log "Verify the MANIFEST and copy this directory OFF the host (object storage / another machine)."
fi
log "Stack will be restarted by the exit trap."
