# Rollback Runbook: Upstream Catch-Up Merge

This runbook is the operational counterpart to `docs/UPSTREAM_CATCHUP_PLAN.md`.
It tells you exactly what to do when the v1.5.0.x catch-up merge needs to be
undone — at any stage.

> **Read this end-to-end once before you need it.** Reading a runbook for the
> first time under production-incident stress is how rollbacks go wrong.

---

## 0. Decide if rollback is the right move

Don't rollback reflexively. Walk through these questions first:

1. **Is the system actually broken, or just behaving differently?**
   Upstream may have changed defaults you weren't expecting (env vars renamed,
   response shapes adjusted, etc.). Check `docs/MIGRATION_TO_1.5.0.md` and the
   upstream changelog before assuming regression.

2. **Can it be fixed forward in minutes?** A small config change or a one-line
   patch is faster than a full rollback + redeploy cycle.

3. **Is data integrity at risk?** If the merged build is corrupting data on
   disk *right now*, prioritize taking the deployment **down** (stop
   ingestion) before deciding rollback vs forward-fix. Stop the bleeding first.

If the answer to (3) is yes, or (2) is "no, this needs more than 30 minutes,"
proceed with rollback.

---

## 1. Identify your scenario

Pick the scenario closest to your current state.

| Scenario | Where merge lives | Where prod runs | Section |
|---|---|---|---|
| A | `upstream-catchup` branch only | pre-merge build | [Scenario A](#scenario-a-merge-not-yet-on-main) |
| B | `main` (pushed) | pre-merge build | [Scenario B](#scenario-b-merge-on-main-not-deployed) |
| C | `main` (pushed) | merged build | [Scenario C](#scenario-c-merge-deployed-to-production) |

---

## 2. The rollback anchor

All scenarios use the same anchor: **annotated tag `pre-1.5.0.3-merge`**, created
in Phase 0.2 of the catch-up plan, pointing at commit `ffbacfb7` on `main`.

```bash
# Confirm anchor exists locally
git show pre-1.5.0.3-merge --stat | head

# Confirm anchor exists on origin
git ls-remote --tags origin | grep pre-1.5.0.3-merge
```

If either is missing, **stop** — the rollback infrastructure is incomplete and
you need to recreate the anchor (see `docs/UPSTREAM_CATCHUP_PLAN.md` Phase 0.2)
before continuing.

---

## Scenario A — Merge not yet on `main`

Easiest case. The merge lives only on the `upstream-catchup` side branch and
nothing has touched `main` yet.

```bash
# Discard the side branch entirely
git switch main
git branch -D upstream-catchup

# Optionally drop intermediate merge-step-N tags
git tag -d merge-step-1 merge-step-2 merge-step-3 merge-step-4 2>/dev/null || true

# If the side branch was pushed
git push origin --delete upstream-catchup
```

No production impact. Re-attempt the catch-up later from a fresh `upstream-catchup`.

---

## Scenario B — Merge on `main`, not deployed

`main` has been fast-forwarded onto the merged result but production is still
running the pre-merge build. No data has been touched by the new code.

```bash
# Reset main to the anchor
git switch main
git reset --hard pre-1.5.0.3-merge

# Force-push (with-lease prevents clobbering anyone else's push)
git push --force-with-lease origin main
```

After this:
- `git log -1 main` should show commit `ffbacfb7`.
- `git rev-list --count main..upstream/main` should be back to 135.
- GitHub fork banner should read "59 ahead, 135 behind" again.

No data restore needed because production never ran the merged build.

---

## Scenario C — Merge deployed to production

Hardest case. The merged build has been serving traffic, possibly mutating
storage. Steps must happen in this order:

### C.1 Stop the bleeding

Take the merged deployment offline so it cannot keep writing. Pick whichever
applies:

```bash
# Docker compose
docker compose down

# Kubernetes
kubectl scale deployment/lightrag-api --replicas=0

# Single-process / systemd
systemctl stop lightrag-server
```

### C.2 Decide: data restore needed?

| Symptom | Restore data? |
|---|---|
| Schema unchanged, code-only regression | **No** — keep current data |
| New schema fields added but populated lazily | **No** — keep current data |
| Existing data mutated/migrated by merged build | **Yes** — restore from Phase 0.7 snapshot |
| Suspected corruption | **Yes** — restore |
| Unsure | **Yes** — err toward restore; the cost of an unneeded restore is hours, the cost of running on corrupt data is days |

### C.3 Reset `main` (same as Scenario B)

```bash
git switch main
git reset --hard pre-1.5.0.3-merge
git push --force-with-lease origin main
```

### C.4 Restore storage volumes (if C.2 = yes)

The snapshot was produced by `scripts/backup/pre-merge-snapshot.sh` into a
`pre-merge-<timestamp>/` directory (Phase 0.7). It contains, for the Full/GPU
preset: `postgres_data.tar.gz`, `neo4j_data.tar.gz`, `milvus_data.tar.gz`,
`milvus-etcd_data.tar.gz`, `milvus-minio_data.tar.gz`, the `files-*.tar.gz`
bind-dir archives, a portable `postgres-<db>.sql.gz` logical dump, and a
`MANIFEST.txt` recording image digests + sha256sums.

> **Restore onto the rolled-back (pre-merge) stack** — same image digests as
> recorded in MANIFEST.txt. Physical volume restore requires matching images.

```bash
SNAP=/path/to/pre-merge-<timestamp>     # the snapshot dir
DC="docker compose -f docker-compose-full.yml --env-file env.docker-compose-full"

# 0. Stack must be DOWN so volumes are not in use
$DC down                                 # NOTE: down, not stop — recreates containers fresh

# 1. Restore each named volume by wiping + extracting into a borrowed mount.
#    Bring the (empty) containers into existence first so volumes exist:
$DC create

restore_vol() {  # <service> <in-container-mount> <archive>
  local c; c="$($DC ps -aq "$1")"
  docker run --rm --volumes-from "$c" -v "$SNAP":/backup alpine \
    sh -c "rm -rf ${2:?}/* ${2}/..?* ${2}/.[!.]* 2>/dev/null; tar xzf /backup/$3 -C $2"
}
restore_vol postgres      /var/lib/postgresql "postgres_data.tar.gz"
restore_vol neo4j         /data               "neo4j_data.tar.gz"
restore_vol milvus        /var/lib/milvus     "milvus_data.tar.gz"
restore_vol milvus-etcd   /etcd               "milvus-etcd_data.tar.gz"
restore_vol milvus-minio  /minio_data         "milvus-minio_data.tar.gz"

# 2. Restore bind-mounted file dirs (host-side)
for base in rag_storage inputs prompts; do
  rm -rf "./data/$base"
  tar xzf "$SNAP/files-$base.tar.gz" -C ./data
done

# 3. Bring the stack up
$DC up -d
```

**Alternative — logical Postgres restore** (use if the physical tar won't mount,
e.g. image mismatch). Restores into a running Postgres:

```bash
$DC up -d postgres
gunzip -c "$SNAP/postgres-<db>.sql.gz" | \
  docker exec -i "$($DC ps -aq postgres)" psql -U <user> -d <db>
```

> Milvus + etcd + minio must be restored **as a set** — etcd holds the
> collection metadata, minio the segment objects, and `milvus_data` the local
> state. Restoring only one corrupts the vector store.

### C.5 Redeploy on the pre-merge build

```bash
# Pull/rebuild image at the rolled-back main
docker compose build
docker compose up -d

# Or k8s
kubectl scale deployment/lightrag-api --replicas=<original>
```

### C.6 Verify

```bash
# Smoke script — must pass against the rolled-back deployment
python tests/enterprise_smoke.py \
  --base-url <prod-url> \
  --username <user> --password <pass> \
  --admin-username <admin> --admin-password <adminpass> \
  --json
```

Compare against `baselines/pre-merge/` if you have ambiguity about expected
results.

### C.7 Sanity check user-visible functionality

Beyond the smoke script:
- Log into the WebUI; confirm dashboards render
- Run a real query in each mode (`naive`, `local`, `global`, `hybrid`, `mix`)
- Confirm document count in `/documents/status_counts` matches pre-merge
- Spot-check a known document is queryable

### C.8 Communicate

- Notify whoever needs to know that rollback happened
- Update incident channel / status page if applicable
- Capture timeline of: when merge was deployed, when problem was detected,
  when rollback completed

---

## 3. After rollback — before any retry

Do **not** immediately re-attempt the catch-up. The rollback exposed a real
defect; jumping back in without understanding it usually reproduces it.

1. **Capture evidence** before the merged build is gone:
   - Save logs (`docker logs <container> > /tmp/incident-logs.txt`)
   - If you have it, save error rate / latency graphs from the time window
   - Note exactly which user actions reproduced the failure
2. **Root-cause** what went wrong. The most common patterns:
   - Conflict resolution flipped a check (e.g., auth bypass)
   - Upstream schema change applied automatically and broke existing rows
   - Test gap missed an enterprise path
3. **Update the plan**:
   - Add the failure mode to `docs/UPSTREAM_CATCHUP_PLAN.md` decision log
   - Add a new check to `tests/enterprise_smoke.py` covering it
   - Capture an updated pre-merge baseline if anything in the working tree
     was patched as a forward-fix
4. **Re-rehearse** the merge on staging with the new test in place. Only
   reattempt production cutover after staging is green on the fuller test set.

---

## 4. Quick reference card

```
# Just reset main, nothing else:
git reset --hard pre-1.5.0.3-merge && git push --force-with-lease origin main

# Verify anchor:
git show pre-1.5.0.3-merge --stat | head
git ls-remote --tags origin | grep pre-1.5.0.3-merge

# Smoke check rolled-back deployment:
python tests/enterprise_smoke.py --base-url <url> --username <u> --password <p>
```

The anchor (`pre-1.5.0.3-merge`) **must not be deleted** until the catch-up is
fully stable in production for at least 30 days (Phase 7.3 of the plan).
