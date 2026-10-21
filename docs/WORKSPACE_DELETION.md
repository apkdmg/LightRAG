# Workspace Deletion

With `ENABLE_MULTI_TENANCY=true`, a workspace owner can permanently delete their
own workspace and everything in it, and an administrator can delete any
workspace. Both use the same purge (`lightrag/api/workspace_deletion.py`).

## What is deleted

| Data | Where it lives | How it is removed |
| --- | --- | --- |
| Documents, chunks, entities, relations, doc status | KV / doc-status storage | `drop()` of each storage (rows of this workspace only) |
| Entity, relation and chunk vectors | Vector storage | `drop()`; on Milvus the workspace's collections are then removed |
| Knowledge graph | Graph storage | `drop()`; on Neo4j the workspace's indexes are then removed |
| Cached LLM answers | `llm_response_cache` | `drop()` |
| Uploaded and parsed files | `INPUT_DIR/<workspace>/` | Directory removed |
| File-based storages, per-user API keys | `WORKING_DIR/<workspace>/` | Directory removed |
| BYO provider credentials | `WORKING_DIR/.workspace_providers/<workspace>.json` | File removed |
| Ownership | Workspace registry | Replaced by a tombstone (below) |

Not deleted: the sign-in account itself (Keycloak, or `AUTH_ACCOUNTS`),
server logs, and copies of the data in backups. For an erasure request, data
in backups is gone only once those backups expire, so document your backup
retention.

## Owner: `DELETE /workspace`

The workspace is always the caller's own, taken from the session. The request
carries no workspace ID, so it cannot be pointed at someone else's workspace.

```http
DELETE /workspace
Content-Type: application/json

{"confirm": "DELETE MY WORKSPACE"}
```

The request is refused unless all of these hold:

- the caller is signed in to the WebUI with a password (`/login`) or SSO
  session: not per-user or global API keys, service accounts, raw Keycloak
  access tokens or admin impersonation tokens (403);
- the sign-in happened within the last 15 minutes; otherwise sign in again
  (403, `Sign in again to confirm deleting your workspace`);
- the caller is the workspace's registered owner (403);
- no `X-Target-Workspace` header is sent (400);
- the confirmation phrase matches exactly (400).

On success the session cookies are cleared and the WebUI signs out. In the
WebUI the button is at the bottom of the **Providers** tab, under
**Delete workspace**. `GET /workspace` returns the caller's workspace ID,
whether they may delete it, and how long their recent sign-in stays valid
(`sign_in_seconds_left`).

## Admin: `DELETE /admin/workspaces/{workspace_id}`

Purges any workspace. It returns 404 when the workspace is unknown (not bound,
not loaded and with no directories) and 400 for the server's default workspace.

## Safety

- **Pipeline:** the purge takes the pipeline's destructive slot, like
  `/documents/clear`. While documents are processing, scanning or being
  enqueued it answers 409; retry once the pipeline is idle. During the purge,
  other requests to the workspace get 409.
- **Isolation check:** nothing is dropped if a `*_WORKSPACE` storage override
  is set, or if any storage is not scoped to the workspace being deleted.
  Either would make a drop reach other workspaces.
- **Partial failure:** if any storage fails to drop, files and ownership are
  kept, and the workspace keeps working, so the deletion can be retried. The
  owner endpoint returns 500, and the server log names the failing storage.
- **Old sessions:** a tombstone records the deletion time. Sessions that began
  before it get 401 (`This workspace has been deleted. Sign in again.`), and
  instances cached by other workers are rebuilt. Other workers notice within
  60 seconds.
- **Reuse of the ID:** only the former owner gets the same workspace ID back,
  as a fresh, empty workspace when they sign in again. Anyone else whose name
  maps to it gets a distinct ID, as in a collision.
- **Privacy of tombstones:** a tombstone holds only SHA-256 hashes of the
  workspace ID and the former owner's identity, plus the deletion time
  (`WORKING_DIR/.workspace_tombstones.json`, or the Postgres table
  `LIGHTRAG_WORKSPACE_TOMBSTONES`).
