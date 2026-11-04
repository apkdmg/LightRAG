/**
 * Where the Documents tab's "pipeline active" flag comes from.
 *
 * `/health` reports the pipeline of the server's workspace for this session,
 * never the one an admin acts on (OQ-13: `X-Target-Workspace` goes on
 * Documents calls only). While acting, the flag is read from
 * `/documents/pipeline_status` instead, which carries the header and so
 * describes the target workspace. Its response passes every pipeline_status
 * key through, so the same four fields as `/health`'s `pipeline_active` are
 * there.
 */

/** The pipeline_status keys that make the pipeline "active" (busy, scanning, clearing, or enqueues in flight). */
export type PipelineFlags = {
  busy?: boolean
  scanning?: boolean
  destructive_busy?: boolean
  pending_enqueues?: number
}

/** Same rule as `/health`'s `pipeline_active` (lightrag_server.py). */
export function pipelineActiveFrom(status: PipelineFlags | null | undefined): boolean {
  if (!status) return false
  return !!status.busy || !!status.scanning || !!status.destructive_busy || Number(status.pending_enqueues ?? 0) > 0
}

/**
 * The flag for this check: from `/health` for the own workspace; from the
 * target's pipeline_status while acting. Null when acting and the target
 * cannot be read, so the caller keeps the last value rather than showing
 * the own workspace's.
 */
export async function resolvePipelineActive(
  health: { pipeline_active?: boolean; pipeline_busy?: boolean },
  acting: boolean,
  readTarget: () => Promise<PipelineFlags>
): Promise<boolean | null> {
  if (!acting) return health.pipeline_active ?? !!health.pipeline_busy
  try {
    return pipelineActiveFrom(await readTarget())
  } catch {
    return null
  }
}
