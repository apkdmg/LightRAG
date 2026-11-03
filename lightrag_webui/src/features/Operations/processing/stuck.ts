import type { Stuck, StuckRun } from '../api'
import type { Drawer } from '../hashState'
import type { RunState } from '../severity'

/** One row of the Stuck table (US-3): the class, what it is about and where it drills to. */
export type StuckRow = {
  key: string
  cls: RunState
  workspace: string
  /** A document (masked name or id), a run id, or nothing (workspace-level flags). */
  docId?: string
  docName?: string | null
  docStatus?: string
  runId?: string | null
  ageSeconds: number | null
  /** Server reason text (already plain language and redacted). */
  reason?: string | null
  since?: string | null
  drawer: Drawer
}

const runRows = (cls: RunState, items: StuckRun[] | null | undefined): StuckRow[] =>
  (items ?? []).map((r, i) => ({
    key: `${cls}:${r.workspace}:${r.run_id ?? i}`,
    cls,
    workspace: r.workspace,
    runId: r.run_id ?? null,
    ageSeconds: typeof r.age_s === 'number' ? r.age_s : null,
    reason: r.reason ?? null,
    since: r.since ?? null,
    drawer: { kind: 'workspace', id: r.workspace }
  }))

/**
 * The five stuck classes in one list, worst first (orphaned, busy without a
 * run, reservations, hung, stale docs). Orphaned runs and stranded
 * reservations only exist under Gunicorn, so single-process mode never shows
 * them, even if a server sends some.
 */
export function stuckRows(data: Stuck | null | undefined, multiWorker: boolean): StuckRow[] {
  if (!data) return []
  const rows: StuckRow[] = [
    ...(multiWorker ? runRows('orphaned', data.orphaned) : []),
    ...runRows('busy_without_run', data.busy_without_run),
    ...(multiWorker ? runRows('reservation', data.reservations) : []),
    ...runRows('hung', data.hung)
  ]
  for (const d of data.stale_docs?.items ?? []) {
    rows.push({
      key: `stale_doc:${d.workspace}:${d.doc_id}`,
      cls: 'stale_doc',
      workspace: d.workspace,
      docId: d.doc_id,
      docName: d.file_path_masked ?? null,
      docStatus: d.status,
      ageSeconds: typeof d.age_s === 'number' ? d.age_s : null,
      since: d.updated_at ?? null,
      drawer: { kind: 'document', workspace: d.workspace, id: d.doc_id }
    })
  }
  return rows
}

/** Rows the other classes contribute (not paged), so the stale-doc pager knows its offset. */
export const runRowCount = (data: Stuck | null | undefined, multiWorker: boolean) =>
  data ? stuckRows({ ...data, stale_docs: { items: [], total: 0 } }, multiWorker).length : 0
