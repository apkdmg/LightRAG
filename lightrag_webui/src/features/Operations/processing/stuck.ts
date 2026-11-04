import type { ReservationFacts } from '../actions'
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
  /** The server's reason code (busy without a run, reservations); translated by stuckWhy, never shown as-is. */
  reasonCode?: string | null
  since?: string | null
  /** The owner process of a run class (orphaned, hung, unresponsive). */
  pid?: number | null
  heartbeatAge?: number | null
  progressAge?: number | null
  /** Unresponsive: age of the owner worker's last heartbeat. */
  workerHeartbeatAge?: number | null
  confirmedBy?: string | null
  /** Reservations: the flag held without an owner. */
  scanning?: boolean | null
  pendingEnqueues?: number | null
  /** Reservations: what the row's owner worker holds on the workspace (Release reservations, OQ-12). */
  reservation?: ReservationFacts | null
  drawer: Drawer
}

/** The stuck classes the server checks (§1.5 US-3), as `checked_classes` spells them. */
export const STUCK_CLASSES = ['orphaned', 'hung', 'stale_docs', 'busy_without_run', 'reservations'] as const

const n = (v: number | null | undefined) => (typeof v === 'number' && Number.isFinite(v) ? v : null)

/** Reason codes that name a run the run table does not hold: the row opens the workspace instead. */
const NO_RUN_ROW = new Set(['run_unknown'])

const runRows = (cls: RunState, items: StuckRun[] | null | undefined): StuckRow[] =>
  (items ?? []).map((r, i) => {
    // A run whose owner stopped beating is listed with the hung runs.
    const rowCls: RunState = cls === 'hung' && (r.owner_unresponsive || r.state === 'unresponsive') ? 'unresponsive' : cls
    const opensRun = !!r.run_id && !NO_RUN_ROW.has(r.reason_code ?? '')
    return {
      key: `${rowCls}:${r.workspace}:${r.run_id ?? i}`,
      cls: rowCls,
      workspace: r.workspace,
      runId: r.run_id ?? null,
      ageSeconds: n(r.age_s),
      reasonCode: r.reason_code ?? null,
      since: r.since ?? null,
      pid: n(r.pid),
      heartbeatAge: n(r.heartbeat_age_s),
      progressAge: n(r.progress_age_s),
      workerHeartbeatAge: n(r.worker_heartbeat_age_s),
      confirmedBy: r.confirmed_by ?? null,
      scanning: r.scanning ?? null,
      pendingEnqueues: n(r.pending_enqueues),
      // A run opens its Run drawer; workspace-level flags open the workspace.
      drawer: opensRun ? { kind: 'run', id: r.run_id as string } : { kind: 'workspace', id: r.workspace }
    }
  })

/**
 * Reservation rows (OQ-12). A pending-enqueue row names each confirmed-dead
 * worker's share (`dead_owners`) and becomes one row per worker, since a
 * release frees one worker's holdings. Each row carries what its worker
 * holds on the workspace: the scan it owns (from the scanning row) and its
 * share (from the pending row), so a release sends both expected values.
 */
function reservationRows(items: StuckRun[] | null | undefined): StuckRow[] {
  const list = items ?? []
  const scanOf = (ws: string, worker: string) => list.find((r) => r.workspace === ws && r.scanning && r.owner_worker_id === worker)
  const shareOf = (ws: string, worker: string) => {
    for (const r of list) {
      if (r.workspace !== ws || r.scanning) continue
      const hit = (r.dead_owners ?? []).find((o) => o.worker_id === worker)
      if (hit && typeof hit.count === 'number') return hit.count
    }
    return 0
  }
  const facts = (ws: string, worker: string | null, reasonCode: string | null): ReservationFacts => {
    const scan = worker ? scanOf(ws, worker) : undefined
    return {
      workerId: worker,
      scanning: !!scan,
      scanStartedAt: scan ? (scan.scan_started_at ?? scan.since ?? null) : null,
      share: worker ? shareOf(ws, worker) : 0,
      reasonCode
    }
  }
  const out: StuckRow[] = []
  runRows('reservation', list).forEach((row, i) => {
    const r = list[i]
    const owners = !r.scanning && Array.isArray(r.dead_owners) ? r.dead_owners.filter((o) => o && typeof o.worker_id === 'string') : []
    if (owners.length) {
      for (const o of owners) {
        out.push({
          ...row,
          key: `${row.key}:${o.worker_id}`,
          pendingEnqueues: typeof o.count === 'number' ? o.count : row.pendingEnqueues,
          reservation: facts(r.workspace, o.worker_id, r.reason_code ?? null)
        })
      }
      return
    }
    // A scanning row names its owner; a counter held with no owner shares names nobody.
    out.push({ ...row, reservation: facts(r.workspace, r.scanning ? (r.owner_worker_id ?? null) : null, r.reason_code ?? null) })
  })
  return out
}

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
    ...(multiWorker ? reservationRows(data.reservations) : []),
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

/**
 * The classes the server checked, limited to the ones this process mode can
 * have. A server that does not say (one release behind) checked stale docs only.
 */
export function checkedClasses(data: Stuck | null | undefined, multiWorker: boolean): string[] {
  const listed = data?.checked_classes ?? ['stale_docs']
  return STUCK_CLASSES.filter((c) => listed.includes(c) && (multiWorker || (c !== 'orphaned' && c !== 'reservations')))
}

/** Reason codes the server sends for each class (operations.stuck.why.busy_<code>, reservationScan_<code>). */
const BUSY_REASONS = new Set(['no_run', 'run_unknown', 'run_finished'])
const SCAN_REASONS = new Set(['owner_dead', 'owner_unknown', 'no_owner'])

/**
 * The plain-language "why" (US-3) as an i18n key under operations.stuck.why.*
 * and its values, built from the row's class, reason code and fields. A
 * code this release does not know falls back to the class's general text.
 */
export function stuckWhy(row: StuckRow): { key: string; values: Record<string, string | number> } {
  const pid = row.pid !== null && row.pid !== undefined
  switch (row.cls) {
    case 'orphaned':
      return {
        key: pid ? 'orphaned' : 'orphanedNoPid',
        values: { pid: row.pid ?? '', how: row.confirmedBy ?? 'pid_gone' }
      }
    case 'hung':
      return { key: pid ? 'hung' : 'hungNoPid', values: { pid: row.pid ?? '' } }
    case 'unresponsive':
      return { key: pid ? 'unresponsive' : 'unresponsiveNoPid', values: { pid: row.pid ?? '' } }
    case 'reservation':
      if (row.scanning) {
        return { key: SCAN_REASONS.has(row.reasonCode ?? '') ? `reservationScan_${row.reasonCode}` : 'reservationScan', values: {} }
      }
      return { key: row.reasonCode === 'owner_dead' ? 'reservationEnqueueDead' : 'reservationEnqueue', values: { count: row.pendingEnqueues ?? 0 } }
    case 'busy_without_run':
      return { key: BUSY_REASONS.has(row.reasonCode ?? '') ? `busy_${row.reasonCode}` : 'busy_without_run', values: {} }
    default:
      return { key: 'stale_doc', values: { status: row.docStatus ?? '' } }
  }
}

/** Classes whose only remedy before admin actions ship is a container restart (§2.4 "→ restart the container"). */
export const needsRestart = (cls: RunState) =>
  cls === 'reservation' || cls === 'busy_without_run' || cls === 'orphaned' || cls === 'unresponsive'

/** The reservation a row is about, for Release reservations (OQ-12); null for every other class. */
export const reservationOf = (row: StuckRow): ReservationFacts | null => (row.cls === 'reservation' ? (row.reservation ?? null) : null)
