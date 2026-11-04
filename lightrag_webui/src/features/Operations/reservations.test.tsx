import { describe, expect, test } from 'bun:test'
import { renderToString } from 'react-dom/server'
import type { OpsStatus, Stuck } from './api'

// The stores behind these components read localStorage as they load.
const memory = new Map<string, string>()
Object.defineProperty(globalThis, 'localStorage', {
  value: {
    getItem: (k: string) => memory.get(k) ?? null,
    setItem: (k: string, v: string) => void memory.set(k, v),
    removeItem: (k: string) => void memory.delete(k)
  },
  configurable: true
})

const { actionOffer, outcomeMessage, outcomeOf, outcomeOfError, releaseReservationsBody, releaseReservationsOffered, stuckActionKinds } = await import('./actions')
const { reservationOf, stuckRows, stuckWhy } = await import('./processing/stuck')
const { default: StuckTable } = await import('./components/StuckTable')
const { OpsShellContext } = await import('./context')

const W1 = 'a'.repeat(32)
const W2 = 'b'.repeat(32)

const status = (patch: { mode?: string; release?: boolean | undefined; operator?: boolean; actions?: boolean } = {}): OpsStatus => {
  const { mode = 'gunicorn', operator = true, actions = true } = patch
  // `release: undefined` stands for a server that does not report the feature.
  const release = 'release' in patch ? patch.release : true
  return {
    available: true,
    features: { durable: true, runs: true, actions, ...(release === undefined ? {} : { release_reservations: release }) },
    process: { mode, expected_workers: 3, workers_live: 2 },
    operator,
    version: null,
    generated_at: '2026-11-04T10:00:00Z'
  }
}

/** The reconcile's reservation rows (ops/reconcile.py): a dead scan owner, two dead workers' enqueue shares, a legacy hold. */
const STUCK: Stuck = {
  orphaned: [],
  hung: [],
  busy_without_run: [],
  reservations: [
    { workspace: 'fsktm', kind: 'scanning', scanning: true, reason_code: 'owner_dead', owner_worker_id: W1, since: '2026-11-04T09:00:00+00:00', scan_started_at: '2026-11-04T09:00:00+00:00', age_s: 3600 },
    {
      workspace: 'fsktm',
      kind: 'pending_enqueues',
      scanning: false,
      reason_code: 'owner_dead',
      owner_worker_id: W1,
      dead_owners: [
        { worker_id: W1, count: 2 },
        { worker_id: W2, count: 1 }
      ],
      pending_enqueues: 4,
      since: '2026-11-04T09:10:00+00:00',
      age_s: 3000
    },
    { workspace: 'fbe', kind: 'pending_enqueues', scanning: false, reason_code: 'held', owner_worker_id: null, pending_enqueues: 3, since: '2026-11-04T09:30:00+00:00', age_s: 1800 }
  ],
  stale_docs: { items: [], total: 0 },
  thresholds: {}
} as unknown as Stuck

describe('reservation rows (OQ-12)', () => {
  const rows = stuckRows(STUCK, true).filter((r) => r.cls === 'reservation')

  test('one row per dead worker; each carries what that worker holds on the workspace', () => {
    expect(rows.length).toBe(4)
    const [scan, share1, share2, legacy] = rows
    // The scan owner's row also knows its enqueue share, so one release sends both expected values.
    expect(reservationOf(scan)).toEqual({ workerId: W1, scanning: true, scanStartedAt: '2026-11-04T09:00:00+00:00', share: 2, reasonCode: 'owner_dead' })
    expect(reservationOf(share1)).toMatchObject({ workerId: W1, scanning: true, share: 2 })
    expect(share1.pendingEnqueues).toBe(2)
    expect(reservationOf(share2)).toEqual({ workerId: W2, scanning: false, scanStartedAt: null, share: 1, reasonCode: 'owner_dead' })
    expect(reservationOf(legacy)).toEqual({ workerId: null, scanning: false, scanStartedAt: null, share: 0, reasonCode: 'held' })
    expect(new Set(rows.map((r) => r.key)).size).toBe(4)
  })

  test('the why: an exited worker\'s enqueues read differently from a hold with no owner', () => {
    expect(stuckWhy(rows[2]).key).toBe('reservationEnqueueDead')
    expect(stuckWhy(rows[3]).key).toBe('reservationEnqueue')
    expect(stuckWhy(rows[0]).key).toBe('reservationScan_owner_dead')
  })

  test('single-process mode lists none (a process cannot outlive its own reservations)', () => {
    expect(stuckRows(STUCK, false).filter((r) => r.cls === 'reservation')).toEqual([])
    expect(reservationOf(stuckRows({ ...STUCK, hung: [{ workspace: 'x', run_id: 'r' }] } as Stuck, true).find((r) => r.cls === 'hung')!)).toBeNull()
  })
})

describe('offering Release reservations', () => {
  const facts = { workerId: W1, scanning: true, scanStartedAt: 'x', share: 2, reasonCode: 'owner_dead' }

  test('Gunicorn, actions on and the server reporting the feature', () => {
    expect(releaseReservationsOffered(status())).toBe(true)
    expect(releaseReservationsOffered(status({ mode: 'uvicorn' }))).toBe(false)
    expect(releaseReservationsOffered(status({ release: false }))).toBe(false)
    // A server one release behind does not have the route.
    expect(releaseReservationsOffered(status({ release: undefined }))).toBe(false)
    expect(releaseReservationsOffered(status({ actions: false }))).toBe(false)
  })

  test('operators only; an owner nobody can confirm dead is disabled with a hint', () => {
    expect(actionOffer('release_reservations', { status: status(), busy: null, reservation: facts })).toEqual({ kind: 'release_reservations', disabled: false, hint: null })
    expect(actionOffer('release_reservations', { status: status({ operator: false }), busy: null, reservation: facts })?.hint).toBe('operatorOnly')
    expect(actionOffer('release_reservations', { status: status(), busy: null, reservation: { ...facts, workerId: null, reasonCode: 'held' } })?.hint).toBe('ownerUnknown')
    expect(actionOffer('release_reservations', { status: status(), busy: null, reservation: { ...facts, reasonCode: 'owner_unknown' } })?.hint).toBe('ownerUnknown')
    expect(actionOffer('release_reservations', { status: status({ mode: 'uvicorn' }), busy: null, reservation: facts })).toBeNull()
    expect(actionOffer('release_reservations', { status: status(), busy: null })).toBeNull()
    expect(stuckActionKinds({ cls: 'reservation' })).toEqual(['release_reservations'])
  })

  test('the request names the worker and what the page saw it hold', () => {
    expect(releaseReservationsBody('dead worker after OOM', ' fsktm ', facts)).toEqual({
      reason: 'dead worker after OOM',
      confirm: 'fsktm',
      worker_id: W1,
      expected_scan_started_at: 'x',
      expected_pending: 2
    })
    // A worker that owns no scan sends no scan start.
    expect(releaseReservationsBody('r', 'ws', { ...facts, scanning: false, scanStartedAt: 'stale' })?.expected_scan_started_at).toBeNull()
    expect(releaseReservationsBody('r', 'ws', { ...facts, workerId: null })).toBeNull()
  })
})

describe('answers', () => {
  test('released, and each refusal in reservation terms', () => {
    const ok = outcomeOf('release_reservations', { released: true, released_scanning: true, released_pending: 2, audit_id: 90 })
    expect(ok).toMatchObject({ tone: 'success', code: 'reservations_released', auditId: '90' })
    expect(outcomeMessage(ok).key).toBe('operations.actions.result.reservations_released')
    const alive = outcomeOf('release_reservations', { released: false, reason: 'owner_alive', audit_id: 91 })
    expect(alive.tone).toBe('refused')
    expect(outcomeMessage(alive).key).toBe('operations.actions.result.reservationOwnerAlive')
    expect(outcomeMessage(outcomeOf('release_reservations', { released: false, reason: 'cas_mismatch' })).key).toBe('operations.actions.result.reservationChanged')
    expect(outcomeMessage(outcomeOf('release_reservations', { released: false, reason: 'owner_unknown' })).key).toBe('operations.refusal.owner_unknown')
    expect(outcomeMessage(outcomeOf('release_reservations', { released: false, reason: 'single_process' })).key).toBe('operations.refusal.single_process')
    expect(outcomeMessage(outcomeOf('release_reservations', { released: false, reason: 'lock_unavailable' })).key).toBe('operations.refusal.lock_unavailable')
    const gone = outcomeOf('release_reservations', { released: false, reason: 'not_held' })
    expect(gone.tone).toBe('info')
    expect(outcomeMessage(gone).key).toBe('operations.actions.result.not_held')
  })

  test('an error the server raised (not operator, rate limit) is a refusal with its cause', () => {
    const e = outcomeOfError('release_reservations', Object.assign(new Error('403'), { status: 403, detail: 'not_operator' }))
    expect(e.tone).toBe('refused')
  })
})

describe('the Stuck table', () => {
  const rows = stuckRows(STUCK, true).filter((r) => r.cls === 'reservation')
  const render = (s: OpsStatus) =>
    renderToString(
      <OpsShellContext.Provider value={{ status: s, openInsights: () => {} }}>
        <StuckTable rows={rows} />
      </OpsShellContext.Provider>
    )

  test('reservation rows get the action menu when the feature is on', () => {
    expect(render(status())).toContain('operations.actions.menuLabel')
    expect(render(status({ release: false }))).not.toContain('operations.actions.menuLabel')
  })
})
