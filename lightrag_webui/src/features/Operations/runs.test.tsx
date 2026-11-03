import { describe, expect, test } from 'bun:test'
import { renderToString } from 'react-dom/server'
import type { ActiveRun, OpsRun, Stuck } from './api'

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
const { RunSummary, RunFailures } = await import('./drawers/RunDrawer')
const { RunsTable } = await import('./processing/RunsView')
const { default: ActiveRunsTable } = await import('./components/ActiveRunsTable')
const { default: StuckTable } = await import('./components/StuckTable')
const { stuckRows } = await import('./processing/stuck')
const { OpsPrefsContext } = await import('./prefs')
const { DrainerText } = await import('./processing/EmailQueueView')

const withPrefs = (node: React.ReactNode, patch: { reduceMotion?: boolean; paused?: boolean } = {}) =>
  renderToString(
    <OpsPrefsContext.Provider value={{ paused: false, setPaused: () => {}, reduceMotion: false, setReduceMotion: () => {}, ...patch }}>{node}</OpsPrefsContext.Provider>
  )

const LOOP_CLASSES = ['ops-breathe', 'ops-flow', 'ops-ring-a', 'ops-ring-b', 'ops-pulse']
const hasLoop = (html: string) => LOOP_CLASSES.some((c) => html.includes(c))

const now = Date.parse('2026-11-03T04:04:00Z')

const hung: OpsRun = {
  run_id: '7f3a9c2e-1111',
  workspace: 'fsktm',
  kind: 'process',
  trigger: 'email',
  status: 'running',
  state: 'hung',
  worker_id: 'w-812',
  pid: 812,
  worker_alive: true,
  started_at: '2026-11-03T03:22:00Z',
  heartbeat_age_s: 6,
  progress_age_s: 1320,
  docs_total: 12,
  docs_done: 4,
  docs_failed: 1,
  stage: 'extract',
  label: 'Rep…[12 files]'
}

describe('Run drawer', () => {
  test('a hung run: state with its explanation, owner process, progress and what to do', () => {
    const html = withPrefs(<RunSummary run={hung} now={now} multiWorker={false} />)
    expect(html).toContain('operations.run.hung.label')
    expect(html).toContain('operations.run.hung.explain')
    expect(html).toContain('operations.runDrawer.pid')
    expect(html).toContain('operations.runDrawer.heartbeat')
    expect(html).toContain('role="progressbar"')
    expect(html).toContain('4<!-- -->/<!-- -->12')
    expect(html).toContain('operations.runDrawer.guide.hung')
    expect(html).toContain('42m')
  })

  test('an orphaned run says the owner is gone; the restart guidance shows only in multi-worker mode', () => {
    const orphaned = { ...hung, status: 'orphaned', state: 'orphaned', worker_alive: false }
    expect(withPrefs(<RunSummary run={orphaned} now={now} multiWorker />)).toContain('operations.runDrawer.guide.orphaned')
    expect(withPrefs(<RunSummary run={orphaned} now={now} multiWorker />)).toContain('operations.runDrawer.ownerGone')
    expect(withPrefs(<RunSummary run={orphaned} now={now} multiWorker={false} />)).not.toContain('operations.runDrawer.guide.orphaned')
  })

  test('a finished run shows its outcome and duration, and flags a run that was orphaned', () => {
    const done = { ...hung, status: 'succeeded', state: 'succeeded', finished_at: '2026-11-03T03:24:30Z', was_orphaned: true }
    const html = withPrefs(<RunSummary run={done} now={now} multiWorker />)
    expect(html).toContain('operations.run.succeeded.label')
    expect(html).toContain('operations.runDrawer.took')
    expect(html).toContain('operations.runDrawer.wasOrphaned')
    expect(html).not.toContain('operations.runDrawer.whatToDo')
  })

  test('the running dot breathes in the drawer header, and stops under reduced motion or pause', () => {
    const running = { ...hung, state: 'running' }
    expect(withPrefs(<RunSummary run={running} now={now} multiWorker={false} />)).toContain('ops-breathe')
    expect(hasLoop(withPrefs(<RunSummary run={running} now={now} multiWorker={false} />, { reduceMotion: true }))).toBe(false)
    expect(hasLoop(withPrefs(<RunSummary run={running} now={now} multiWorker={false} />, { paused: true }))).toBe(false)
  })

  test('failed documents: a summary by kind, the masked name, stage, kind with status, model and attempt', () => {
    const html = withPrefs(
      <RunFailures
        workspace="fsktm"
        failures={[
          { ts: '2026-11-03T03:44:58Z', doc_id: 'doc-9a2f', file_path_masked: 'R*****_2024.pdf', stage: 'extract', error_kind: 'provider_budget', http_status: 402, model: 'gpt-x', attempt: 3, message: 'Payment required' },
          { ts: '2026-11-03T03:40:00Z', doc_id: 'doc-77c0', error_kind: 'parser_timeout' }
        ]}
      />
    )
    expect(html).toContain('R*****_2024.pdf')
    expect(html).toContain('gpt-x')
    expect(html).toContain('operations.document.attempt')
    expect(html).toContain('Payment required')
    // Untranslated in tests the kind reads as its id (the i18n default), so the HTTP status is appended.
    expect(html).toContain('provider_budget (402)')
    expect(html).toContain('parser_timeout')
    expect(withPrefs(<RunFailures workspace="w" failures={[]} />)).toContain('operations.runDrawer.noFailures')
  })
})

describe('Active runs from run tracking', () => {
  const runs: ActiveRun[] = [
    { run_id: 'r1', workspace: 'space1', kind: 'email_batch', state: 'running', started_at: '2026-11-03T04:01:00Z', docs_done: 6, docs_total: 10, stage: 'extract', label: 'Email batch [10]' },
    { run_id: 'r2', workspace: 'hr', kind: 'process', state: 'orphaned', worker_alive: false, pid: 811, docs_done: 4, docs_total: 12, docs_failed: 2 },
    { run_id: null, workspace: 'interim', kind: 'process', state: 'stale_doc' }
  ]

  test('rows link to their Run drawer (interim rows to the workspace), with stage and failed counts', () => {
    const html = withPrefs(<ActiveRunsTable runs={runs} now={now} multiWorker />)
    expect(html).toContain('orun=r1')
    expect(html).toContain('ows=space1')
    expect(html).not.toContain('orun=null')
    expect(html).toContain('>extract</span>')
    expect(html).toContain('operations.active.failedDocs')
    expect(html).toContain('operations.active.ownerGone')
    expect(html).toContain('operations.run.orphaned.label')
  })

  test('under reduced motion: no loops, final values', () => {
    const html = withPrefs(<ActiveRunsTable runs={runs} now={now} multiWorker />, { reduceMotion: true })
    expect(hasLoop(html)).toBe(false)
    expect(html).toContain('6<!-- -->/<!-- -->10')
  })
})

describe('Run history table', () => {
  const rows: OpsRun[] = [
    { ...hung, run_id: 'a1', status: 'running', state: 'running' },
    { ...hung, run_id: 'b2', status: 'failed', state: 'failed', finished_at: '2026-11-03T03:30:00Z', error_kind: 'storage_error' }
  ]

  test('each row is a real link to its run; outcomes carry icon and text; history dots never breathe', () => {
    const html = withPrefs(<RunsTable runs={rows} now={now} />)
    expect(html).toContain('orun=a1')
    expect(html).toContain('orun=b2')
    expect(html).toContain('operations.run.failed.label')
    expect(html).toContain('>storage_error</td>')
    expect(html).toContain('8m')
    expect(hasLoop(html)).toBe(false)
  })
})

describe('Stuck table with the five classes', () => {
  const data: Stuck = {
    orphaned: [{ workspace: 'fsktm', run_id: 'r1', pid: 811, heartbeat_age_s: 540, age_s: 600, confirmed_by: 'pid_gone' }],
    hung: [{ workspace: 'hr', run_id: 'r2', pid: 812, progress_age_s: 1320, age_s: 1320 }],
    busy_without_run: [{ workspace: 'law' }],
    reservations: [{ workspace: 'space1', scanning: true, age_s: 840 }],
    stale_docs: { items: [{ workspace: 'library', doc_id: 'doc-77c0', status: 'processing', file_path_masked: 'R*****.pdf', age_s: 4320 }], total: 1 },
    thresholds: {}
  }

  test('multi-worker: every class, runs link to the Run drawer, reasons built from the row', () => {
    const html = renderToString(<StuckTable rows={stuckRows(data, true)} />)
    for (const cls of ['orphaned', 'hung', 'busy_without_run', 'reservation', 'stale_doc']) expect(html).toContain(`operations.run.${cls}.label`)
    expect(html).toContain('orun=r1')
    expect(html).toContain('orun=r2')
    expect(html).toContain('ows=space1')
    expect(html).toContain('operations.stuck.why.orphaned')
    expect(html).toContain('operations.stuck.why.hung')
    expect(html).toContain('operations.stuck.why.reservationScan')
    expect(html).toContain('operations.stuck.flagScanning')
  })

  test('restart guidance on reservations, busy without a run and orphaned runs, not on hung runs or stale docs', () => {
    const html = renderToString(<StuckTable rows={stuckRows(data, true)} />)
    expect((html.match(/operations\.stuck\.restart/g) ?? []).length).toBe(3)
  })

  test('single-process mode never shows orphaned runs or reservations', () => {
    const html = renderToString(<StuckTable rows={stuckRows(data, false)} />)
    expect(html).not.toContain('operations.run.orphaned.label')
    expect(html).not.toContain('operations.run.reservation.label')
    expect(html).toContain('operations.run.hung.label')
  })

  test('server reason codes are translated, never printed as-is', () => {
    const rows = stuckRows(
      {
        ...data,
        busy_without_run: [{ workspace: 'law', reason_code: 'run_finished', run_id: 'r9' }],
        reservations: [{ workspace: 'hr_office', scanning: true, reason_code: 'owner_dead' }]
      },
      true
    )
    const html = renderToString(<StuckTable rows={rows} />)
    expect(html).toContain('operations.stuck.why.busy_run_finished')
    expect(html).toContain('operations.stuck.why.reservationScan_owner_dead')
    expect(html).not.toContain('>run_finished')
    expect(html).not.toContain('>owner_dead')
  })
})

describe('Email queue drainer and backoff (FR-32)', () => {
  test('a live drainer in backoff shows the time left, measured against the confirmed time', () => {
    const html = renderToString(
      <DrainerText item={{ pending: 2, drainer: { pid: 812, state: 'backoff', backoff_until: '2026-11-03T04:04:40Z' } }} now={now} />
    )
    expect(html).toContain('operations.email.worker')
    expect(html).toContain('operations.email.backoffLeft')
    expect(html).not.toContain('operations.workerDrawer.drainer.backoff')
  })

  test('without a live drainer, the recorded backoff deadline; a past deadline or nothing reads "–"', () => {
    expect(renderToString(<DrainerText item={{ pending: 2, backoff_until: '2026-11-03T04:05:00Z' }} now={now} />)).toContain('operations.email.backoffLeft')
    expect(renderToString(<DrainerText item={{ pending: 2, backoff_until: '2026-11-03T04:00:00Z' }} now={now} />)).toBe('–')
    expect(renderToString(<DrainerText item={{ pending: 2 }} now={now} />)).toBe('–')
  })

  test('a draining worker says so', () => {
    const html = renderToString(<DrainerText item={{ pending: 29, drainer: { worker_id: 'w-813-aaaa-bbbb', state: 'draining' } }} now={now} />)
    expect(html).toContain('operations.email.worker · draining')
  })
})
