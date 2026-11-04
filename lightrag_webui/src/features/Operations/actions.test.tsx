import { describe, expect, mock, test } from 'bun:test'
import { renderToString } from 'react-dom/server'
import type { ActionResponse, ActiveRun, OpsStatus } from './api'

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

// Every toast call is recorded, so a result's toast is checked as the user sees it.
const toasts: { level: string; text: string; opts?: { action?: { label: string; onClick: () => void } } }[] = []
const record = (level: string) => (text: string, opts?: { action?: { label: string; onClick: () => void } }) => void toasts.push({ level, text, opts })
mock.module('sonner', () => ({
  toast: Object.assign(record('message'), {
    success: record('success'),
    info: record('info'),
    warning: record('warning'),
    error: record('error'),
    message: record('message')
  }),
  Toaster: () => null
}))

const {
  EMPTY_FIELDS,
  actionOffer,
  actionOffers,
  confirmGate,
  dryRunCounts,
  inProgressOf,
  outcomeMessage,
  outcomeOf,
  outcomeOfError,
  reasonState,
  releaseFacts,
  stuckActionKinds,
  toastSpec,
  typedMatches
} = await import('./actions')
const { showOutcomeToast } = await import('./actionToast')
const { AlertDialog } = await import('@/components/ui/AlertDialog')
const { ConfirmActionPanel } = await import('./components/ConfirmActionDialog')
const { WorkspaceActionBar, DryRunPreview, HostedActionDialogs } = await import('./components/WorkspaceActions')
const { openHostedRun, useActionHost } = await import('./actionHost')
const { default: MaskedValue } = await import('./components/MaskedValue')
const { default: ActiveRunsTable } = await import('./components/ActiveRunsTable')
const { OpsShellContext } = await import('./context')

const status = (patch: Partial<OpsStatus> & { actions?: boolean; release?: boolean } = {}): OpsStatus => {
  const { actions = true, release, ...rest } = patch
  return {
    available: true,
    features: { durable: true, runs: true, actions, ...(release === undefined ? {} : { release_busy: release }) },
    process: { mode: 'gunicorn', expected_workers: 3, workers_live: 3 },
    operator: true,
    version: '1.5.1',
    generated_at: '2026-11-04T10:00:00Z',
    ...rest
  }
}

const withShell = (node: React.ReactNode, s: OpsStatus | null) =>
  renderToString(<OpsShellContext.Provider value={{ status: s, openInsights: () => {} }}>{node}</OpsShellContext.Provider>)

/** A stand-in for i18next's t(): the key plus its values, so a test can see both. */
const fakeT = ((key: string, values?: Record<string, unknown>) =>
  values && Object.keys(values).length ? `${key}${JSON.stringify(values)}` : key) as unknown as Parameters<typeof showOutcomeToast>[0]

const apiError = (status: number, detail: unknown, requestId: string | null = null) =>
  Object.assign(new Error(`${status}`), { status, detail, requestId })

// ---------- the confirm gate (§2.10) ----------

describe('confirmGate: reason, typed confirm, pending', () => {
  test('a reason must be 3 to 512 characters, trimmed', () => {
    expect(reasonState('')).toBe('empty')
    expect(reasonState('  ab  ')).toBe('short')
    expect(reasonState('abc')).toBe('ok')
    expect(reasonState('x'.repeat(512))).toBe('ok')
    expect(reasonState('x'.repeat(513))).toBe('long')
    expect(confirmGate({ reason: 'ab' }).canConfirm).toBe(false)
    expect(confirmGate({ reason: 'ticket 4411' }).canConfirm).toBe(true)
  })

  test('a typed confirm must match the workspace exactly (case and all)', () => {
    expect(typedMatches('fsktm', 'fsktm')).toBe(true)
    expect(typedMatches('  fsktm ', 'fsktm')).toBe(true)
    expect(typedMatches('FSKTM', 'fsktm')).toBe(false)
    expect(typedMatches('fskt', 'fsktm')).toBe(false)
    const base = { reason: 'worker gone', typedExpected: 'fsktm' }
    expect(confirmGate({ ...base, typed: '' })).toMatchObject({ canConfirm: false, typedOk: false })
    expect(confirmGate({ ...base, typed: 'fsktm' })).toMatchObject({ canConfirm: true, typedOk: true })
    // A good name never stands in for a reason.
    expect(confirmGate({ reason: '', typedExpected: 'fsktm', typed: 'fsktm' }).canConfirm).toBe(false)
  })

  test('nothing more is sent while pending, before the dialog is ready, or without permission', () => {
    expect(confirmGate({ reason: 'ok reason', pending: true }).canConfirm).toBe(false)
    expect(confirmGate({ reason: 'ok reason', ready: false }).canConfirm).toBe(false)
    expect(confirmGate({ reason: 'ok reason', allowed: false }).canConfirm).toBe(false)
  })
})

const panel = (patch: Partial<React.ComponentProps<typeof ConfirmActionPanel>> = {}) =>
  renderToString(
    <AlertDialog open>
      <ConfirmActionPanel
        title="Release busy in fsktm"
        body={<p>copy</p>}
        confirmLabel="Release busy"
        fields={EMPTY_FIELDS}
        onFields={() => {}}
        pending={false}
        refusal={null}
        onCancel={() => {}}
        onSubmit={() => {}}
        {...patch}
      />
    </AlertDialog>
  )

const submitButton = (html: string) => html.match(/<button type="submit"[^>]*>/)?.[0] ?? ''
/** The submit button carries the disabled attribute (not just the disabled: style variant). */
const isDisabled = (html: string) => / disabled=""/.test(submitButton(html))

describe('ConfirmActionPanel', () => {
  test('the confirm is disabled until the reason is long enough', () => {
    expect(isDisabled(panel({ fields: { reason: 'ab', typed: '' } }))).toBe(true)
    expect(isDisabled(panel({ fields: { reason: 'ticket 4411', typed: '' } }))).toBe(false)
    expect(panel({ fields: { reason: 'ab', typed: '' } })).toContain('operations.actions.reasonShort')
  })

  test('a typed confirm shows the name to type and gates the confirm', () => {
    const wrong = panel({ typedExpected: 'fsktm', fields: { reason: 'worker gone', typed: 'fsk' } })
    expect(wrong).toContain('operations.actions.typeToConfirm')
    expect(wrong).toContain('aria-invalid="true"')
    expect(isDisabled(wrong)).toBe(true)
    expect(isDisabled(panel({ typedExpected: 'fsktm', fields: { reason: 'worker gone', typed: 'fsktm' } }))).toBe(false)
  })

  test('without a typed confirm there is no name field', () => {
    expect(panel({ fields: { reason: 'ticket 4411', typed: '' } })).not.toContain('operations.actions.typeToConfirm')
  })

  test('pending: aria-busy, spinner and "Working…" text, inputs locked', () => {
    const html = panel({ pending: true, fields: { reason: 'ticket 4411', typed: '' } })
    expect(submitButton(html)).toContain('aria-busy="true"')
    expect(isDisabled(html)).toBe(true)
    expect(html).toContain('operations.actions.working')
    expect(html).toContain('animate-spin')
  })

  test('a refusal stays under the buttons as an announced alert with its audit id and Ref', () => {
    const refusal = outcomeOfError('release_busy', apiError(409, { code: 'in_progress_by', actor: 'khairil', action: 'run' }, 'req-1'))
    const html = panel({ refusal })
    expect(html).toContain('role="alert"')
    expect(html).toContain('operations.actions.inProgressByAction')
    expect(html).toContain('req-1')
  })

  test('an operator-only action says so and stays disabled for others', () => {
    const html = panel({ allowed: false, notAllowedText: 'Only operators can release busy.', fields: { reason: 'ticket 4411', typed: '' } })
    expect(html).toContain('Only operators can release busy.')
    expect(isDisabled(html)).toBe(true)
  })
})

// ---------- what is offered (§2.4, §2.9) ----------

describe('actionOffer', () => {
  const ctx = (patch: Partial<Parameters<typeof actionOffer>[1]> = {}) => ({ status: status(), busy: true, ownerSuspect: true, ...patch })

  test('nothing is offered while actions are off on the server (OPS_ACTIONS_ENABLED)', () => {
    expect(actionOffers(ctx({ status: status({ actions: false }) }))).toEqual([])
    expect(actionOffers(ctx({ status: null }))).toEqual([])
  })

  test('Release busy is not rendered in single-process mode', () => {
    const single = status({ process: { mode: 'uvicorn', expected_workers: 1, workers_live: 1 } })
    expect(actionOffer('release_busy', ctx({ status: single }))).toBeNull()
    expect(actionOffers(ctx({ status: single })).map((o) => o.kind)).toEqual(['cancel', 'run', 'drain_email'])
    // Nor when the server reports it off.
    expect(actionOffer('release_busy', ctx({ status: status({ release: false }) }))).toBeNull()
  })

  test('Release busy is disabled with a hint for non-operators and while the owner looks alive', () => {
    expect(actionOffer('release_busy', ctx({ status: status({ operator: false }) }))).toEqual({ kind: 'release_busy', disabled: true, hint: 'operatorOnly' })
    expect(actionOffer('release_busy', ctx({ ownerSuspect: false }))).toMatchObject({ disabled: true, hint: 'ownerAlive' })
    expect(actionOffer('release_busy', ctx({ busy: false }))).toMatchObject({ disabled: true, hint: 'notBusy' })
    expect(actionOffer('release_busy', ctx())).toMatchObject({ disabled: false, hint: null })
  })

  test('Run now is disabled while busy or destructive_busy; Cancel only when busy', () => {
    expect(actionOffer('run', ctx())).toMatchObject({ disabled: true, hint: 'busy' })
    expect(actionOffer('run', ctx({ busy: false, destructiveBusy: true }))).toMatchObject({ disabled: true, hint: 'destructive' })
    expect(actionOffer('run', ctx({ busy: false }))).toMatchObject({ disabled: false })
    expect(actionOffer('cancel', ctx({ busy: false }))).toMatchObject({ disabled: true, hint: 'notBusy' })
    expect(actionOffer('cancel', ctx({ cancellationRequested: true }))).toMatchObject({ disabled: true, hint: 'cancelling' })
    // Unknown live state: offered, the server decides.
    expect(actionOffer('cancel', ctx({ busy: null }))).toMatchObject({ disabled: false })
  })

  test('Drain is disabled only when the queue is known to be empty', () => {
    expect(actionOffer('drain_email', ctx({ emailPending: 0 }))).toMatchObject({ disabled: true, hint: 'noEmail' })
    expect(actionOffer('drain_email', ctx({ emailPending: 2 }))).toMatchObject({ disabled: false })
    expect(actionOffer('drain_email', ctx({ emailPending: null }))).toMatchObject({ disabled: false })
  })

  test('stuck rows: reservations release (OQ-12), stale docs Run now, held runs cancel or release', () => {
    expect(stuckActionKinds({ cls: 'reservation' })).toEqual(['release_reservations'])
    expect(stuckActionKinds({ cls: 'stale_doc' })).toEqual(['run'])
    expect(stuckActionKinds({ cls: 'orphaned' })).toEqual(['cancel', 'release_busy'])
  })
})

describe('operator gating in the views', () => {
  const bar = (s: OpsStatus | null, ctx = { busy: true, ownerSuspect: true }) => withShell(<WorkspaceActionBar workspace="fsktm" ctx={ctx} />, s)

  test('the action row renders nothing while actions are off', () => {
    expect(bar(status({ actions: false }))).toBe('')
  })

  test('multi-worker: Release busy is shown, disabled with its hint, for a non-operator', () => {
    const html = bar(status({ operator: false }))
    expect(html).toContain('operations.actions.button.release_busy')
    expect(html).toContain('operations.actions.hint.operatorOnly')
    expect(html).toMatch(/<button[^>]*disabled[^>]*>operations\.actions\.button\.release_busy/)
  })

  test('single process: no Release busy at all', () => {
    const html = bar(status({ process: { mode: 'uvicorn', expected_workers: 1, workers_live: 1 } }))
    expect(html).toContain('operations.actions.button.cancel')
    expect(html).not.toContain('release_busy')
  })

  test('Reveal is hidden for non-operators and when actions are off', () => {
    const field = (s: OpsStatus) =>
      withShell(
        <MaskedValue kind="doc_file_path" workspace="fsktm" id="doc-1">
          <span>R*****.pdf</span>
        </MaskedValue>,
        s
      )
    expect(field(status())).toContain('operations.reveal.button')
    expect(field(status({ operator: false }))).not.toContain('operations.reveal.button')
    expect(field(status({ actions: false }))).not.toContain('operations.reveal.button')
    // The masked value is what everyone sees before a reveal.
    expect(field(status({ operator: false }))).toContain('R*****.pdf')
  })

  test('the Active table gets a "⋯" menu per row only when actions are on', () => {
    const run: ActiveRun = { run_id: 'r1', workspace: 'fsktm', state: 'orphaned', started_at: '2026-11-04T09:14:00Z', worker_alive: false }
    expect(withShell(<ActiveRunsTable runs={[run]} now={Date.parse('2026-11-04T10:00:00Z')} />, status())).toContain('operations.actions.menuLabel')
    expect(withShell(<ActiveRunsTable runs={[run]} now={Date.parse('2026-11-04T10:00:00Z')} />, status({ actions: false }))).not.toContain(
      'operations.actions.menuLabel'
    )
    // The compact Overview table never carries actions.
    expect(withShell(<ActiveRunsTable runs={[run]} now={null} compact />, status())).not.toContain('operations.actions.menuLabel')
  })
})

// ---------- results and their toasts (§2.10: every result is a toast with its audit id) ----------

const resp = (r: ActionResponse) => r

describe('outcomes: one toast per result code', () => {
  const cases: [Parameters<typeof outcomeOf>[0], ActionResponse, string, string][] = [
    ['cancel', { result: 'cancellation_requested', audit_id: 57 }, 'success', 'operations.actions.result.cancellation_requested'],
    ['cancel', { result: 'not_busy', audit_id: 58 }, 'info', 'operations.actions.result.not_busy'],
    ['cancel', { result: 'lock_unavailable', audit_id: 59 }, 'warning', 'operations.refusal.lock_unavailable'],
    ['run', { result: 'started', audit_id: 60 }, 'success', 'operations.actions.result.started'],
    ['run', { result: 'nothing_to_process', audit_id: 61 }, 'info', 'operations.actions.result.nothing_to_process'],
    ['run', { result: 'request_pending', audit_id: 62 }, 'info', 'operations.actions.result.request_pending'],
    ['run', { result: 'refused_busy', audit_id: 63 }, 'warning', 'operations.refusal.refused_busy'],
    ['run', { result: 'refused_paused', audit_id: 64 }, 'warning', 'operations.refusal.refused_paused'],
    ['drain_email', { result: 'kicked', audit_id: 65 }, 'success', 'operations.actions.result.kicked'],
    ['drain_email', { result: 'started_local', audit_id: 66 }, 'success', 'operations.actions.result.started_local'],
    ['drain_email', { result: 'nothing_pending', audit_id: 67 }, 'info', 'operations.actions.result.nothing_pending'],
    ['drain_email', { result: 'queue_unavailable', audit_id: 68 }, 'warning', 'operations.refusal.queue_unavailable'],
    ['release_busy', { result: 'released', released: true, audit_id: 69 }, 'success', 'operations.actions.result.released'],
    ['release_busy', { result: 'owner_alive', released: false, reason: 'owner_alive', audit_id: 70 }, 'warning', 'operations.refusal.owner_alive'],
    ['release_busy', { result: 'owner_unknown', released: false, reason: 'owner_unknown', audit_id: 71 }, 'warning', 'operations.refusal.owner_unknown'],
    ['release_busy', { result: 'cas_mismatch', released: false, reason: 'cas_mismatch', audit_id: 72 }, 'warning', 'operations.refusal.cas_mismatch'],
    ['release_busy', { released: false, reason: 'lock_unavailable', audit_id: 73 }, 'warning', 'operations.refusal.lock_unavailable'],
    ['release_busy', { result: 'single_process', released: false, reason: 'single_process', audit_id: 74 }, 'warning', 'operations.refusal.single_process'],
    ['release_busy', { result: 'not_busy', released: false, reason: 'not_busy', audit_id: 75 }, 'info', 'operations.actions.result.notBusyRelease'],
    ['reveal', { audit_id: 76 }, 'success', 'operations.actions.result.revealed']
  ]
  for (const [action, r, level, key] of cases) {
    test(`${action}: ${r.result ?? r.reason ?? 'ok'} → ${level}`, () => {
      const spec = toastSpec(outcomeOf(action, resp(r)))
      expect(spec.level).toBe(level as never)
      expect(spec.key).toBe(key)
      expect(spec.auditId).toBe(String(r.audit_id))
    })
  }

  test('in_progress_by names the admin holding the lease, with no audit id (nothing was written)', () => {
    const o = outcomeOf('run', { result: 'in_progress_by', in_progress_by: { actor: 'khairil', action: 'run', since: '2026-11-04T12:03:00Z' }, audit_id: null })
    expect(o).toMatchObject({ tone: 'refused', code: 'in_progress_by', auditId: null })
    expect(outcomeMessage(o)).toEqual({ key: 'operations.actions.inProgressByAction', values: { actor: 'khairil', action: 'run' } })
    expect(toastSpec(o).level).toBe('warning')
    // Without a named holder, the generic refusal.
    expect(outcomeMessage(outcomeOf('cancel', { result: 'in_progress_by' })).key).toBe('operations.refusal.in_progress_by')
  })

  test('a release that kept a queued request offers Run now', () => {
    const o = outcomeOf('release_busy', { result: 'released', released: true, run_now_offered: true, audit_id: 80 })
    expect(o.offerRunNow).toBe(true)
    expect(toastSpec(o)).toMatchObject({ key: 'operations.actions.result.releasedRunPending', runNow: true })
    // The live state seen on open stands in when the server does not say.
    expect(outcomeOf('release_busy', { released: true }, { requestPending: true }).offerRunNow).toBe(true)
    expect(outcomeOf('release_busy', { released: true, run_now_offered: false }, { requestPending: true }).offerRunNow).toBe(false)
  })

  test('HTTP refusals: rate limit, operator, live admin, audit write failure', () => {
    expect(toastSpec(outcomeOfError('cancel', apiError(429, 'rate_limited')))).toMatchObject({ level: 'warning', key: 'operations.refusal.rate_limited' })
    expect(toastSpec(outcomeOfError('reveal', apiError(403, 'not_operator')))).toMatchObject({ level: 'warning', key: 'operations.refusal.not_operator' })
    expect(toastSpec(outcomeOfError('run', apiError(403, 'admin_not_current')))).toMatchObject({ key: 'operations.refusal.admin_not_current' })
    expect(toastSpec(outcomeOfError('run', apiError(403, 'impersonation_not_allowed')))).toMatchObject({ key: 'operations.refusal.impersonation_not_allowed' })
    expect(toastSpec(outcomeOfError('cancel', apiError(503, 'audit_unavailable')))).toMatchObject({ level: 'warning', key: 'operations.refusal.audit_unavailable' })
    expect(toastSpec(outcomeOfError('release_busy', apiError(400, 'confirm_mismatch')))).toMatchObject({ key: 'operations.refusal.confirm_mismatch' })
  })

  test('a reveal of something that changed since the page was drawn says so', () => {
    expect(toastSpec(outcomeOfError('reveal', apiError(404, 'not_found')))).toMatchObject({ level: 'error', key: 'operations.reveal.notFound' })
    expect(toastSpec({ ...outcomeOf('reveal', {}), tone: 'info', code: 'nothing_to_reveal' })).toMatchObject({ level: 'info', key: 'operations.actions.result.nothing_to_reveal' })
  })

  test('an unexpected server error is an error toast with its Ref', () => {
    const spec = toastSpec(outcomeOfError('run', apiError(500, 'internal_error', 'req-9')))
    expect(spec).toMatchObject({ level: 'error', key: 'operations.error.server', requestId: 'req-9' })
  })

  test('an unknown result code from a newer server still reads, as information', () => {
    expect(toastSpec(outcomeOf('run', { result: 'queued_elsewhere' }))).toMatchObject({ level: 'info', key: 'operations.actions.result.unknown', values: { code: 'queued_elsewhere' } })
  })

  test('the lease holder is read however it is spelled', () => {
    expect(inProgressOf({ in_progress_by: 'opso' })).toEqual({ actor: 'opso', action: null, since: null })
    expect(inProgressOf({ code: 'in_progress_by', actor: 'opso', action: 'cancel' })).toEqual({ actor: 'opso', action: 'cancel', since: null })
    expect(inProgressOf(null)).toBeNull()
  })

  test('showOutcomeToast: the sentence, the audit id, and Run now on the toast', () => {
    toasts.length = 0
    showOutcomeToast(fakeT, outcomeOf('cancel', { result: 'cancellation_requested', audit_id: 57 }))
    expect(toasts[0]).toMatchObject({ level: 'success' })
    expect(toasts[0].text).toBe('operations.actions.result.cancellation_requested · operations.actions.auditRef{"id":"57"}')
    let ran = false
    showOutcomeToast(fakeT, outcomeOf('release_busy', { released: true, run_now_offered: true, audit_id: 80 }), { onRunNow: () => void (ran = true) })
    expect(toasts[1].opts?.action?.label).toBe('operations.actions.runNowButton')
    toasts[1].opts?.action?.onClick()
    expect(ran).toBe(true)
    showOutcomeToast(fakeT, outcomeOfError('run', apiError(500, 'internal_error', 'req-9')))
    expect(toasts[2]).toMatchObject({ level: 'error' })
    expect(toasts[2].text).toContain('req-9')
  })
})

describe('Run now after a release: the shell hosts the dialog', () => {
  test('the toast Run now opens the hosted Run dialog even after the released row is gone', () => {
    useActionHost.getState().reset()
    toasts.length = 0
    // The released Stuck row unmounts once the table reloads; its toast must still work.
    showOutcomeToast(fakeT, outcomeOf('release_busy', { released: true, run_now_offered: true, audit_id: 81 }), {
      onRunNow: () => openHostedRun('fsktm')
    })
    expect(useActionHost.getState().hosted).toBeNull()
    toasts[0].opts?.action?.onClick()
    expect(useActionHost.getState().hosted).toEqual({ workspace: 'fsktm', kind: 'run' })
    // The host renders for that workspace (dialogs portal out, so nothing shows server-side).
    expect(() =>
      renderToString(
        <OpsShellContext.Provider value={{ status: status(), openInsights: () => {} }}>
          <HostedActionDialogs />
        </OpsShellContext.Provider>
      )
    ).not.toThrow()
    // Closing keeps the workspace, so the dialog can play its exit.
    useActionHost.getState().setKind(null)
    expect(useActionHost.getState().hosted).toEqual({ workspace: 'fsktm', kind: null })
    useActionHost.getState().reset()
    expect(useActionHost.getState().hosted).toBeNull()
  })
})

// ---------- run now: the dry run (§2.10) ----------

describe('dry run', () => {
  test('failed already includes the user-cancelled ones; in-flight docs are counted too', () => {
    expect(dryRunCounts({ pending: 310, failed: 4, cancelled: 2, in_flight: 3 })).toEqual({ pending: 310, failed: 4, cancelled: 2, inFlight: 3, total: 317 })
    expect(dryRunCounts({ pending: 0, failed: 0, cancelled: 0 })?.total).toBe(0)
    expect(dryRunCounts(null)).toBeNull()
  })

  test('the preview says what will run, and what a run now would meet', () => {
    const counts = dryRunCounts({ pending: 310, failed: 4, cancelled: 2 })!
    const html = renderToString(<DryRunPreview counts={counts} workspace="fsktm" live={{ busy: false, paused: true }} />)
    expect(html).toContain('operations.actions.run.preview')
    expect(html).toContain('operations.actions.run.meets.paused')
    expect(renderToString(<DryRunPreview counts={dryRunCounts({ pending: 0 })!} workspace="fsktm" />)).toContain('operations.actions.run.previewNothing')
  })
})

describe('releaseFacts: what the release dialog quotes and compares', () => {
  test('owner keys, heartbeat age and the run the OS confirmed gone', () => {
    const facts = releaseFacts(
      {
        workspace: 'fsktm',
        live: { busy: true, run_id: 'r1', job_start: '2026-11-04T09:14:00+00:00', owner_worker_id: 'w811', owner_pid: 811, heartbeat_at: '2026-11-04T09:51:00Z', request_pending: true },
        recent_runs: [{ run_id: 'r1', workspace: 'fsktm', status: 'orphaned', state: 'orphaned', worker_alive: false }]
      },
      Date.parse('2026-11-04T10:00:00Z')
    )
    expect(facts).toEqual({
      busy: true,
      runId: 'r1',
      jobStart: '2026-11-04T09:14:00+00:00',
      requestPending: true,
      worker: 'w811',
      pid: 811,
      beatAge: 540,
      processGone: true
    })
  })

  test('no live state: nothing to release', () => {
    expect(releaseFacts(null, 0)).toMatchObject({ busy: false, runId: null, jobStart: null })
  })
})
