import { describe, expect, mock, test } from 'bun:test'
import { readFileSync } from 'node:fs'
import { renderToString } from 'react-dom/server'
import type { AlertRule, AlertRules, AlertsPage, LogCaptureState, OpsAlert, OpsOverview, OpsStatus } from './api'

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
const toasts: { level: string; text: string }[] = []
const record = (level: string) => (text: string) => void toasts.push({ level, text })
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

const alertsModel = await import('./activity/alerts')
const {
  MUTE_MAX_MINUTES,
  MUTE_PRESETS,
  CAPTURE_MAX_MINUTES,
  alertChannelOf,
  alertTarget,
  bannerEnters,
  captureRemaining,
  capturePresets,
  changeLines,
  draftErrors,
  draftOf,
  durationKey,
  emailStatus,
  emailedSeverity,
  fmtCountdown,
  fromLocalInput,
  incidentDefaults,
  incidentProblem,
  maintenanceActive,
  muteOf,
  normalizeAlert,
  normalizeAlertDetail,
  normalizeAlertRules,
  normalizeAlertsPage,
  normalizeLogCapture,
  rulesChange,
  sortOpenAlerts,
  THRESHOLD_PAIRS,
  thresholdError,
  toLocalInput,
  validMinutes,
  validModuleName
} = alertsModel
const { outcomeMessage, outcomeOf, outcomeOfError, toastSpec } = await import('./actions')
const { showOutcomeToast } = await import('./actionToast')
const { alertActionsEnabled, safeHttpUrl, sendAlertAction } = await import('./activity/alertActions')
const { AlertListBody, AlertTable, EmailModeNote, MutesList, RulesEditor } = await import('./activity/AlertsView')
const { default: AlertsView } = await import('./activity/AlertsView')
const { AlertEmailBanner, MaintenanceBanner, default: OpsBanners } = await import('./components/OpsBanners')
const { CaptureCountdown, default: LogCaptureControl } = await import('./components/LogCaptureControl')
const { MuteMenu, AckButton } = await import('./components/AlertActions')
const { AlertSummary } = await import('./drawers/AlertDrawer')
const { default: AttentionList } = await import('./components/AttentionList')
const { OpsOverviewContext, OpsShellContext } = await import('./context')
const { OpsPrefsContext } = await import('./prefs')
const { DEFAULT_OPS_HASH, readOpsHash } = await import('./hashState')
const { normalizeOverview } = await import('./normalize')
const { opsApi } = await import('./api')
const { axiosInstance } = await import('@/api/lightrag')

const NOW = Date.parse('2026-11-04T12:00:00Z')

const status = (patch: Partial<OpsStatus> & { alerts?: boolean } = {}): OpsStatus => {
  const { alerts = true, ...rest } = patch
  return {
    available: true,
    features: { durable: true, alerts, logs: true },
    process: { mode: 'uvicorn', expected_workers: 1, workers_live: 1 },
    operator: true,
    version: null,
    generated_at: '2026-11-04T12:00:00Z',
    ...rest
  }
}

const prefs = (reduceMotion = false) => ({ paused: false, setPaused: () => {}, reduceMotion, setReduceMotion: () => {} })

const render = (node: React.ReactNode, s: OpsStatus | null = status(), overview: unknown = null, reduceMotion = false) =>
  renderToString(
    <OpsPrefsContext.Provider value={prefs(reduceMotion)}>
      <OpsShellContext.Provider value={{ status: s, openInsights: () => {} }}>
        <OpsOverviewContext.Provider value={overview as never}>{node}</OpsOverviewContext.Provider>
      </OpsShellContext.Provider>
    </OpsPrefsContext.Provider>
  )

const alert = (patch: Partial<OpsAlert> = {}): OpsAlert => ({
  id: '12',
  rule: 'dependency_down',
  fingerprint: 'dependency_down:milvus',
  severity: 'critical',
  state: 'firing',
  title: null,
  label: null,
  scope: { dep: 'milvus' },
  target: null,
  runbook: null,
  runbook_url: null,
  first_seen: '2026-11-04T11:01:00Z',
  last_seen: '2026-11-04T11:59:00Z',
  resolved_at: null,
  acked_by: null,
  acked_at: null,
  ack_note: null,
  last_emailed_at: null,
  email: null,
  muted_until: null,
  ...patch
})

const rule = (patch: Partial<AlertRule> = {}): AlertRule => ({
  rule: 'stale_docs',
  label: null,
  severities: ['warning', 'serious'],
  email: true,
  condition: null,
  runbook: null,
  runbook_url: null,
  thresholds: [
    { key: 'stale_doc_minutes', value: 60, default: 60, min: 5, max: 1440, unit: 'min', integer: true },
    { key: 'stale_doc_serious_minutes', value: 180, default: 180, min: 10, max: 2880, unit: 'min', integer: true }
  ],
  direction: 'above',
  muted_until: null,
  mute_reason: null,
  muted_by: null,
  ...patch
})

// ---------- read models ----------

describe('alert rows (§3.5 LIGHTRAG_OPS_ALERTS)', () => {
  test('numeric ids become text; a row with resolved_at is resolved whatever its state says; rows without a rule are dropped', () => {
    const a = normalizeAlert({ id: 7, rule: 'run_hung', severity: 'serious', state: 'firing', resolved_at: '2026-11-04T11:30:00Z', scope: { workspace: 'fsktm', count: 2 } })!
    expect(a).toMatchObject({ id: '7', rule: 'run_hung', state: 'resolved', scope: { workspace: 'fsktm', count: 2 } })
    expect(normalizeAlert({ id: 8 })).toBeNull()
    expect(normalizeAlert({ rule: 'x' })).toBeNull()
  })

  test('the link becomes a drill target; a link to the alert itself adds none', () => {
    expect(normalizeAlert({ id: 1, rule: 'run_hung', link: '#/?ops=1&otab=processing&osub=stuck&orun=7f3a' })!.target).toMatchObject({ tab: 'processing', sub: 'stuck', run_id: '7f3a' })
    expect(normalizeAlert({ id: 1, rule: 'run_hung', link: '#/?ops=1&oal=1' })!.target).toBeNull()
    expect(normalizeAlert({ id: 1, rule: 'run_hung', link: 'javascript:alert(1)' })!.target).toBeNull()
  })

  test('a page keeps its keyset cursor, the evaluator switch and the email state', () => {
    const page = normalizeAlertsPage({ items: [{ id: 1, rule: 'x' }, { nope: true }], next_before: 99, enabled: true, email: { enabled: false, recipients: 2 } })
    expect(page.items).toHaveLength(1)
    expect(page.next_before).toBe('99')
    expect(page.email).toMatchObject({ enabled: false, recipients: 2 })
    expect(normalizeAlertsPage(null)).toMatchObject({ items: [], next_before: null, email: null })
  })

  test('open alerts list worst first, the server order breaking ties', () => {
    const sorted = sortOpenAlerts([alert({ id: 'a', severity: 'warning' }), alert({ id: 'b', severity: 'critical' }), alert({ id: 'c', severity: 'warning' })])
    expect(sorted.map((a) => a.id)).toEqual(['b', 'a', 'c'])
  })

  test('an alert drills into its scope on the rule view', () => {
    expect(alertTarget(alert())).toMatchObject({ tab: 'health', dep: 'milvus' })
    expect(alertTarget(alert({ rule: 'run_hung', scope: { workspace: 'fsktm', run_id: 'r1' } }))).toMatchObject({ tab: 'processing', sub: 'stuck', run_id: 'r1', workspace: 'fsktm' })
  })

  test('the drawer answer: {alert, rule, history} or a bare row; history never repeats the row itself', () => {
    const d = normalizeAlertDetail({ alert: { id: 3, rule: 'smtp_probe' }, history: [{ id: 3, rule: 'smtp_probe' }, { id: 1, rule: 'smtp_probe', resolved_at: '2026-11-01T00:00:00Z' }], rule: { rule: 'smtp_probe' } })
    expect(d.alert?.id).toBe('3')
    expect(d.history.map((h) => h.id)).toEqual(['1'])
    expect(d.rule?.rule).toBe('smtp_probe')
    expect(normalizeAlertDetail({ id: 4, rule: 'disk_free' }).alert?.id).toBe('4')
    expect(normalizeAlertDetail({}).alert).toBeNull()
  })
})

describe('rules, mutes and maintenance (GET /alerts/rules)', () => {
  test('thresholds as a map with defaults and bounds beside them, or as a list', () => {
    const r = normalizeAlertRules({
      rules: [
        { rule: 'stale_docs', severities: ['serious', 'warning'], thresholds: { stale_doc_minutes: 90 }, defaults: { stale_doc_minutes: 60 }, bounds: { stale_doc_minutes: [5, 1440] } },
        { rule: 'disk_free', thresholds: [{ key: 'disk_free_pct', value: 15, min: 1, max: 50, integer: true }] }
      ]
    })
    expect(r.rules[0].thresholds[0]).toEqual({ key: 'stale_doc_minutes', value: 90, default: 60, min: 5, max: 1440, unit: 'min', integer: false })
    expect(r.rules[0].severities).toEqual(['warning', 'serious'])
    expect(r.rules[0].email).toBe(true)
    // Lower is worse for disk free, whether or not the server says so.
    expect(r.rules[1]).toMatchObject({ direction: 'below', thresholds: [{ key: 'disk_free_pct', unit: '%', integer: true }] })
  })

  test('provider_paused is never emailed, warnings-only rules neither (§1.7)', () => {
    const r = normalizeAlertRules({ rules: [{ rule: 'provider_paused', severity: 'critical' }, { rule: 'email_queue_bad', severity: 'warning' }] })
    expect(r.rules.map((x) => x.email)).toEqual([false, false])
    expect(emailedSeverity('critical', 'provider_paused')).toBe(false)
    expect(emailedSeverity('serious', 'run_hung')).toBe(true)
    expect(emailedSeverity('warning', 'run_hung')).toBe(false)
  })

  test('mutes come as a list or a map, and a rule carrying its own mute adds it', () => {
    const fromMap = normalizeAlertRules({ mutes: { run_hung: { until: '2026-11-04T13:00:00Z', by: 'opso', reason: 'deploy' } } })
    expect(fromMap.mutes).toEqual([{ rule: 'run_hung', until: '2026-11-04T13:00:00Z', by: 'opso', reason: 'deploy', at: null }])
    const fromRule = normalizeAlertRules({ rules: [{ rule: 'stale_docs', mute: { until: '2026-11-04T13:00:00Z' } }] })
    expect(fromRule.mutes.map((m) => m.rule)).toEqual(['stale_docs'])
    expect(muteOf('stale_docs', fromRule.mutes, NOW)?.until).toBe('2026-11-04T13:00:00Z')
    expect(muteOf('stale_docs', fromRule.mutes, Date.parse('2026-11-04T14:00:00Z'))).toBeNull()
  })

  test('maintenance is on only while its end is ahead', () => {
    const r = normalizeAlertRules({ maintenance: { until: '2026-11-04T13:00:00Z', reason: 'deploy', by: 'opso' } })
    expect(maintenanceActive(r.maintenance, NOW)).toBe(true)
    expect(maintenanceActive(r.maintenance, Date.parse('2026-11-04T13:00:01Z'))).toBe(false)
    expect(normalizeAlertRules({ maintenance: {} }).maintenance).toBeNull()
  })

  test('the overview maintenance and alert channel are read the same way', () => {
    const o = normalizeOverview({ state: 'ok', kpis: {}, attention: [{ id: 'a', rule: 'run_hung', alert_id: 41 }], maintenance: { until: '2026-11-04T13:00:00Z' }, alert_channel: { ok: false, reason: 'smtp_probe_failing', failures: 3 } })
    expect(o.maintenance?.until).toBe('2026-11-04T13:00:00Z')
    expect(o.attention[0].alert_id).toBe('41')
    expect(alertChannelOf(o)).toMatchObject({ ok: false, reason: 'smtp_probe_failing', failures: 3, email_enabled: null })
    expect(alertChannelOf({ alert_channel: null })).toBeNull()
  })
})

describe('the Notified cell (FR-62)', () => {
  const on = { enabled: true, recipients: 2, ok: true, reason: null, since: null, failures: null, last_sent_at: null }
  test('sent once the claim stamped last_emailed_at', () => {
    expect(emailStatus(alert({ last_emailed_at: '2026-11-04T11:02:00Z' }), on)).toBe('sent')
  })
  test('warnings and provider pauses stay in the app; record-only while OPS_ALERT_EMAIL is off; else pending', () => {
    expect(emailStatus(alert({ severity: 'warning' }), on)).toBe('in_app_only')
    expect(emailStatus(alert({ rule: 'provider_paused' }), on)).toBe('in_app_only')
    expect(emailStatus(alert(), { ...on, enabled: false })).toBe('record_only')
    expect(emailStatus(alert(), on)).toBe('pending')
  })
  test('the word the server gives wins (muted, maintenance, suppressed after a deploy)', () => {
    expect(emailStatus(alert({ email: 'maintenance' }), on)).toBe('maintenance')
    expect(emailStatus(alert({ email: 'suppressed', last_emailed_at: '2026-11-04T11:02:00Z' }), on)).toBe('suppressed')
  })
})

// ---------- durations and validation ----------

describe('mute, maintenance and capture durations', () => {
  test('mutes are 1, 4 or 24 h, or 7 days at most (FR-61)', () => {
    expect([...MUTE_PRESETS]).toEqual([60, 240, 1440, 10080])
    expect(Math.max(...MUTE_PRESETS)).toBe(MUTE_MAX_MINUTES)
    expect(validMinutes(10080, MUTE_MAX_MINUTES)).toBe(true)
    expect(validMinutes(10081, MUTE_MAX_MINUTES)).toBe(false)
    expect(validMinutes(0, MUTE_MAX_MINUTES)).toBe(false)
    expect(validMinutes(1.5, MUTE_MAX_MINUTES)).toBe(false)
  })

  test('durations read in the largest whole unit', () => {
    expect(durationKey(30)).toEqual({ key: 'operations.alerts.duration.minutes', count: 30 })
    expect(durationKey(240)).toEqual({ key: 'operations.alerts.duration.hours', count: 4 })
    expect(durationKey(10080)).toEqual({ key: 'operations.alerts.duration.days', count: 7 })
  })

  test('an INFO capture lasts at most 60 minutes, whatever the server says (FR-26)', () => {
    expect(CAPTURE_MAX_MINUTES).toBe(60)
    expect(capturePresets(60)).toEqual([5, 15, 30, 60])
    expect(capturePresets(20)).toEqual([5, 15, 20])
    expect(capturePresets(600)).toEqual([5, 15, 30, 60])
    expect(normalizeLogCapture({ max_minutes: 600 }).max_minutes).toBe(60)
    expect(validModuleName('lightrag.pipeline')).toBe(true)
    expect(validModuleName('pipeline; drop')).toBe(false)
    expect(validModuleName('')).toBe(false)
  })
})

describe('rules editor validation', () => {
  const rules = [rule()]
  test('each value: present, a number, above zero, whole where it must be, inside its bounds', () => {
    const t = rules[0].thresholds[0]
    expect(thresholdError(t, '')).toBe('required')
    expect(thresholdError(t, 'abc')).toBe('number')
    expect(thresholdError(t, '0')).toBe('positive')
    expect(thresholdError(t, '-5')).toBe('positive')
    expect(thresholdError(t, '7.5')).toBe('integer')
    expect(thresholdError(t, '4')).toBe('min')
    expect(thresholdError(t, '1441')).toBe('max')
    expect(thresholdError(t, '90')).toBeNull()
    expect(thresholdError({ ...t, integer: false, min: null, max: null }, '0.05')).toBeNull()
  })

  test('the pairs the server checks are the pairs the editor checks', () => {
    expect(THRESHOLD_PAIRS).toContainEqual(['stale_doc_minutes', 'stale_doc_serious_minutes'])
    expect(THRESHOLD_PAIRS).toContainEqual(['http_5xx_warn', 'http_5xx_serious'])
    expect(THRESHOLD_PAIRS).toContainEqual(['failure_spike_min', 'failure_spike_serious'])
  })

  test('a serious level may not come before its warning level (equal is allowed, as on the server)', () => {
    const draft = draftOf(rules)
    expect(draftErrors(rules, draft)).toEqual({})
    expect(draftErrors(rules, { stale_docs: { stale_doc_minutes: '200', stale_doc_serious_minutes: '180' } })).toEqual({ stale_docs: { stale_doc_serious_minutes: 'order' } })
    expect(draftErrors(rules, { stale_docs: { stale_doc_minutes: '180', stale_doc_serious_minutes: '180' } })).toEqual({})
    // Where lower is worse, the serious level may not be above the warning one.
    const below = [rule({ direction: 'below' })]
    expect(draftErrors(below, { stale_docs: { stale_doc_minutes: '180', stale_doc_serious_minutes: '60' } })).toEqual({})
    expect(draftErrors(below, draftOf(below))).toEqual({ stale_docs: { stale_doc_serious_minutes: 'order' } })
    // A value with its own error is not compared.
    expect(draftErrors(rules, { stale_docs: { stale_doc_minutes: 'x', stale_doc_serious_minutes: '10' } })).toEqual({ stale_docs: { stale_doc_minutes: 'number' } })
  })

  test('only changed values are sent, as numbers; nothing changed sends nothing', () => {
    expect(rulesChange(rules, draftOf(rules))).toBeNull()
    const change = rulesChange(rules, { stale_docs: { stale_doc_minutes: ' 90 ', stale_doc_serious_minutes: '180' } })
    expect(change).toEqual({ stale_docs: { stale_doc_minutes: 90 } })
    expect(changeLines(rules, change!)).toEqual([{ rule: 'stale_docs', key: 'stale_doc_minutes', from: 60, to: 90 }])
  })

  const data = (patch: Partial<AlertRules> = {}): AlertRules => ({ rules, mutes: [], maintenance: null, email: null, ...patch })

  test('operators get inputs with the current values and a Save… that waits for a change', () => {
    const html = render(<RulesEditor data={data()} operator canAct now={NOW} />)
    expect(html).toContain('id="ops-th-stale_docs-stale_doc_minutes"')
    expect(html).toContain('value="60"')
    expect(html).toContain('operations.alerts.rules.save')
    expect(html).toMatch(/<button[^>]*disabled=""[^>]*>operations\.alerts\.rules\.save<\/button>/)
    expect(html).not.toContain('operations.alerts.rules.operatorOnly')
  })

  test('other admins read the same table without inputs or Save', () => {
    const html = render(<RulesEditor data={data()} operator={false} canAct now={NOW} />, status({ operator: false }))
    expect(html).toContain('operations.alerts.rules.operatorOnly')
    expect(html).not.toContain('<input')
    expect(html).not.toContain('operations.alerts.rules.save')
    expect(html).toContain('operations.alerts.rules.emailed')
  })
})

// ---------- outcomes: ack, mute, maintenance, capture ----------

describe('alert action outcomes', () => {
  test('{ok, audit_id} reads as the action done, with the audit id on the toast', () => {
    const cases = [
      ['ack', 'acked'],
      ['mute', 'muted'],
      ['unmute', 'unmuted'],
      ['maintenance', 'maintenance_started'],
      ['maintenance_end', 'maintenance_ended'],
      ['capture', 'capture_started'],
      ['capture_stop', 'capture_stopped'],
      ['rules', 'rules_saved']
    ] as const
    for (const [action, code] of cases) {
      const o = outcomeOf(action, { ok: true, audit_id: 57 } as never)
      expect(o).toMatchObject({ code, tone: 'success', auditId: '57' })
      expect(outcomeMessage(o).key).toBe(`operations.actions.result.${code}`)
      expect(toastSpec(o).level).toBe('success')
    }
  })

  test('server result words are read in the terms of each action; a no-op is information', () => {
    expect(outcomeOf('ack', { ok: true, result: 'already_acked', audit_id: 3 } as never)).toMatchObject({ code: 'already_acked', tone: 'info' })
    expect(outcomeOf('ack', { ok: true, result: 'resolved' } as never)).toMatchObject({ code: 'alert_resolved', tone: 'info' })
    expect(outcomeOf('maintenance', { ok: true, result: 'started' } as never).code).toBe('maintenance_started')
    expect(outcomeOf('capture', { ok: true, result: 'started' } as never).code).toBe('capture_started')
    expect(outcomeOf('capture_stop', { ok: true, result: 'not_active' } as never)).toMatchObject({ code: 'no_capture', tone: 'info' })
    expect(outcomeOf('maintenance_end', { ok: true, result: 'not_active' } as never)).toMatchObject({ code: 'not_in_maintenance', tone: 'info' })
    expect(outcomeOf('rules', { ok: true, result: 'unchanged' } as never)).toMatchObject({ code: 'no_change', tone: 'info' })
    expect(outcomeOf('rules', { ok: true, result: 'updated' } as never)).toMatchObject({ code: 'rules_saved', tone: 'success' })
    expect(outcomeOf('ack', { ok: false } as never).code).toBe('unknown')
  })

  test('a refusal raised as an HTTP error stays a refusal with its cause', () => {
    const err = Object.assign(new Error('x'), { status: 400, detail: 'threshold_out_of_bounds', requestId: 'req-1' })
    const o = outcomeOfError('rules', err)
    expect(o).toMatchObject({ tone: 'refused', code: 'threshold_out_of_bounds', requestId: 'req-1', error: false })
    expect(outcomeMessage(o).key).toBe('operations.refusal.threshold_out_of_bounds')
    // An ack of an alert that is gone says so, not "older server".
    expect(outcomeMessage(outcomeOfError('ack', Object.assign(new Error('x'), { status: 404, detail: 'not_found' }))).key).toBe('operations.alertDrawer.notFound')
    expect(outcomeMessage(outcomeOfError('capture', Object.assign(new Error('x'), { status: 409, detail: 'log_sink_off' }))).key).toBe('operations.refusal.log_sink_off')
    const op = outcomeOfError('rules', Object.assign(new Error('x'), { status: 403, detail: 'not_operator' }))
    expect(outcomeMessage(op).key).toBe('operations.refusal.not_operator')
  })

  test('sendAlertAction: an answer or a thrown refusal, never a throw', async () => {
    const ok = await sendAlertAction('mute', async () => ({ ok: true, audit_id: 9, until: '2026-11-04T13:00:00Z' }))
    expect(ok).toMatchObject({ code: 'muted', auditId: '9' })
    const refused = await sendAlertAction('mute', async () => {
      throw Object.assign(new Error('x'), { status: 400, detail: 'invalid_minutes' })
    })
    expect(refused).toMatchObject({ tone: 'refused', code: 'invalid_minutes' })
  })

  test('the toast carries the sentence and the audit id', () => {
    toasts.length = 0
    showOutcomeToast(((key: string, values?: Record<string, string>) => (values?.id ? `${key}:${values.id}` : key)) as never, outcomeOf('ack', { ok: true, audit_id: 12 } as never))
    expect(toasts).toEqual([{ level: 'success', text: 'operations.actions.result.acked · operations.actions.auditRef:12' }])
  })

  test('alert actions are offered only while the evaluator runs on PostgreSQL', () => {
    expect(alertActionsEnabled(status())).toBe(true)
    expect(alertActionsEnabled(status({ alerts: false }))).toBe(false)
    expect(alertActionsEnabled(status({ available: false }))).toBe(false)
    expect(alertActionsEnabled(null)).toBe(false)
  })
})

describe('request shapes (§3.7)', () => {
  test('ack, mute, unmute, rules, maintenance and capture send what the catalogue says', async () => {
    const calls: { method: string; url: string; body?: unknown }[] = []
    const original = { post: axiosInstance.post, put: axiosInstance.put }
    axiosInstance.post = (async (url: string, body?: unknown) => {
      calls.push({ method: 'post', url, body })
      return { data: { ok: true, audit_id: 1 } }
    }) as never
    axiosInstance.put = (async (url: string, body?: unknown) => {
      calls.push({ method: 'put', url, body })
      return { data: { ok: true, audit_id: 2 } }
    }) as never
    try {
      await opsApi.ackAlert('12', 'on it')
      await opsApi.muteRule('run_hung', 240, 'deploy')
      await opsApi.muteRule('run_hung', 0, 'done')
      await opsApi.saveRules({ stale_docs: { stale_doc_minutes: 90 } }, 'too noisy')
      await opsApi.setMaintenance(60, 'deploy')
      await opsApi.setMaintenance(0, 'deploy done')
      await opsApi.startCapture({ module: 'pipeline', minutes: 60, reason: 'debug' })
    } finally {
      axiosInstance.post = original.post
      axiosInstance.put = original.put
    }
    expect(calls).toEqual([
      { method: 'post', url: '/admin/ops/alerts/12/ack', body: { reason: 'on it' } },
      { method: 'post', url: '/admin/ops/alerts/rules/run_hung/mute', body: { minutes: 240, reason: 'deploy' } },
      { method: 'post', url: '/admin/ops/alerts/rules/run_hung/mute', body: { minutes: 0, reason: 'done' } },
      { method: 'put', url: '/admin/ops/alerts/rules', body: { rules: { stale_docs: { stale_doc_minutes: 90 } }, reason: 'too noisy' } },
      { method: 'post', url: '/admin/ops/maintenance', body: { minutes: 60, reason: 'deploy' } },
      { method: 'post', url: '/admin/ops/maintenance', body: { minutes: 0, reason: 'deploy done' } },
      { method: 'post', url: '/admin/ops/logs/capture', body: { module: 'pipeline', minutes: 60, reason: 'debug' } }
    ])
  })
})

// ---------- views ----------

describe('alerts list states', () => {
  const page = (patch: Partial<AlertsPage> = {}): AlertsPage => ({ items: [], next_before: null, email: null, ...patch })

  test('turned off, empty open, empty history', () => {
    expect(render(<AlertListBody data={page({ enabled: false })} mode="open" now={NOW} email={null} mutes={[]} canAct />)).toContain('operations.alerts.off')
    expect(render(<AlertListBody data={page()} mode="open" now={NOW} email={null} mutes={[]} canAct />)).toContain('operations.alerts.noneOpen')
    expect(render(<AlertListBody data={page()} mode="history" now={NOW} email={null} mutes={[]} canAct />)).toContain('operations.alerts.noneHistory')
  })

  test('the Alerts section says so when the server has alerts off or no PostgreSQL, and fetches nothing', () => {
    expect(render(<AlertsView />, status({ alerts: false }))).toContain('operations.alerts.off')
    expect(render(<AlertsView />, status({ available: false }))).toContain('operations.unavailableShort')
  })

  test('first load is a sized skeleton under the sub-view row', () => {
    const html = render(<AlertsView />)
    expect(html).toContain('operations.alerts.sub.open')
    expect(html).toContain('operations.alerts.sub.maintenance')
    expect(html).toContain('ops-skeleton')
  })

  test('open rows: severity, rule, scope, notified, runbook, Ack and Mute ▾; the title opens the drawer, the arrow is a real link', () => {
    const html = render(
      <AlertTable
        mode="open"
        now={NOW}
        email={{ enabled: true, recipients: 1, ok: true, reason: null, since: null, failures: null, last_sent_at: null }}
        canAct
        alerts={[alert({ runbook_url: 'https://wiki.example/runbooks/dependency_down', runbook: 'Check the Milvus container' })]}
      />
    )
    expect(html).toContain('operations.state.critical')
    expect(html).toContain('dependency_down')
    // The dependency's name (its i18n default here) opens its drawer.
    expect(html).toContain('>milvus</button>')
    expect(html).toContain('operations.alerts.email.pending')
    expect(html).toContain('Check the Milvus container')
    expect(html).toContain('href="https://wiki.example/runbooks/dependency_down"')
    expect(html).toContain('rel="noopener noreferrer"')
    expect(html).toContain('operations.alerts.ack.button')
    expect(html).toContain('operations.alerts.mute.button')
    expect(html).toContain('href="#/?ops=1&amp;otab=health&amp;osub=deps&amp;odep=milvus"')
  })

  test('an acked alert shows who acked it and offers no second Ack; without alert actions there are no buttons', () => {
    const acked = alert({ acked_by: 'opso', acked_at: '2026-11-04T11:10:00Z', ack_note: 'restarting milvus' })
    const html = render(<AlertTable mode="open" now={NOW} email={null} canAct alerts={[acked]} />)
    expect(html).toContain('opso')
    expect(html).toContain('restarting milvus')
    expect(html).not.toContain('operations.alerts.ack.button')
    expect(html).toContain('operations.alerts.mute.button')
    const ro = render(<AlertTable mode="open" now={NOW} email={null} canAct={false} alerts={[alert()]} />)
    expect(ro).not.toContain('operations.alerts.ack.button')
    expect(ro).not.toContain('operations.alerts.mute.button')
  })

  test('a muted rule is marked on its rows; history shows when it resolved; server text is escaped', () => {
    const muted = render(
      <AlertTable mode="open" now={NOW} email={null} canAct alerts={[alert()]} mutes={[{ rule: 'dependency_down', until: '2026-11-04T13:00:00Z', reason: null, by: null, at: null }]} />
    )
    expect(muted).toContain('operations.alerts.mutedUntil')
    const history = render(
      <AlertTable mode="history" now={NOW} email={null} canAct alerts={[alert({ state: 'resolved', resolved_at: '2026-11-04T11:31:00Z', title: '<img src=x onerror=1>' })]} />
    )
    expect(history).toContain('operations.alerts.col.resolved')
    expect(history).not.toContain('operations.alerts.ack.button')
    expect(history).toContain('&lt;img')
    expect(history).not.toContain('<img')
  })

  test('only plain web links are followed for runbooks', () => {
    expect(safeHttpUrl('https://wiki/x')).toBe('https://wiki/x')
    expect(safeHttpUrl('javascript:alert(1)')).toBeNull()
    expect(safeHttpUrl('//evil')).toBeNull()
    const html = render(<AlertTable mode="open" now={NOW} email={null} canAct alerts={[alert({ runbook_url: 'javascript:alert(1)' })]} />)
    expect(html).not.toContain('javascript:')
  })

  test('the email mode note: record-only, nobody to send to, or on with N recipients', () => {
    const base = { recipients: 2, ok: true, reason: null, since: null, failures: null, last_sent_at: null }
    expect(render(<EmailModeNote email={{ ...base, enabled: false }} />)).toContain('operations.alerts.emailMode.recordOnly')
    expect(render(<EmailModeNote email={{ ...base, enabled: true, recipients: 0 }} />)).toContain('operations.alerts.emailMode.noRecipients')
    expect(render(<EmailModeNote email={{ ...base, enabled: true }} />)).toContain('operations.alerts.emailMode.on')
    expect(render(<EmailModeNote email={null} />)).toBe('')
  })
})

describe('mute and ack flows', () => {
  test('Mute ▾ is a labelled menu trigger; Ack… opens a reason dialog (nothing is sent without one)', () => {
    const mute = render(<MuteMenu rule="run_hung" ruleTitle="Run hung" />)
    expect(mute).toContain('aria-label="operations.alerts.mute.menuLabel"')
    expect(mute).toContain('aria-haspopup="dialog"')
    const ack = render(<AckButton alertId="12" title="Milvus down" />)
    expect(ack).toContain('operations.alerts.ack.button')
    // The dialog is closed until clicked: no reason field, no request.
    expect(ack).not.toContain('<textarea')
  })

  test('the mutes list shows who, why and until when, with Unmute… for admins who may act', () => {
    const mutes = [
      { rule: 'run_hung', until: '2026-11-04T13:00:00Z', reason: 'deploy', by: 'opso', at: null },
      { rule: 'stale_docs', until: '2026-11-04T11:00:00Z', reason: 'old', by: 'opso', at: null }
    ]
    const html = render(<MutesList mutes={mutes} now={NOW} canAct />)
    expect(html).toContain('run_hung')
    expect(html).not.toContain('stale_docs')
    expect(html).toContain('operations.alerts.unmute.button')
    expect(render(<MutesList mutes={mutes} now={NOW} canAct={false} />)).not.toContain('operations.alerts.unmute.button')
    expect(render(<MutesList mutes={[]} now={NOW} canAct />)).toContain('operations.alerts.mutes.none')
  })

  test('the Alert drawer summary: severity and state as icon plus text, acked by, scope links', () => {
    const firing = render(<AlertSummary alert={alert({ acked_by: 'opso' })} now={NOW} />)
    expect(firing).toContain('operations.alerts.state.firing')
    expect(firing).toContain('opso')
    expect(firing).toContain('>milvus</button>')
    const resolved = render(<AlertSummary alert={alert({ state: 'resolved', resolved_at: '2026-11-04T11:31:00Z' })} now={NOW} />)
    expect(resolved).toContain('operations.alerts.state.resolved')
    expect(resolved).toContain('operations.alertDrawer.wasOpen')
  })

  test('an alert email link (oal) opens the Alert drawer', () => {
    expect(readOpsHash('#/?ops=1&oal=12')?.drawer).toEqual({ kind: 'alert', id: '12' })
  })

  test('an attention item backed by an alert links to it beside its drill link, never nested', () => {
    const html = render(
      <AttentionList items={[{ id: 'a', rule: 'run_hung', severity: 'serious', workspace: 'fsktm', alert_id: '41' }]} base={DEFAULT_OPS_HASH} baseline now={NOW} empty="none" />
    )
    expect(html).toContain('oal=41')
    expect(html).toContain('operations.attention.openAlert')
    const acked = render(
      <AttentionList items={[{ id: 'a', rule: 'run_hung', severity: 'serious', alert_id: '41', acked_by: 'opso', muted: true }]} base={DEFAULT_OPS_HASH} baseline now={NOW} empty="none" />
    )
    expect(acked).toContain('operations.attention.ackedBy')
    expect(acked).toContain('operations.attention.muted')
    expect(html).not.toMatch(/<a [^>]*>(?:(?!<\/a>).)*<a /s)
  })
})

// ---------- banners ----------

const overviewPoll = (data: Partial<OpsOverview>, baseline = true) => ({
  data: normalizeOverview({ state: 'ok', kpis: {}, attention: [], ...data }),
  error: null,
  loading: false,
  slow: false,
  updatedAt: NOW,
  previousAt: null,
  baseline,
  reload: () => {}
})

describe('shell banners (FR-63, FR-64)', () => {
  test('alert email not delivered: why, since when, failures, and Open SMTP; not a live region', () => {
    const html = render(<AlertEmailBanner channel={{ ok: false, reason: 'smtp_probe_failing', since: '2026-11-04T09:12:00Z', failures: 3 }} onOpenSmtp={() => {}} />)
    expect(html).toContain('operations.banner.email.title')
    expect(html).toContain('operations.banner.email.reason.smtp_probe_failing')
    expect(html).toContain('operations.banner.email.since')
    expect(html).toContain('operations.banner.email.failures')
    expect(html).toContain('operations.banner.email.open')
    expect(html).not.toContain('aria-live')
    expect(html).not.toContain('role="status"')
    // A reason code the console does not know reads generically.
    expect(render(<AlertEmailBanner channel={{ ok: false, reason: 'weird' }} />)).toContain('operations.banner.email.reason.unknown')
  })

  test('maintenance: until when, by whom and why; End… only for admins who may act', () => {
    const m = { until: '2026-11-04T13:00:00Z', reason: 'deploy', by: 'opso' }
    expect(render(<MaintenanceBanner maintenance={m} onEnd={() => {}} />)).toContain('operations.maintenance.endButton')
    const html = render(<MaintenanceBanner maintenance={m} />)
    expect(html).toContain('operations.banner.maintenance.title')
    expect(html).toContain('operations.banner.maintenance.effect')
    expect(html).not.toContain('operations.maintenance.endButton')
  })

  test('the shell shows them from the overview poll; nothing when all is well or the maintenance ended', () => {
    const both = render(<OpsBanners />, status(), overviewPoll({ alert_channel: { ok: false, reason: 'smtp_not_configured' }, maintenance: { until: '2999-01-01T00:00:00Z', reason: 'x', by: 'y' } }))
    expect(both).toContain('data-banner="alert-email"')
    expect(both).toContain('data-banner="maintenance"')
    expect(render(<OpsBanners />, status(), overviewPoll({ alert_channel: { ok: true } }))).toBe('')
    expect(render(<OpsBanners />, status(), overviewPoll({ maintenance: { until: '2000-01-01T00:00:00Z', reason: null, by: null } }))).toBe('')
    expect(render(<OpsBanners />, status(), null)).toBe('')
  })
})

// ---------- INFO capture ----------

describe('INFO capture countdown (FR-26)', () => {
  const state = (patch: Partial<LogCaptureState> = {}): LogCaptureState => ({
    active: { module: 'pipeline', expires_at: '2026-11-04T12:42:13Z', by: 'opso', reason: 'debug', started_at: '2026-11-04T11:42:13Z' },
    modules: ['pipeline', 'operate'],
    max_minutes: 60,
    sink_enabled: true,
    generated_at: '2026-11-04T12:00:00Z',
    ...patch
  })

  test('counts down on the server clock, so a skewed browser clock neither stretches nor cuts it', () => {
    // The browser clock runs 5 minutes ahead of the server; the answer arrived at local `fetched`.
    const fetched = NOW + 5 * 60_000
    expect(captureRemaining(state(), fetched, fetched)).toBe(42 * 60 + 13)
    expect(captureRemaining(state(), fetched, fetched + 13_000)).toBe(42 * 60)
    expect(captureRemaining(state(), fetched, fetched + 3 * 3600_000)).toBe(0)
    expect(captureRemaining(state({ active: null }), fetched, fetched)).toBeNull()
    // Without the server time, the local clock is used.
    expect(captureRemaining(state({ generated_at: undefined }), NOW, NOW)).toBe(42 * 60 + 13)
  })

  test('formats as m:ss, and h:mm:ss at an hour', () => {
    expect(fmtCountdown(42 * 60 + 13)).toBe('42:13')
    expect(fmtCountdown(5)).toBe('0:05')
    expect(fmtCountdown(3600)).toBe('1:00:00')
    expect(fmtCountdown(null)).toBe('–')
  })

  test('the countdown is plain text (no motion), and says when it ended', () => {
    // The component ticks on the real clock; the answer is taken as just arrived.
    const fetched = Date.now()
    const html = render(<CaptureCountdown state={state()} fetchedAt={fetched} />, status(), null, true)
    expect(html).toContain('operations.capture.active')
    expect(html).toContain('operations.capture.left')
    expect(html).not.toMatch(/ops-(breathe|ring|pulse|flow|fade-in)/)
    const ended = render(<CaptureCountdown state={state({ generated_at: '2026-11-04T13:00:00Z' })} fetchedAt={fetched} />)
    expect(ended).toContain('operations.capture.ended')
  })

  const info = (patch: Partial<{ supported: boolean; canCapture: boolean; data: LogCaptureState | null }> = {}) => ({
    supported: patch.supported ?? true,
    canCapture: patch.canCapture ?? true,
    sinkOff: false,
    poll: { data: patch.data === undefined ? state({ active: null }) : patch.data, error: null, loading: false, slow: false, updatedAt: NOW, previousAt: null, baseline: true, reload: () => {} }
  })

  test('operators start a capture; other admins see the control disabled with a hint; an older server shows nothing', () => {
    const op = render(<LogCaptureControl info={info()} open={false} onOpenChange={() => {}} onShow={() => {}} />)
    expect(op).toContain('operations.capture.start')
    expect(op).not.toMatch(/<button[^>]*disabled=""[^>]*>operations\.capture\.start/)
    const admin = render(<LogCaptureControl info={info({ canCapture: false })} open={false} onOpenChange={() => {}} onShow={() => {}} />, status({ operator: false }))
    expect(admin).toMatch(/<button[^>]*disabled=""[^>]*title="operations\.capture\.operatorOnly"/)
    expect(render(<LogCaptureControl info={info({ supported: false })} open={false} onOpenChange={() => {}} onShow={() => {}} />)).toBe('')
  })

  test('a running capture shows its countdown, Show INFO lines and Stop… (operators only)', () => {
    const html = render(<LogCaptureControl info={info({ data: state() })} open={false} onOpenChange={() => {}} onShow={() => {}} />)
    expect(html).toContain('data-capture="on"')
    expect(html).toContain('operations.capture.show')
    expect(html).toContain('operations.capture.stop')
    const admin = render(<LogCaptureControl info={info({ data: state(), canCapture: false })} open={false} onOpenChange={() => {}} onShow={() => {}} />)
    expect(admin).toContain('operations.capture.show')
    expect(admin).not.toContain('operations.capture.stop')
  })
})

// ---------- motion ----------

describe('motion for banners and alerts (§2.12)', () => {
  test('a banner fades in only when it appears on a later poll: never on first paint or a baseline poll', () => {
    expect(bannerEnters(null, true, false)).toBe(false)
    expect(bannerEnters(false, true, true)).toBe(false)
    expect(bannerEnters(false, true, false)).toBe(true)
    expect(bannerEnters(true, true, false)).toBe(false)
    expect(bannerEnters(false, false, false)).toBe(false)
  })

  test('first paint of a banner carries no animation class', () => {
    const html = render(<OpsBanners />, status(), overviewPoll({ alert_channel: { ok: false, reason: 'smtp_probe_failing' } }, false))
    expect(html).toContain('data-banner="alert-email"')
    expect(html).not.toContain('ops-fade-in')
    expect(render(<AlertEmailBanner channel={{ ok: false }} enter />)).toContain('ops-fade-in')
  })

  test('the banner fade is opacity only, and the reduced-motion kill switch covers it', () => {
    const css = readFileSync(new URL('./ops.css', import.meta.url), 'utf8')
    const banner = css.slice(css.indexOf('.ops-banner {'), css.indexOf('.ops-banner[data-tone="crit"]'))
    expect(banner).not.toMatch(/animation|transition|transform/)
    expect(css).toMatch(/@keyframes ops-fade-in \{\s*from \{ opacity: 0; \}\s*to \{ opacity: 1; \}\s*\}/)
    expect(css).toContain('.ops-root[data-motion="reduced"] *')
    expect(css).toMatch(/@media \(prefers-reduced-motion: reduce\) \{\s*\.ops-root \*,/)
  })

  test('alert tables render the same twice: unchanged data adds no animation', () => {
    const a = render(<AlertTable mode="open" now={NOW} email={null} canAct alerts={[alert()]} />, status(), null, true)
    const b = render(<AlertTable mode="open" now={NOW} email={null} canAct alerts={[alert()]} />, status(), null, true)
    expect(a).toBe(b)
    expect(a).not.toMatch(/ops-(ring|breathe|pulse|flow)/)
  })
})

// ---------- incident export ----------

describe('incident export window (FR-65)', () => {
  test('local datetime inputs round-trip to UTC instants', () => {
    const ms = new Date(2026, 10, 4, 9, 30).getTime()
    expect(toLocalInput(ms)).toBe('2026-11-04T09:30')
    expect(fromLocalInput('2026-11-04T09:30')).toBe(new Date(ms).toISOString())
    expect(fromLocalInput('04/11/2026 09:30')).toBeNull()
  })

  test('starts at the last 6 hours, or around an alert: 30 min before it opened to 10 min after it resolved', () => {
    const d = incidentDefaults(NOW)
    expect(fromLocalInput(d.to)).toBe(new Date(Math.floor(NOW / 60_000) * 60_000).toISOString())
    expect(Date.parse(fromLocalInput(d.to)!) - Date.parse(fromLocalInput(d.from)!)).toBe(6 * 3600_000)
    const around = incidentDefaults(NOW, { first_seen: '2026-11-04T10:00:00Z', resolved_at: '2026-11-04T10:40:00Z' })
    expect(fromLocalInput(around.from)).toBe('2026-11-04T09:30:00.000Z')
    expect(fromLocalInput(around.to)).toBe('2026-11-04T10:50:00.000Z')
  })

  test('an end that is not after the start, a missing time, or more than 31 days cannot be sent', () => {
    expect(incidentProblem({ from: '2026-10-01T10:00', to: '2026-11-04T09:00' })).toBe('too_long')
    expect(incidentProblem({ from: '2026-10-05T10:00', to: '2026-11-04T09:00' })).toBeNull()
    expect(incidentProblem({ from: '2026-11-04T10:00', to: '2026-11-04T09:00' })).toBe('order')
    expect(incidentProblem({ from: '2026-11-04T10:00', to: '' })).toBe('malformed')
    expect(incidentProblem({ from: '2026-11-04T09:00', to: '2026-11-04T10:00' })).toBeNull()
  })
})

