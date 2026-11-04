import type {
  AlertChannel,
  AlertDetail,
  AlertEmailState,
  AlertMute,
  AlertRule,
  AlertRules,
  AlertScope,
  AlertsPage,
  LogCapture,
  LogCaptureState,
  Maintenance,
  OpsAlert,
  OpsOverview,
  OpsTarget,
  RuleThreshold,
  RulesChange
} from '../api'
import type { HashTarget } from '../hashState'
import { readOpsHash } from '../hashState'
import { attentionTarget } from '../overview'
import { rankOf, toState } from '../severity'

/**
 * Alerts (spec §1.6 FR-60–FR-65, §1.7, §2.8): read models for the Alerts
 * section, the Alert drawer, the shell banners and the INFO capture control,
 * plus the pure rules behind them (what is emailed, mute and maintenance
 * durations, threshold validation, countdowns). The server owns every
 * decision; these only decide what to show and what not to offer.
 */

type Loose = Record<string, any>

const obj = (v: unknown): Loose => (v && typeof v === 'object' && !Array.isArray(v) ? (v as Loose) : {})
const num = (v: unknown): number | null => (typeof v === 'number' && Number.isFinite(v) ? v : null)
const str = (v: unknown): string | null => (typeof v === 'string' && v ? v : null)
const arr = (v: unknown): unknown[] => (Array.isArray(v) ? v : [])
const idText = (v: unknown): string | null => (typeof v === 'number' && Number.isFinite(v) ? String(v) : str(v))
const bool = (v: unknown): boolean | null => (typeof v === 'boolean' ? v : null)

// ---------- durations ----------

const HOUR = 60
const DAY = 24 * HOUR

/** FR-61: mutes for 1, 4 or 24 h, at most 7 days. */
export const MUTE_PRESETS = [HOUR, 4 * HOUR, DAY, 7 * DAY] as const
export const MUTE_MAX_MINUTES = 7 * DAY

/** Maintenance covers a deploy or a planned change; a day at most, so a forgotten one ends on its own. */
export const MAINTENANCE_PRESETS = [30, HOUR, 2 * HOUR, 4 * HOUR, 8 * HOUR, DAY] as const
export const MAINTENANCE_MAX_MINUTES = DAY

/** FR-26: an INFO capture lasts at most 60 minutes. */
export const CAPTURE_PRESETS = [5, 15, 30, HOUR] as const
export const CAPTURE_MAX_MINUTES = HOUR

/** "1 h", "4 h", "7 days": an i18n key under operations.alerts.duration and its count. */
export function durationKey(minutes: number): { key: string; count: number } {
  if (minutes >= DAY && minutes % DAY === 0) return { key: 'operations.alerts.duration.days', count: minutes / DAY }
  if (minutes >= HOUR && minutes % HOUR === 0) return { key: 'operations.alerts.duration.hours', count: minutes / HOUR }
  return { key: 'operations.alerts.duration.minutes', count: minutes }
}

/** A duration a request may carry: a whole number of minutes from 1 to `max`. */
export const validMinutes = (minutes: number, max: number) => Number.isInteger(minutes) && minutes >= 1 && minutes <= max

// ---------- normalisers ----------

/** `link` is an ops hash (`#/?ops=1&…`) or its query; anything else is ignored, never followed. */
function targetOfLink(link: unknown): OpsTarget | null {
  const raw = str(link)
  if (!raw) return null
  const view = readOpsHash(raw.startsWith('#') ? raw : `#/?${raw.replace(/^[/?#]+/, '')}`)
  if (!view) return null
  const d = view.drawer
  const target: OpsTarget = { tab: view.tab, sub: view.sub }
  if (d?.kind === 'document') return { ...target, workspace: d.workspace, doc_id: d.id }
  if (d?.kind === 'workspace') return { ...target, workspace: d.id }
  if (d?.kind === 'run') return { ...target, run_id: d.id }
  if (d?.kind === 'dependency') return { ...target, dep: d.id }
  if (d?.kind === 'worker') return { ...target, worker_id: d.id }
  // A link to the alert itself (the email's `oal`) or to the bare Overview adds no drill target.
  return view.tab === 'overview' && !view.sub ? null : target
}

function normalizeScope(raw: unknown): AlertScope {
  const s = obj(raw)
  return {
    ...s,
    workspace: str(s.workspace),
    dep: str(s.dep) ?? str(s.dependency),
    worker_id: str(s.worker_id),
    run_id: str(s.run_id),
    count: num(s.count)
  }
}

/** One alert row; rows without an id or a rule are dropped. */
export function normalizeAlert(raw: unknown): OpsAlert | null {
  const a = obj(raw)
  const id = idText(a.id)
  const rule = str(a.rule)
  if (!id || !rule) return null
  const resolvedAt = str(a.resolved_at)
  return {
    id,
    rule,
    fingerprint: str(a.fingerprint),
    severity: str(a.severity) ?? 'warning',
    // The partial unique index keys "open" on resolved_at; a row with it set is resolved whatever `state` says.
    state: resolvedAt ? 'resolved' : (str(a.state) ?? 'firing'),
    title: str(a.title),
    label: str(a.label),
    scope: normalizeScope(a.scope),
    target: a.target ? (obj(a.target) as OpsTarget) : targetOfLink(a.link),
    runbook: str(a.runbook),
    runbook_url: str(a.runbook_url),
    first_seen: str(a.first_seen),
    last_seen: str(a.last_seen),
    resolved_at: resolvedAt,
    acked_by: str(a.acked_by),
    acked_at: str(a.acked_at),
    ack_note: str(a.ack_note),
    last_emailed_at: str(a.last_emailed_at),
    email: str(a.email) ?? str(a.notified),
    muted_until: str(a.muted_until)
  }
}

export function normalizeAlertEmail(raw: unknown): AlertEmailState | null {
  if (!raw || typeof raw !== 'object') return null
  const e = obj(raw)
  const recipients = num(e.recipients) ?? (typeof e.recipients === 'boolean' ? (e.recipients ? 1 : 0) : num(e.recipient_count))
  return {
    enabled: bool(e.enabled) ?? bool(e.email_enabled),
    recipients,
    ok: bool(e.ok),
    reason: str(e.reason),
    since: str(e.since),
    failures: num(e.failures),
    last_sent_at: str(e.last_sent_at)
  }
}

export function normalizeAlertsPage(raw: unknown): AlertsPage {
  const p = obj(raw)
  const items = arr(p.items ?? p.alerts)
    .map(normalizeAlert)
    .filter((a): a is OpsAlert => a !== null)
  return {
    ...p,
    items,
    next_before: idText(p.next_before),
    enabled: typeof p.enabled === 'boolean' ? p.enabled : undefined,
    email: normalizeAlertEmail(p.email ?? p.channel)
  } as AlertsPage
}

const BELOW_RULES = new Set(['disk_free', 'tls_cert_expiry'])

function normalizeThreshold(key: string, raw: unknown, defaults: Loose, bounds: Loose): RuleThreshold | null {
  const t = typeof raw === 'number' ? { value: raw } : obj(raw)
  const value = num(t.value)
  if (!key || value === null) return null
  const b = bounds[key]
  const pair = Array.isArray(b) ? b : null
  return {
    key,
    value,
    default: num(t.default) ?? num(defaults[key]),
    min: num(t.min) ?? (pair ? num(pair[0]) : num(obj(b).min)),
    max: num(t.max) ?? (pair ? num(pair[1]) : num(obj(b).max)),
    unit: str(t.unit) ?? unitOfKey(key),
    integer: t.integer === true || t.type === 'int' || t.type === 'integer'
  }
}

/** A unit the threshold's name states; nothing is guessed for bare numbers and ratios. */
export function unitOfKey(key: string): string | null {
  if (/_minutes$/.test(key)) return 'min'
  if (/_seconds$|_s$/.test(key)) return 's'
  if (/_ms$/.test(key)) return 'ms'
  if (/_days$/.test(key)) return 'd'
  if (/_pct$|_percent$/.test(key)) return '%'
  return null
}

/** One rule: thresholds as a list or as a `{key: value}` map (with `defaults` and `bounds` beside it). */
export function normalizeAlertRule(raw: unknown): AlertRule | null {
  const r = obj(raw)
  const rule = str(r.rule) ?? str(r.id) ?? str(r.name)
  if (!rule) return null
  const defaults = obj(r.defaults)
  const bounds = obj(r.bounds)
  const thresholds = (
    Array.isArray(r.thresholds)
      ? r.thresholds.map((t: unknown) => normalizeThreshold(str(obj(t).key) ?? '', t, defaults, bounds))
      : Object.entries(obj(r.thresholds)).map(([key, v]) => normalizeThreshold(key, v, defaults, bounds))
  ).filter((t): t is RuleThreshold => t !== null)
  const severities = (Array.isArray(r.severities) ? r.severities : [r.severity, r.max_severity])
    .map((s: unknown) => str(s))
    .filter((s): s is string => !!s)
  const uniq = Array.from(new Set(severities.map((s) => toState(s)).filter((s) => rankOf(s) > 0))).sort((a, b) => rankOf(a) - rankOf(b))
  const direction = r.direction === 'above' || r.direction === 'below' ? r.direction : BELOW_RULES.has(rule) ? 'below' : null
  const mute = obj(r.mute)
  return {
    rule,
    label: str(r.label),
    severities: uniq,
    email: typeof r.email === 'boolean' ? r.email : typeof r.emailed === 'boolean' ? r.emailed : rule !== 'provider_paused' && uniq.some((s) => rankOf(s) >= 2),
    condition: str(r.condition),
    runbook: str(r.runbook),
    runbook_url: str(r.runbook_url),
    thresholds,
    direction,
    muted_until: str(r.muted_until) ?? str(mute.until),
    mute_reason: str(r.mute_reason) ?? str(mute.reason),
    muted_by: str(r.muted_by) ?? str(mute.by)
  }
}

export function normalizeMaintenance(raw: unknown): Maintenance | null {
  if (!raw || typeof raw !== 'object') return null
  const m = obj(raw)
  const until = str(m.until)
  if (!until && m.active !== true) return null
  return { until, reason: str(m.reason), by: str(m.by) ?? str(m.actor), at: str(m.at) ?? str(m.started_at) }
}

function normalizeMute(raw: unknown, rule?: string): AlertMute | null {
  const m = obj(raw)
  const name = str(m.rule) ?? rule ?? null
  if (!name) return null
  return { rule: name, until: str(m.until), reason: str(m.reason), by: str(m.by) ?? str(m.actor), at: str(m.at) }
}

/** GET /alerts/rules. Mutes come as a list or a `{rule: mute}` map; rules carrying a mute add theirs. */
export function normalizeAlertRules(raw: unknown): AlertRules {
  const p = obj(raw)
  const rules = arr(p.rules)
    .map(normalizeAlertRule)
    .filter((r): r is AlertRule => r !== null)
  const listed = Array.isArray(p.mutes)
    ? p.mutes.map((m: unknown) => normalizeMute(m))
    : Object.entries(obj(p.mutes)).map(([rule, m]) => normalizeMute(m, rule))
  const mutes = listed.filter((m): m is AlertMute => m !== null)
  for (const r of rules) {
    if (r.muted_until && !mutes.some((m) => m.rule === r.rule)) mutes.push({ rule: r.rule, until: r.muted_until, reason: r.mute_reason, by: r.muted_by, at: null })
  }
  return {
    ...p,
    rules,
    mutes,
    maintenance: normalizeMaintenance(p.maintenance),
    email: normalizeAlertEmail(p.email ?? p.channel),
    enabled: typeof p.enabled === 'boolean' ? p.enabled : undefined
  } as AlertRules
}

/** GET /alerts/{id}: `{alert, rule, history}`, or the bare row. */
export function normalizeAlertDetail(raw: unknown): AlertDetail {
  const p = obj(raw)
  const alert = normalizeAlert(p.alert ?? (p.id !== undefined ? p : null))
  return {
    ...(p.alert ? p : {}),
    available: typeof p.available === 'boolean' ? p.available : undefined,
    generated_at: str(p.generated_at) ?? undefined,
    alert,
    email: normalizeAlertEmail(p.email ?? p.channel),
    rule: p.rule && typeof p.rule === 'object' ? normalizeAlertRule(p.rule) : null,
    history: arr(p.history)
      .map(normalizeAlert)
      .filter((a): a is OpsAlert => a !== null && a.id !== alert?.id)
  } as AlertDetail
}

export function normalizeLogCapture(raw: unknown): LogCaptureState {
  const p = obj(raw)
  const a = obj(p.active ?? p.capture)
  const module = str(a.module)
  const expires = str(a.expires_at) ?? str(a.until)
  const active: LogCapture | null =
    module && expires ? { module, expires_at: expires, by: str(a.by) ?? str(a.actor), reason: str(a.reason), started_at: str(a.started_at) ?? str(a.at) } : null
  const max = num(p.max_minutes)
  return {
    ...p,
    active,
    modules: arr(p.modules ?? p.allowed_modules)
      .map((m) => str(m))
      .filter((m): m is string => !!m),
    max_minutes: max !== null && max > 0 ? Math.min(max, CAPTURE_MAX_MINUTES) : CAPTURE_MAX_MINUTES,
    sink_enabled: typeof p.sink_enabled === 'boolean' ? p.sink_enabled : null
  } as LogCaptureState
}

/** The overview's alert channel with its fields checked; null when the server does not report one. */
export function alertChannelOf(o: Pick<OpsOverview, 'alert_channel'> | null | undefined): AlertChannel | null {
  const c = o?.alert_channel
  if (!c || typeof c !== 'object') return null
  return {
    ok: typeof c.ok === 'boolean' ? c.ok : null,
    reason: str(c.reason),
    since: str(c.since),
    failures: num(c.failures),
    email_enabled: typeof c.email_enabled === 'boolean' ? c.email_enabled : null,
    last_sent_at: str(c.last_sent_at)
  }
}

// ---------- email ----------

/**
 * Whether a rule at a severity is ever emailed (FR-62, §1.7): critical and
 * serious only, and never `provider_paused`, which provider_alerts.py
 * already emails.
 */
export const emailedSeverity = (severity: string | null | undefined, rule: string) => rule !== 'provider_paused' && rankOf(severity) >= 2

export type EmailStatus = 'sent' | 'pending' | 'in_app_only' | 'record_only' | 'muted' | 'maintenance' | 'suppressed' | 'failed'

const EMAIL_STATUSES = new Set<EmailStatus>(['sent', 'pending', 'in_app_only', 'record_only', 'muted', 'maintenance', 'suppressed', 'failed'])

/**
 * The "Notified" cell: the server's word when it gives one; else sent once
 * the claim stamped `last_emailed_at`, in-app only for what is never
 * emailed, record-only while OPS_ALERT_EMAIL is off, pending otherwise.
 */
export function emailStatus(alert: Pick<OpsAlert, 'email' | 'last_emailed_at' | 'severity' | 'rule'>, email: AlertEmailState | null): EmailStatus {
  const said = (alert.email ?? '').toLowerCase()
  if (EMAIL_STATUSES.has(said as EmailStatus)) return said as EmailStatus
  if (said === 'in_app' || said === 'in-app only' || said === 'app') return 'in_app_only'
  if (alert.last_emailed_at) return 'sent'
  if (!emailedSeverity(alert.severity, alert.rule)) return 'in_app_only'
  if (email?.enabled === false) return 'record_only'
  return 'pending'
}

// ---------- views ----------

/** Where an alert drills into: its link, else its scope on the rule's view (as the attention list does). */
export function alertTarget(alert: OpsAlert): HashTarget {
  return attentionTarget({
    id: alert.id,
    rule: alert.rule,
    severity: alert.severity,
    target: alert.target,
    workspace: alert.scope.workspace ?? null,
    run_id: alert.scope.run_id ?? null,
    dep: alert.scope.dep ?? null,
    worker_id: alert.scope.worker_id ?? null
  })
}

/** Open alerts, worst first, then the oldest; the server order breaks ties. */
export function sortOpenAlerts(items: OpsAlert[]): OpsAlert[] {
  return items
    .map((a, i) => ({ a, i }))
    .sort((x, y) => rankOf(y.a.severity) - rankOf(x.a.severity) || x.i - y.i)
    .map(({ a }) => a)
}

/** Seconds an alert has been (or was) open. */
export function alertOpenSeconds(alert: Pick<OpsAlert, 'first_seen' | 'resolved_at'>, now: number): number | null {
  const start = alert.first_seen ? Date.parse(alert.first_seen) : Number.NaN
  if (!Number.isFinite(start)) return null
  const end = alert.resolved_at ? Date.parse(alert.resolved_at) : now
  return Number.isFinite(end) ? Math.max(0, (end - start) / 1000) : null
}

/** An instant that is still ahead (a mute, maintenance mode, a capture); false for missing or past times. */
export const isFuture = (iso: string | null | undefined, now: number) => {
  const t = iso ? Date.parse(iso) : Number.NaN
  return Number.isFinite(t) && t > now
}

export const activeMutes = (mutes: AlertMute[], now: number) => mutes.filter((m) => isFuture(m.until, now))

/** The mute holding a rule now, if any. */
export const muteOf = (rule: string, mutes: AlertMute[], now: number) => activeMutes(mutes, now).find((m) => m.rule === rule) ?? null

/** Maintenance is on while its end is ahead. */
export const maintenanceActive = (m: Maintenance | null | undefined, now: number) => !!m && isFuture(m.until, now)

/**
 * Whether a banner that is shown now should fade in: only when it appears on
 * a poll after the first (§2.12 rule 7: baseline polls replay nothing).
 */
export const bannerEnters = (wasShown: boolean | null, shown: boolean, baseline: boolean) => shown && wasShown === false && !baseline

// ---------- thresholds ----------

/** One draft value per rule and threshold key, as typed. */
export type RulesDraft = Record<string, Record<string, string>>

export type ThresholdError = 'required' | 'number' | 'positive' | 'integer' | 'min' | 'max' | 'order'

/** The draft that starts from the current values. */
export function draftOf(rules: AlertRule[]): RulesDraft {
  return Object.fromEntries(rules.map((r) => [r.rule, Object.fromEntries(r.thresholds.map((t) => [t.key, String(t.value)]))]))
}

const parse = (raw: string | undefined): number | null => {
  const v = (raw ?? '').trim()
  if (!v) return null
  const n = Number(v)
  return Number.isFinite(n) ? n : Number.NaN
}

/** One value on its own: present, a number, above zero (the server ignores the rest), whole, inside its bounds. */
export function thresholdError(t: RuleThreshold, raw: string | undefined): ThresholdError | null {
  const n = parse(raw)
  if (n === null) return 'required'
  if (Number.isNaN(n)) return 'number'
  if (n <= 0) return 'positive'
  if (t.integer && !Number.isInteger(n)) return 'integer'
  if (t.min !== null && n < t.min) return 'min'
  if (t.max !== null && n > t.max) return 'max'
  return null
}

/**
 * Warning/serious threshold pairs (alerts.py THRESHOLD_PAIRS): the server
 * refuses a serious level below its warning level, so the editor says so
 * before asking.
 */
export const THRESHOLD_PAIRS: readonly (readonly [string, string])[] = [
  ['stale_doc_minutes', 'stale_doc_serious_minutes'],
  ['hung_minutes', 'hung_serious_minutes'],
  ['oldest_pending_minutes', 'oldest_pending_serious_minutes'],
  ['email_waiting_minutes', 'email_waiting_serious_minutes'],
  ['email_silence_minutes', 'email_silence_serious_minutes'],
  ['llm_error_rate', 'llm_error_rate_serious'],
  ['llm_p95_ratio_warn', 'llm_p95_ratio_serious'],
  ['http_5xx_warn', 'http_5xx_serious'],
  ['dep_error_warn', 'dep_error_serious'],
  ['failure_spike_min', 'failure_spike_serious']
]

/**
 * Every error in the draft, by rule and key. Besides each value's own checks,
 * a serious level may not come before its warning level: not below it for a
 * rule where higher is worse, not above it where lower is worse.
 */
export function draftErrors(rules: AlertRule[], draft: RulesDraft): Record<string, Record<string, ThresholdError>> {
  const out: Record<string, Record<string, ThresholdError>> = {}
  for (const r of rules) {
    const errs: Record<string, ThresholdError> = {}
    const values = draft[r.rule] ?? {}
    for (const t of r.thresholds) {
      const e = thresholdError(t, values[t.key])
      if (e) errs[t.key] = e
    }
    const keys = new Set(r.thresholds.map((t) => t.key))
    for (const [warn, serious] of THRESHOLD_PAIRS) {
      if (!keys.has(warn) || !keys.has(serious) || errs[warn] || errs[serious]) continue
      const w = Number(values[warn])
      const s = Number(values[serious])
      const ordered = r.direction === 'below' ? s <= w : s >= w
      if (!ordered) errs[serious] = 'order'
    }
    if (Object.keys(errs).length) out[r.rule] = errs
  }
  return out
}

/** Only the values that changed, as numbers; null when nothing changed. */
export function rulesChange(rules: AlertRule[], draft: RulesDraft): RulesChange | null {
  const change: RulesChange = {}
  for (const r of rules) {
    for (const t of r.thresholds) {
      const n = parse(draft[r.rule]?.[t.key])
      if (n === null || Number.isNaN(n) || n === t.value) continue
      change[r.rule] = { ...change[r.rule], [t.key]: n }
    }
  }
  return Object.keys(change).length ? change : null
}

/** "stale_docs: stale_doc_minutes 60 → 90" lines for the save dialog. */
export function changeLines(rules: AlertRule[], change: RulesChange): { rule: string; key: string; from: number; to: number }[] {
  const lines: { rule: string; key: string; from: number; to: number }[] = []
  for (const r of rules) {
    for (const [key, to] of Object.entries(change[r.rule] ?? {})) {
      const t = r.thresholds.find((x) => x.key === key)
      if (t) lines.push({ rule: r.rule, key, from: t.value, to })
    }
  }
  return lines
}

// ---------- INFO capture ----------

/**
 * Seconds left in a capture, on the server's clock: the remaining time at
 * the server's `generated_at`, less what has passed here since the answer
 * arrived, so a skewed browser clock does not stretch or cut it. Zero once
 * it has expired; null without a capture.
 */
export function captureRemaining(
  state: Pick<LogCaptureState, 'active' | 'generated_at'> | null | undefined,
  fetchedAt: number | null,
  now: number
): number | null {
  const a = state?.active
  if (!a) return null
  const expires = Date.parse(a.expires_at)
  if (!Number.isFinite(expires)) return null
  const serverNow = state?.generated_at ? Date.parse(state.generated_at) : Number.NaN
  const base = fetchedAt ?? now
  const atAnswer = Number.isFinite(serverNow) ? (expires - serverNow) / 1000 : (expires - base) / 1000
  return Math.max(0, Math.floor(atAnswer - (now - base) / 1000))
}

/** "42:13" for a countdown, "1:00:00" at an hour. */
export function fmtCountdown(seconds: number | null): string {
  if (seconds === null) return '–'
  const s = Math.max(0, Math.floor(seconds))
  const h = Math.floor(s / 3600)
  const m = Math.floor((s % 3600) / 60)
  const pad = (v: number) => String(v).padStart(2, '0')
  return h ? `${h}:${pad(m)}:${pad(s % 60)}` : `${m}:${pad(s % 60)}`
}

// ---------- incident export ----------

const pad2 = (v: number) => String(v).padStart(2, '0')

/** Epoch ms → an `<input type="datetime-local">` value in local time ("2026-11-04T09:30"). */
export function toLocalInput(ms: number): string {
  const d = new Date(ms)
  return `${d.getFullYear()}-${pad2(d.getMonth() + 1)}-${pad2(d.getDate())}T${pad2(d.getHours())}:${pad2(d.getMinutes())}`
}

const LOCAL = /^(\d{4})-(\d{2})-(\d{2})T(\d{2}):(\d{2})$/

/** A datetime-local value (local time) → the UTC instant; null when malformed. */
export function fromLocalInput(value: string): string | null {
  const m = LOCAL.exec(value.trim())
  if (!m) return null
  const d = new Date(Number(m[1]), Number(m[2]) - 1, Number(m[3]), Number(m[4]), Number(m[5]))
  return Number.isNaN(d.getTime()) ? null : d.toISOString()
}

export type IncidentWindow = { from: string; to: string }

const MINUTE_MS = 60_000

/**
 * The window an incident export starts with: the last 6 hours, or around an
 * alert (30 min before it opened to 10 min after it resolved, or now).
 */
export function incidentDefaults(now: number, alert?: Pick<OpsAlert, 'first_seen' | 'resolved_at'> | null): IncidentWindow {
  const first = alert?.first_seen ? Date.parse(alert.first_seen) : Number.NaN
  if (Number.isFinite(first)) {
    const resolved = alert?.resolved_at ? Date.parse(alert.resolved_at) : Number.NaN
    const end = Number.isFinite(resolved) ? Math.min(resolved + 10 * MINUTE_MS, now) : now
    return { from: toLocalInput(first - 30 * MINUTE_MS), to: toLocalInput(Math.max(end, first)) }
  }
  return { from: toLocalInput(now - 6 * 60 * MINUTE_MS), to: toLocalInput(now) }
}

/** The server exports at most 31 days at once (alerts.py EXPORT_MAX_DAYS). */
export const INCIDENT_MAX_DAYS = 31

/** Why a window cannot be sent: a missing time, an end not after the start, or more than 31 days. */
export function incidentProblem(w: IncidentWindow): 'malformed' | 'order' | 'too_long' | null {
  const from = fromLocalInput(w.from)
  const to = fromLocalInput(w.to)
  if (!from || !to) return 'malformed'
  const span = Date.parse(to) - Date.parse(from)
  if (span <= 0) return 'order'
  return span > INCIDENT_MAX_DAYS * 24 * 60 * MINUTE_MS ? 'too_long' : null
}

/** The email state the overview's channel implies, for views without their own (the Alert drawer). */
export function emailFromChannel(c: AlertChannel | null): AlertEmailState | null {
  if (!c || (c.email_enabled === null && c.ok === null)) return null
  return {
    enabled: c.email_enabled ?? null,
    recipients: null,
    ok: c.ok,
    reason: c.reason ?? null,
    since: c.since ?? null,
    failures: typeof c.failures === 'number' ? c.failures : null,
    last_sent_at: c.last_sent_at ?? null
  }
}

/** A module name the capture may name: a dotted Python module or logger, nothing else. */
export const validModuleName = (name: string) => /^[A-Za-z_][A-Za-z0-9_.]{0,127}$/.test(name.trim())

/** The presets a server allows: those at or below its maximum, with the maximum itself offered. */
export function capturePresets(maxMinutes: number): number[] {
  const max = Math.max(1, Math.min(maxMinutes, CAPTURE_MAX_MINUTES))
  const list: number[] = CAPTURE_PRESETS.filter((m) => m <= max)
  return list.includes(max) ? list : [...list, max]
}
