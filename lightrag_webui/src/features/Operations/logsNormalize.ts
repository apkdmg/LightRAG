import type { Correlate, CorrelateMatch, CorrelateType, LogCursor, LogDetail, LogEntry, LogGroup, LogGroups, LogPage } from './api'
import { levelNumber } from './logs'

/**
 * Server answers for the log routes → the shapes the Logs tab reads (§3.7).
 * Like normalize.ts: both spellings a route may use are accepted, unknown
 * fields are dropped, and a partial payload never throws.
 */

type Loose = Record<string, any>

const obj = (v: unknown): Loose => (v && typeof v === 'object' && !Array.isArray(v) ? (v as Loose) : {})
const num = (v: unknown): number | null => (typeof v === 'number' && Number.isFinite(v) ? v : null)
const str = (v: unknown): string | null => (typeof v === 'string' && v ? v : null)
const arr = (v: unknown): unknown[] => (Array.isArray(v) ? v : [])
const intOf = (v: unknown): number | null => {
  if (typeof v === 'number' && Number.isFinite(v)) return Math.trunc(v)
  if (typeof v === 'string' && /^\d{1,19}$/.test(v)) return Number(v)
  return null
}

/** A keyset cursor as `{ts, id}`, `[ts, id]` or the string "ts,id"; null when unreadable. */
export function normalizeCursor(raw: unknown): LogCursor | null {
  if (!raw) return null
  if (typeof raw === 'string') {
    const at = raw.lastIndexOf(',')
    if (at <= 0) return null
    const id = intOf(raw.slice(at + 1))
    const ts = raw.slice(0, at)
    return id !== null && ts ? { ts, id } : null
  }
  if (Array.isArray(raw)) {
    const ts = str(raw[0])
    const id = intOf(raw[1])
    return ts && id !== null ? { ts, id } : null
  }
  const o = obj(raw)
  const ts = str(o.ts)
  const id = intOf(o.id)
  return ts && id !== null ? { ts, id } : null
}

/** One row; null when it has no id or time (it could not be keyed or ordered). */
export function normalizeLogEntry(raw: unknown): LogEntry | null {
  const o = obj(raw)
  const id = intOf(o.id)
  const ts = str(o.ts)
  if (id === null || !ts) return null
  const exc = typeof o.exc === 'string' ? o.exc : null
  return {
    id,
    ts,
    level: levelNumber(o.level ?? o.level_name ?? o.levelname),
    logger: str(o.logger) ?? '',
    module: str(o.module),
    lineno: num(o.lineno),
    pid: num(o.pid),
    worker_id: str(o.worker_id),
    workspace: str(o.workspace),
    request_id: str(o.request_id),
    client_request_id: str(o.client_request_id),
    run_id: str(o.run_id),
    track_id: str(o.track_id),
    doc_id: str(o.doc_id),
    operation: str(o.operation),
    source: str(o.source),
    msg: typeof o.msg === 'string' ? o.msg : typeof o.message === 'string' ? o.message : '',
    exc_type: str(o.exc_type),
    has_exc: o.has_exc === true || !!exc,
    exc,
    fingerprint: str(o.fingerprint),
    repeat_count: Math.max(1, num(o.repeat_count) ?? 1)
  }
}

export function normalizeLogPage(raw: unknown): LogPage {
  const o = obj(raw)
  const items: LogEntry[] = []
  for (const r of arr(o.items ?? o.entries)) {
    const e = normalizeLogEntry(r)
    if (e) items.push(e)
  }
  return {
    generated_at: str(o.generated_at) ?? undefined,
    available: o.available === false ? false : undefined,
    enabled: typeof o.enabled === 'boolean' ? o.enabled : undefined,
    items,
    next_before: normalizeCursor(o.next_before),
    newest_ts: str(o.newest_ts)
  }
}

/** /logs/{id}: the row with its redacted traceback, or the row inside `entry`. */
export function normalizeLogDetail(raw: unknown): LogDetail {
  const o = obj(raw)
  const entry = normalizeLogEntry(o.entry ?? o)
  return {
    generated_at: str(o.generated_at) ?? undefined,
    available: o.available === false ? false : undefined,
    entry,
    /** The server withheld the traceback (it may be shown to operators only). */
    exc_withheld: o.exc_withheld === true || o.exc_hidden === true
  }
}

export function normalizeLogGroups(raw: unknown): LogGroups {
  const o = obj(raw)
  const groups: LogGroup[] = []
  for (const r of arr(o.groups ?? o.items)) {
    const g = obj(r)
    const fingerprint = str(g.fingerprint)
    if (!fingerprint) continue
    groups.push({
      fingerprint,
      logger: str(g.logger) ?? '',
      module: str(g.module),
      lineno: num(g.lineno),
      level: levelNumber(g.level),
      exc_type: str(g.exc_type),
      sample: typeof g.sample === 'string' ? g.sample : (str(g.msg) ?? ''),
      count: Math.max(0, num(g.count) ?? 0),
      first_seen: str(g.first_seen),
      last_seen: str(g.last_seen),
      workspaces: num(g.workspaces)
    })
  }
  return { generated_at: str(o.generated_at) ?? undefined, available: o.available === false ? false : undefined, groups }
}

const CORRELATE_TYPES: readonly CorrelateType[] = ['request', 'log', 'run', 'track', 'doc', 'workspace', 'owner']

const TYPE_ALIASES: Record<string, CorrelateType> = {
  request_id: 'request',
  run_id: 'run',
  track_id: 'track',
  doc_id: 'doc',
  document: 'doc',
  email: 'owner',
  log_id: 'log',
  entry: 'log'
}

/** /correlate: at most 10 matches, each with its type, id and (where known) workspace and time. */
export function normalizeCorrelate(raw: unknown): Correlate {
  const o = obj(raw)
  const matches: CorrelateMatch[] = []
  for (const r of arr(o.matches ?? o.items)) {
    const m = obj(r)
    const rawType = str(m.type)?.toLowerCase() ?? ''
    const type = (CORRELATE_TYPES as readonly string[]).includes(rawType) ? (rawType as CorrelateType) : TYPE_ALIASES[rawType]
    const id = str(m.id) ?? (typeof m.id === 'number' ? String(m.id) : null)
    if (!type || !id) continue
    matches.push({ type, id, workspace: str(m.workspace), ts: str(m.ts), label: str(m.label), count: num(m.count) })
    if (matches.length >= 10) break
  }
  return { generated_at: str(o.generated_at) ?? undefined, available: o.available === false ? false : undefined, query: str(o.query), matches }
}
