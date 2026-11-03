import type { CorrelateMatch, LogCursor, LogEntry, LogPage, Params } from './api'
import type { OpsHash } from './hashState'

/**
 * The Logs tab's pure parts (spec §2.5, §3.10 logs.ts, FR-20 to FR-22): the
 * filter and its URL form (`olq`), the query it sends, the (ts, id) keyset
 * order, the follow-mode merge with de-duplication by id, the 1000-row DOM
 * cap and the "N new" pill count. Everything here is covered by logs.test.ts.
 */

/** Minimum levels the filter offers. INFO is stored only while an INFO capture runs (PR-8). */
export const LOG_LEVELS = ['WARNING', 'ERROR', 'CRITICAL', 'INFO'] as const
export type LogLevel = (typeof LOG_LEVELS)[number]

/** Query windows; at most 7 days per query, and at most 24 h with a text search (FR-20). */
export const LOG_WINDOWS = ['15m', '1h', '6h', '24h', '7d'] as const
export type LogWindow = (typeof LOG_WINDOWS)[number]

export const WINDOW_SECONDS: Record<LogWindow, number> = {
  '15m': 15 * 60,
  '1h': 3600,
  '6h': 6 * 3600,
  '24h': 24 * 3600,
  '7d': 7 * 24 * 3600
}

/** Text search runs over at most 24 h until pg_trgm is available (FR-20, V7). */
export const TEXT_SEARCH_WINDOW: LogWindow = '24h'

/** The DOM holds at most this many rows; older ones are paged with "Load older" (§2.5). */
export const LOG_CAP = 1000
/** Rows per request: 100 by default, 500 at most (FR-21). */
export const LOG_PAGE = 100
export const LOG_PAGE_MAX = 500
/** Follow re-queries from the newest row's time minus this overlap, so late commits are not skipped (FR-22). */
export const FOLLOW_OVERLAP_MS = 10_000
/** Follow polls every 5 s while the tab is visible (§3.10). */
export const FOLLOW_SECONDS = 5

const MAX_FIELD = 256
const MAX_TEXT = 200

/** The ID filters a row's context line and the entry drawer pivot on. */
export const ID_FIELDS = ['request_id', 'run_id', 'track_id', 'doc_id', 'fingerprint'] as const
export type IdField = (typeof ID_FIELDS)[number]

export type LogFilter = {
  level: LogLevel
  window: LogWindow
  workspace: string
  /** record.module, or a logger name for records from outside `lightrag` (§2.5). */
  module: string
  q: string
  request_id: string
  run_id: string
  track_id: string
  doc_id: string
  fingerprint: string
  /** The Worker drawer's section: rows from one process. */
  worker_id: string
}

export const DEFAULT_LOG_FILTER: LogFilter = {
  level: 'WARNING',
  window: '1h',
  workspace: '',
  module: '',
  q: '',
  request_id: '',
  run_id: '',
  track_id: '',
  doc_id: '',
  fingerprint: '',
  worker_id: ''
}

/** Short URL keys for `olq` (the whole filter stays well under the 2000-character cap). */
const URL_KEYS: Record<keyof LogFilter, string> = {
  level: 'lv',
  window: 'w',
  workspace: 'ws',
  module: 'mod',
  q: 'q',
  request_id: 'req',
  run_id: 'run',
  track_id: 'trk',
  doc_id: 'doc',
  fingerprint: 'fp',
  worker_id: 'wk'
}

const TEXT_FIELDS = ['workspace', 'module', 'q', ...ID_FIELDS, 'worker_id'] as const

// Printable, no control characters; IDs and names never need them.
const cleanText = (raw: string | null | undefined, max = MAX_FIELD): string => {
  if (!raw) return ''
  let out = ''
  for (const ch of raw) {
    const c = ch.charCodeAt(0)
    out += c < 0x20 || c === 0x7f ? ' ' : ch
  }
  out = out.trim()
  return out.length > max ? out.slice(0, max) : out
}

const isLevel = (v: unknown): v is LogLevel => typeof v === 'string' && (LOG_LEVELS as readonly string[]).includes(v)
const isWindow = (v: unknown): v is LogWindow => typeof v === 'string' && (LOG_WINDOWS as readonly string[]).includes(v)

/** A filter with every field present and cleaned; anything malformed becomes the default. */
export function normalizeFilter(patch: Partial<LogFilter> | null | undefined): LogFilter {
  const p = patch ?? {}
  const out: LogFilter = { ...DEFAULT_LOG_FILTER }
  out.level = isLevel(p.level) ? p.level : DEFAULT_LOG_FILTER.level
  out.window = isWindow(p.window) ? p.window : DEFAULT_LOG_FILTER.window
  for (const k of TEXT_FIELDS) out[k] = cleanText(p[k], k === 'q' ? MAX_TEXT : MAX_FIELD)
  return out
}

/** The `olq` value for a filter; null when it is the default (the URL stays short). */
export function encodeLogFilter(filter: Partial<LogFilter>): string | null {
  const f = normalizeFilter(filter)
  const q = new URLSearchParams()
  for (const k of Object.keys(URL_KEYS) as (keyof LogFilter)[]) {
    if (f[k] && f[k] !== DEFAULT_LOG_FILTER[k]) q.set(URL_KEYS[k], f[k])
  }
  const s = q.toString()
  return s || null
}

/** Reads `olq` back; unknown keys and bad values are dropped rather than trusted. */
export function decodeLogFilter(raw: string | null | undefined): LogFilter {
  if (!raw) return { ...DEFAULT_LOG_FILTER }
  let q: URLSearchParams
  try {
    q = new URLSearchParams(raw)
  } catch {
    return { ...DEFAULT_LOG_FILTER }
  }
  const patch: Partial<LogFilter> = {}
  for (const [k, key] of Object.entries(URL_KEYS) as [keyof LogFilter, string][]) {
    const v = q.get(key)
    if (v !== null) (patch as Record<string, string>)[k] = v
  }
  return normalizeFilter(patch)
}

/** The window a query really covers: a text search is capped at 24 h. */
export function effectiveWindow(filter: Pick<LogFilter, 'window' | 'q'>): LogWindow {
  if (filter.q && WINDOW_SECONDS[filter.window] > WINDOW_SECONDS[TEXT_SEARCH_WINDOW]) return TEXT_SEARCH_WINDOW
  return filter.window
}

/** Filters the user set beyond level and window (the "Active:" chips and the "Filters (n)" count). */
export function activeFields(filter: LogFilter): (keyof LogFilter)[] {
  return TEXT_FIELDS.filter((k) => !!filter[k])
}

/** Same filters, different text: the query to the server is what matters for caching and paging. */
export const filterKey = (filter: LogFilter) => encodeLogFilter(filter) ?? ''

/** A timestamp minus a number of milliseconds, as ISO; null when unreadable. */
const isoMinus = (at: string | number, ms: number): string | null => {
  const t = typeof at === 'number' ? at : Date.parse(at)
  return Number.isFinite(t) ? new Date(t - ms).toISOString() : null
}

/**
 * The /logs query for a filter (§3.7). `from` is now minus the window; the
 * server checks the window again. A minute bucket keeps the query stable
 * across renders, so polling compares like with like.
 */
export function logsQuery(filter: LogFilter, now: number, limit = LOG_PAGE): Params {
  const window = effectiveWindow(filter)
  const minute = Math.floor(now / 60_000) * 60_000
  return {
    level: filter.level,
    from: isoMinus(minute, WINDOW_SECONDS[window] * 1000),
    workspace: filter.workspace || undefined,
    module: filter.module || undefined,
    q: filter.q || undefined,
    request_id: filter.request_id || undefined,
    run_id: filter.run_id || undefined,
    track_id: filter.track_id || undefined,
    doc_id: filter.doc_id || undefined,
    fingerprint: filter.fingerprint || undefined,
    worker_id: filter.worker_id || undefined,
    limit: Math.max(1, Math.min(LOG_PAGE_MAX, Math.floor(limit)))
  }
}

/** The filters an export carries (the server applies its own 10k cap). */
export function exportFilters(filter: LogFilter, now: number): Record<string, string> {
  const { limit: _limit, ...rest } = logsQuery(filter, now)
  void _limit
  return Object.fromEntries(Object.entries(rest).filter(([, v]) => v !== undefined && v !== null && v !== '')) as Record<string, string>
}

/** The "before" keyset parameter: "ts,id" (the id breaks ties between equal timestamps). */
export const beforeParam = (cursor: LogCursor | null | undefined) => (cursor ? `${cursor.ts},${cursor.id}` : undefined)

/** `attachment; filename="ops-logs-….ndjson"` → the name, without any path. */
export function filenameFromDisposition(disposition: string | null, fallback: string): string {
  if (!disposition) return fallback
  const star = /filename\*=(?:UTF-8'')?([^;]+)/i.exec(disposition)
  const plain = /filename="?([^";]+)"?/i.exec(disposition)
  let name = star ? star[1] : plain ? plain[1] : ''
  try {
    name = decodeURIComponent(name.trim())
  } catch {
    name = name.trim()
  }
  name = name.replace(/^.*[\\/]/, '').replace(/[^\w.-]+/g, '_')
  return name || fallback
}

// ---------- keyset order ----------

const tsOf = (e: Pick<LogEntry, 'ts'>) => {
  const t = Date.parse(e.ts)
  return Number.isFinite(t) ? t : 0
}

/** Newest first by (ts, id): rows with the same timestamp keep a stable order by id. */
export const compareNewestFirst = (a: Pick<LogEntry, 'ts' | 'id'>, b: Pick<LogEntry, 'ts' | 'id'>) => tsOf(b) - tsOf(a) || b.id - a.id

export const cursorOf = (e: Pick<LogEntry, 'ts' | 'id'>): LogCursor => ({ ts: e.ts, id: e.id })

/** The newest timestamp among rows (follow starts from it). */
export function newestTs(items: Pick<LogEntry, 'ts' | 'id'>[]): string | null {
  let best: Pick<LogEntry, 'ts' | 'id'> | null = null
  for (const e of items) if (!best || compareNewestFirst(e, best) < 0) best = e
  return best ? best.ts : null
}

/** Where a follow poll starts: the newest row's time minus the overlap; null for a fresh head query. */
export function followSince(newest: string | null): string | null {
  return newest ? isoMinus(newest, FOLLOW_OVERLAP_MS) : null
}

/** Rows in (ts, id) order without duplicates: the first copy of an id wins. */
function sortedUnique(rows: LogEntry[]): LogEntry[] {
  const seen = new Set<number>()
  const out: LogEntry[] = []
  for (const r of rows) {
    if (seen.has(r.id)) continue
    seen.add(r.id)
    out.push(r)
  }
  return out.sort(compareNewestFirst)
}

// ---------- the stream state ----------

export type LogStream = {
  /** The filter the rows belong to; another filter starts over. */
  key: string
  items: LogEntry[]
  /** Ids of the last follow batch: they carry the 2 px marker until the next batch (§2.12). */
  fresh: ReadonlySet<number>
  /** The cursor for "Load older"; null once the window is exhausted. */
  older: LogCursor | null
  /** Rows added while the user was scrolled away from the newest end ("↑ N new"). */
  pending: number
  /** The server answered `available: false` (no PostgreSQL). */
  unavailable: boolean
}

/** No rows yet. Its key matches no filter (the default filter's key is ''), so nothing reads it as loaded. */
export const EMPTY_STREAM: LogStream = { key: '\u0000', items: [], fresh: new Set(), older: null, pending: 0, unavailable: false }

const NO_FRESH: ReadonlySet<number> = new Set()

/** The newest page for a filter: a fresh list, nothing marked (a baseline, §2.12 rule 7). */
export function applyHead(key: string, page: LogPage, cap = LOG_CAP): LogStream {
  const all = sortedUnique(page.items)
  const items = all.slice(0, cap)
  const trimmed = all.length > items.length
  return {
    key,
    items,
    fresh: NO_FRESH,
    older: trimmed ? cursorOf(items[items.length - 1]) : page.next_before,
    pending: 0,
    unavailable: page.available === false
  }
}

export type FollowCtx = {
  /** The poll was a baseline (resume, visible again, error cleared): rows go in unmarked. */
  baseline: boolean
  /** The list is scrolled to the newest end: no pill. */
  atNewest: boolean
  /** The page size asked for: a full page may hide a gap, so it replaces the list. */
  limit: number
  cap?: number
}

/**
 * Merges one follow answer (rows with ts ≥ newest − 10 s) into the stream.
 * Rows already shown are skipped by id, so the overlap never duplicates; a
 * late commit with an older timestamp lands in its (ts, id) place. Newer rows
 * push the oldest out past the cap, and the "Load older" cursor moves to the
 * last row kept. A full page means more rows arrived than one poll returns:
 * the list restarts from that page rather than leave a silent gap.
 */
export function applyFollow(state: LogStream, page: LogPage, ctx: FollowCtx): LogStream {
  const cap = ctx.cap ?? LOG_CAP
  if (page.available === false) return { ...state, unavailable: true }
  if (page.items.length >= ctx.limit) return applyHead(state.key, page, cap)
  const known = new Set(state.items.map((r) => r.id))
  const added = page.items.filter((r) => !known.has(r.id))
  if (!added.length) return state.unavailable ? { ...state, unavailable: false } : state
  const all = sortedUnique([...added, ...state.items])
  const items = all.slice(0, cap)
  const kept = new Set(items.map((r) => r.id))
  const shown = added.filter((r) => kept.has(r.id)).map((r) => r.id)
  return {
    key: state.key,
    items,
    fresh: ctx.baseline ? NO_FRESH : new Set(shown),
    // An empty list takes the page's cursor (the first rows of a filter came in as a batch).
    older: all.length > items.length ? cursorOf(items[items.length - 1]) : state.items.length ? state.older : page.next_before,
    pending: nextPending(state.pending, shown.length, ctx.atNewest),
    unavailable: false
  }
}

/** Appends an older page ("Load older"); past the cap the list stops growing and keeps a cursor. */
export function applyOlder(state: LogStream, page: LogPage, cap = LOG_CAP): LogStream {
  const all = sortedUnique([...state.items, ...page.items])
  const items = all.slice(0, cap)
  return { ...state, items, older: all.length > items.length ? cursorOf(items[items.length - 1]) : page.next_before }
}

/** "Load older" is offered while the window has more rows and the DOM is under its cap. */
export const canLoadOlder = (state: LogStream, cap = LOG_CAP) => state.older !== null && state.items.length < cap

/** The pill count: rows that arrived while scrolled away; back at the newest end it clears. */
export function nextPending(pending: number, added: number, atNewest: boolean): number {
  return atNewest ? 0 : pending + Math.max(0, added)
}

/** The list is at its newest end when scrolled to within a few pixels of the top. */
export const AT_NEWEST_PX = 4
export const isAtNewest = (scrollTop: number) => scrollTop <= AT_NEWEST_PX

/**
 * The scrollTop that keeps the rows on screen where they were after rows
 * went in above them: one scrollHeight measure before, one after (Safari has
 * no overflow-anchor). At the newest end nothing moves.
 */
export function anchoredScrollTop(before: { scrollTop: number; scrollHeight: number }, afterScrollHeight: number): number {
  if (isAtNewest(before.scrollTop)) return before.scrollTop
  return Math.max(0, before.scrollTop + (afterScrollHeight - before.scrollHeight))
}

// ---------- levels ----------

/** Python logging levels by number. */
const LEVEL_NAMES: [number, string][] = [
  [50, 'CRITICAL'],
  [40, 'ERROR'],
  [30, 'WARNING'],
  [20, 'INFO'],
  [10, 'DEBUG']
]

export const levelNumber = (raw: unknown): number => {
  if (typeof raw === 'number' && Number.isFinite(raw)) return raw
  if (typeof raw === 'string') {
    const up = raw.trim().toUpperCase()
    if (up === 'WARN') return 30
    if (up === 'FATAL') return 50
    const hit = LEVEL_NAMES.find(([, n]) => n === up)
    if (hit) return hit[0]
    const n = Number(up)
    if (Number.isFinite(n)) return n
  }
  return 0
}

export const levelName = (n: number): string => LEVEL_NAMES.find(([v]) => n >= v)?.[1] ?? 'NOTSET'

/** A level's look in the status vocabulary (§2.2): icon and text, never colour alone. */
export function levelState(n: number): 'critical' | 'serious' | 'warning' | 'unknown' {
  if (n >= 50) return 'critical'
  if (n >= 40) return 'serious'
  if (n >= 30) return 'warning'
  return 'unknown'
}

/**
 * The module column (§2.5): `record.module` for lightrag's own records, the
 * logger name for everything else (uvicorn.error, httpx, asyncpg, …).
 */
export function sourceLabel(e: Pick<LogEntry, 'logger' | 'module' | 'lineno'>): string {
  const own = !e.logger || e.logger === 'lightrag' || e.logger.startsWith('lightrag.')
  const name = own ? e.module || e.logger || '–' : e.logger
  return name
}

/** "pipeline:2596": where the line was logged. */
export function callSite(e: Pick<LogEntry, 'logger' | 'module' | 'lineno'>): string {
  const name = e.module || e.logger || '–'
  return typeof e.lineno === 'number' ? `${name}:${e.lineno}` : name
}

/** Local "10:45:02.114" for a row (the tooltip carries UTC). */
export function fmtLogTime(at: string | null | undefined): string {
  if (!at) return '–'
  const d = new Date(at)
  if (Number.isNaN(d.getTime())) return '–'
  const pad = (v: number, n = 2) => String(v).padStart(n, '0')
  return `${pad(d.getHours())}:${pad(d.getMinutes())}:${pad(d.getSeconds())}.${pad(d.getMilliseconds(), 3)}`
}

// ---------- redacted spans ----------

/** Placeholders the server's redactor writes (ops/redact.py). */
export const REDACTION_TOKENS = ['email', 'ic', 'phone', 'redacted', 'jwt', 'hex', 'b64', 'num'] as const

export type TextPart = { kind: 'text'; text: string } | { kind: 'token'; token: string } | { kind: 'file'; text: string }

const TOKEN_RE = new RegExp(`<(${REDACTION_TOKENS.join('|')})>|[^\\s'"\`()<>\\[\\]{},;|=]*•••[^\\s'"\`()<>\\[\\]{},;|=]*`, 'g')

/**
 * Splits redacted text into plain runs, placeholders (`<email>`, `<ic>`, …)
 * and masked file names (`Sen•••er.pdf`), so the page can mark what was
 * masked on the server (§2.5). Pure string work: React escapes every part.
 */
export function splitRedacted(text: string | null | undefined): TextPart[] {
  if (!text) return []
  const parts: TextPart[] = []
  let last = 0
  for (const m of text.matchAll(TOKEN_RE)) {
    const at = m.index ?? 0
    if (!m[0]) continue
    if (at > last) parts.push({ kind: 'text', text: text.slice(last, at) })
    parts.push(m[1] ? { kind: 'token', token: m[1] } : { kind: 'file', text: m[0] })
    last = at + m[0].length
  }
  if (last < text.length) parts.push({ kind: 'text', text: text.slice(last) })
  return parts
}

/** The first line of a message, for the one-line row. */
export const firstLine = (text: string | null | undefined) => {
  if (!text) return ''
  const nl = text.indexOf('\n')
  return nl < 0 ? text : text.slice(0, nl)
}

/** Lines in a traceback, for "▸ traceback (34)". */
export const lineCount = (text: string | null | undefined) => (text ? text.split('\n').filter((l) => l.trim()).length : 0)

// ---------- links ----------

/** The Logs tab with a filter (pivots from drawers, rows and the correlate box). The drawer closes. */
export function logsView(filter: Partial<LogFilter>, sub: 'stream' | 'grouped' = 'stream'): OpsHash {
  return { tab: 'logs', sub, drawer: null, logQuery: encodeLogFilter(filter) }
}

/** A pivot on one ID: the whole 7-day window, since an ID is already narrow (FR-20). */
export const pivotFilter = (field: IdField | 'workspace' | 'worker_id', value: string): Partial<LogFilter> => ({ [field]: value, window: '7d' })

/**
 * Where a correlate match lands (FR-50, §2.9 flows 3 and 5): a run, a
 * document (with its workspace), a log entry and a workspace or owner open
 * their drawer over the current tab; a request or track ID opens the Logs
 * tab filtered on it. A doc without a known workspace is looked up in the
 * logs. Null for something the page cannot show.
 */
export function correlateView(base: OpsHash, match: CorrelateMatch): OpsHash | null {
  const keep = { ...base, logQuery: base.tab === 'logs' ? base.logQuery : null }
  switch (match.type) {
    case 'run':
      return { ...keep, drawer: { kind: 'run', id: match.id } }
    case 'log':
      return { ...keep, drawer: { kind: 'log', id: match.id } }
    case 'doc':
      return match.workspace ? { ...keep, drawer: { kind: 'document', workspace: match.workspace, id: match.id } } : logsView(pivotFilter('doc_id', match.id))
    case 'workspace':
      return { ...keep, drawer: { kind: 'workspace', id: match.workspace || match.id } }
    case 'owner':
      return match.workspace ? { ...keep, drawer: { kind: 'workspace', id: match.workspace } } : null
    case 'request':
      return logsView(pivotFilter('request_id', match.id))
    case 'track':
      return logsView(pivotFilter('track_id', match.id))
    default:
      return null
  }
}

/** What the correlate box accepts: one token of printable characters, at most 320 (an email). */
export const MAX_CORRELATE = 320
export function cleanCorrelateInput(raw: string): string | null {
  const v = raw.trim()
  if (!v || v.length > MAX_CORRELATE) return null
  for (const ch of v) {
    const c = ch.charCodeAt(0)
    if (c < 0x20 || c === 0x7f) return null
  }
  return /\s/.test(v) ? null : v
}

/**
 * Rows from one process: the Worker drawer asks the server for its worker_id,
 * and keeps only rows that carry it (or its pid), so a server that does not
 * filter on worker_id never shows other processes' lines as this one's.
 */
export function onlyWorker(items: LogEntry[], workerId: string, pid: number | null): LogEntry[] {
  return items.filter((e) => (e.worker_id ? e.worker_id === workerId : pid !== null && e.pid === pid))
}
