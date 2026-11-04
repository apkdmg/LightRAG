import type { LogTail, TailLine } from '../api'
import { levelNumber, nextPending } from '../logs'

/**
 * The Live tail of the Logs tab (§3.8.6): the newest lines of every live
 * process's redacted JSONL file (OPS_JSONL_LOGS), INFO included, merged by
 * time on the server. The page polls; each answer is merged into what is
 * shown by line key, newest first, capped at TAIL_CAP rows. Pure, so the
 * merge, the cap and the pause rule are tested without a DOM.
 */

/** Rows kept on the page; older ones fall off the end. */
export const TAIL_CAP = 500
/** Lines asked for per poll (the server reads at most this many from the end of each file). */
export const TAIL_LINES = 200
/** Seconds between polls while following (the Logs follow cadence, §3.10). */
export const TAIL_SECONDS = 5
/** Levels the tail can be narrowed to; INFO is the point of it. */
export const TAIL_LEVELS = ['INFO', 'WARNING', 'ERROR'] as const
export type TailLevel = (typeof TAIL_LEVELS)[number]

type Loose = Record<string, any>

const obj = (v: unknown): Loose => (v && typeof v === 'object' && !Array.isArray(v) ? (v as Loose) : {})
const num = (v: unknown): number | null => (typeof v === 'number' && Number.isFinite(v) ? v : null)
const str = (v: unknown): string | null => (typeof v === 'string' && v ? v : null)
const arr = (v: unknown): unknown[] => (Array.isArray(v) ? v : [])

/** An ISO time from an ISO string or epoch seconds/ms; null when unreadable. */
function isoOf(raw: unknown): string | null {
  if (typeof raw === 'number' && Number.isFinite(raw)) return new Date(raw < 1e11 ? raw * 1000 : raw).toISOString()
  const s = str(raw)
  if (!s) return null
  return Number.isFinite(Date.parse(s)) ? s : null
}

const tsMs = (line: Pick<TailLine, 'ts'>) => {
  const t = Date.parse(line.ts)
  return Number.isFinite(t) ? t : 0
}

/** Small, stable string hash (FNV-1a) so a key does not carry the whole message. */
function hash(text: string): string {
  let h = 0x811c9dc5
  for (let i = 0; i < text.length; i += 1) {
    h ^= text.charCodeAt(i)
    h = Math.imul(h, 0x01000193)
  }
  return (h >>> 0).toString(36)
}

/**
 * One tail line, or null without a time (it could not be ordered). Its key
 * is the server's own (`key`, or the file and byte offset) when it sends
 * one, else time, process, call site and message: the same line read by two
 * polls gets the same key.
 */
export function normalizeTailLine(raw: unknown): TailLine | null {
  const o = obj(raw)
  const ts = isoOf(o.ts ?? o.time ?? o.timestamp)
  if (!ts) return null
  const msg = typeof o.msg === 'string' ? o.msg : typeof o.message === 'string' ? o.message : ''
  const pid = num(o.pid)
  const lineno = num(o.lineno)
  const logger = str(o.logger) ?? str(o.name) ?? ''
  const offset = num(o.offset) ?? num(o.pos)
  const serverKey = str(o.key) ?? (offset !== null ? `${pid ?? str(o.file) ?? '-'}@${offset}` : null)
  return {
    key: serverKey ?? `${ts}|${pid ?? '-'}|${logger}:${lineno ?? '-'}|${hash(msg)}`,
    ts,
    // A level number of 0 means the file named it only by name.
    level: levelNumber(typeof o.level === 'number' && o.level > 0 ? o.level : (o.level_name ?? o.levelname ?? o.level)),
    logger,
    module: str(o.module),
    lineno,
    pid,
    workspace: str(o.workspace),
    request_id: str(o.request_id),
    run_id: str(o.run_id),
    msg,
    exc_type: str(o.exc_type)
  }
}

/**
 * GET /logs/tail: `{enabled, items[], sources[], sources_capped}`. `sources`
 * lists the process files read (each says whether its read was cut short);
 * any spelling the route may use is accepted and nothing throws.
 */
export function normalizeLogTail(raw: unknown): LogTail {
  const o = obj(raw)
  const items: TailLine[] = []
  for (const r of arr(o.items ?? o.entries ?? (Array.isArray(o.lines) ? o.lines : null))) {
    const line = normalizeTailLine(r)
    if (line) items.push(line)
  }
  const sources = Array.isArray(o.sources) ? o.sources : Array.isArray(o.files) ? o.files : null
  const files = sources ? sources.length : num(o.files)
  const cut = !!sources && sources.some((x) => obj(x).truncated === true)
  return {
    generated_at: str(o.generated_at) ?? undefined,
    available: o.available === false ? false : undefined,
    enabled: typeof o.enabled === 'boolean' ? o.enabled : null,
    items,
    files,
    truncated: o.truncated === true || o.sources_capped === true || cut
  }
}

/** Newest first; equal times keep the order they arrived in (the server's merge order). */
export function sortTail(lines: TailLine[]): TailLine[] {
  return lines
    .map((line, i) => ({ line, i, t: tsMs(line) }))
    .sort((a, b) => b.t - a.t || a.i - b.i)
    .map((x) => x.line)
}

export type TailState = {
  /** The filter (level and module) the rows belong to; another filter starts over. */
  key: string
  items: TailLine[]
  /** Keys of the last batch's new rows: they carry the marker until the next batch (§2.12). */
  fresh: ReadonlySet<string>
  /** Rows that arrived while the reader was scrolled away from the newest end. */
  pending: number
  /** The server's answer: false when per-process JSONL files are off (OPS_JSONL_LOGS). */
  enabled: boolean | null
  /** The server stopped reading before it had every file's lines. */
  truncated: boolean
  files: number | null
}

const NO_FRESH: ReadonlySet<string> = new Set()

export const EMPTY_TAIL: TailState = { key: '\u0000', items: [], fresh: NO_FRESH, pending: 0, enabled: null, truncated: false, files: null }

export type TailCtx = {
  /** First answer, resume or a visible tab again: rows go in unmarked (§2.12 rule 7). */
  baseline: boolean
  /** The list is at its newest end: no "↑ N new" pill. */
  atNewest: boolean
  cap?: number
}

/**
 * Merges one tail answer into the rows shown. Lines already shown are
 * skipped by key, so overlapping polls never duplicate; the rest go in at
 * their time and the oldest fall off past the cap. Nothing new keeps the
 * same state object, so React does no work.
 */
export function mergeTail(state: TailState, key: string, tail: LogTail, ctx: TailCtx): TailState {
  const cap = ctx.cap ?? TAIL_CAP
  const restart = state.key !== key
  const base = restart ? { ...EMPTY_TAIL, key } : state
  const meta = { enabled: tail.enabled, truncated: tail.truncated, files: tail.files }
  const known = new Set(base.items.map((l) => l.key))
  const added: TailLine[] = []
  for (const line of tail.items) {
    if (known.has(line.key)) continue
    known.add(line.key)
    added.push(line)
  }
  const metaSame = base.enabled === meta.enabled && base.truncated === meta.truncated && base.files === meta.files
  if (!added.length) return metaSame && !restart ? state : { ...base, ...meta, fresh: NO_FRESH }
  const all = sortTail([...added, ...base.items])
  const items = all.slice(0, cap)
  const kept = new Set(items.map((l) => l.key))
  const shown = added.filter((l) => kept.has(l.key))
  const first = restart || base.items.length === 0
  return {
    key,
    items,
    fresh: ctx.baseline || first ? NO_FRESH : new Set(shown.map((l) => l.key)),
    pending: first ? 0 : nextPending(base.pending, shown.length, ctx.atNewest),
    ...meta
  }
}

/**
 * Whether a poll answer goes in. While paused nothing new is shown, even an
 * answer already on its way; the first answer for a new level or module is
 * the exception, so a paused tail still shows what was asked for.
 */
export function takesBatch(following: boolean, shownKey: string, batchKey: string, currentKey: string): boolean {
  if (batchKey !== currentKey) return false
  return following || shownKey !== currentKey
}

/** The filter key the rows are kept under. */
export const tailKey = (level: string, module: string) => `${level}|${module}`

/** The tail's query: lines per file, the lowest level, and a module when one is chosen. */
export function tailQuery(level: string, module: string, lines = TAIL_LINES) {
  return { lines: Math.max(1, Math.min(1000, Math.floor(lines))), level, module: module || undefined }
}

/** Live tail is offered when the server says it writes the JSONL files; a server one release behind says nothing. */
export const tailFeature = (features: Record<string, unknown> | null | undefined): boolean | null => {
  const f = features ?? {}
  const v = f.log_tail ?? f.jsonl_logs
  return typeof v === 'boolean' ? v : null
}
