/**
 * The URL holds the Operations view, so an alert email, a colleague or a
 * support ticket can open the same tab, sub-view and drawer directly:
 *
 *   #/?ops=1&otab=processing&osub=stuck&ows=<workspace>
 *
 * Keys: ops=1 (marker), otab, osub, one drawer key (ows, orun, odoc=<ws>/<id>,
 * olog, odep, owk, oal) and olq (an encoded log filter). Anything unknown or
 * malformed is dropped rather than trusted.
 */

export const OPS_TABS = ['overview', 'processing', 'logs', 'health', 'performance', 'activity'] as const
export type OpsTab = (typeof OPS_TABS)[number]

/** Sub-views each tab understands; osub values outside these are ignored. */
export const OPS_SUBS: Record<OpsTab, readonly string[]> = {
  overview: [],
  processing: ['active', 'stuck', 'backlog', 'failures', 'email', 'pauses', 'runs'],
  logs: ['stream', 'grouped'],
  health: ['deps', 'signin', 'workers', 'host', 'config'],
  performance: ['llm', 'deps', 'http', 'throughput', 'concurrency', 'top'],
  activity: ['alerts', 'audit', 'events', 'export']
}

/** Drawer kinds in the order their keys are read; only one drawer is open at a time. */
export const DRAWER_KEYS = {
  workspace: 'ows',
  run: 'orun',
  document: 'odoc',
  log: 'olog',
  dependency: 'odep',
  worker: 'owk',
  alert: 'oal'
} as const
export type DrawerKind = keyof typeof DRAWER_KEYS

export type Drawer =
  | { kind: Exclude<DrawerKind, 'document'>; id: string }
  | { kind: 'document'; id: string; workspace: string }

export type OpsHash = {
  tab: OpsTab
  sub: string | null
  drawer: Drawer | null
  logQuery: string | null
}

export const DEFAULT_OPS_HASH: OpsHash = { tab: 'overview', sub: null, drawer: null, logQuery: null }

const MAX_ID = 256
const MAX_LOG_QUERY = 2000

// IDs are workspace names, UUIDs, doc ids, fingerprints: printable, no whitespace or control characters.
const isPrintableToken = (v: string) => {
  for (let i = 0; i < v.length; i += 1) {
    const c = v.charCodeAt(i)
    if (c <= 0x20 || c === 0x7f) return false
  }
  return !/\s/.test(v)
}

const cleanId = (raw: string | null): string | null => {
  if (!raw) return null
  const v = raw.trim()
  return v && v.length <= MAX_ID && isPrintableToken(v) ? v : null
}

const queryOf = (hash: string): URLSearchParams | null => {
  const at = hash.indexOf('?')
  if (at < 0) return null
  return new URLSearchParams(hash.slice(at + 1))
}

/** True when the hash asks for the Operations tab (the App-level deep link). */
export const hasOpsMarker = (hash: string) => queryOf(hash)?.get('ops') === '1'

export function readOpsHash(hash: string): OpsHash | null {
  const q = queryOf(hash)
  if (!q || q.get('ops') !== '1') return null
  const rawTab = q.get('otab')
  const tab: OpsTab = rawTab && (OPS_TABS as readonly string[]).includes(rawTab) ? (rawTab as OpsTab) : 'overview'
  const rawSub = q.get('osub')
  const sub = rawSub && OPS_SUBS[tab].includes(rawSub) ? rawSub : null

  let drawer: Drawer | null = null
  for (const [kind, key] of Object.entries(DRAWER_KEYS) as [DrawerKind, string][]) {
    const raw = q.get(key)
    if (!raw) continue
    if (kind === 'document') {
      const slash = raw.indexOf('/')
      const workspace = slash > 0 ? cleanId(raw.slice(0, slash)) : null
      const id = slash > 0 ? cleanId(raw.slice(slash + 1)) : null
      if (workspace && id) drawer = { kind, workspace, id }
    } else {
      const id = cleanId(raw)
      if (id) drawer = { kind, id }
    }
    if (drawer) break
  }

  const rawLq = q.get('olq')
  const logQuery = rawLq && rawLq.length <= MAX_LOG_QUERY ? rawLq : null
  return { tab, sub, drawer, logQuery }
}

export function writeOpsHash(state: OpsHash): string {
  const q = new URLSearchParams({ ops: '1', otab: state.tab })
  if (state.sub && OPS_SUBS[state.tab].includes(state.sub)) q.set('osub', state.sub)
  const d = state.drawer
  if (d) {
    if (d.kind === 'document') q.set(DRAWER_KEYS.document, `${d.workspace}/${d.id}`)
    else q.set(DRAWER_KEYS[d.kind], d.id)
  }
  if (state.logQuery) q.set('olq', state.logQuery)
  return `#/?${q.toString()}`
}

/** A drill-down target from the server (attention items, KPI tiles). */
export type HashTarget = {
  tab?: string | null
  sub?: string | null
  workspace?: string | null
  doc_id?: string | null
  run_id?: string | null
  /** A dependency id (opens the Dependency drawer). */
  dep?: string | null
  /** A worker id (opens the Worker drawer). */
  worker_id?: string | null
}

/**
 * The view a target opens, starting from the current one: a valid tab and
 * sub-view switch the page; a doc (with its workspace), a run, a workspace,
 * a dependency or a worker opens its drawer, in that order of precedence (a
 * hung-run alert names both its run and its workspace, and the run is the
 * more specific). Anything malformed is ignored, never trusted.
 */
export function viewForTarget(base: OpsHash, target: HashTarget | null | undefined): OpsHash {
  if (!target) return base
  const tab: OpsTab = target.tab && (OPS_TABS as readonly string[]).includes(target.tab) ? (target.tab as OpsTab) : base.tab
  const sub = target.sub && OPS_SUBS[tab].includes(target.sub) ? target.sub : tab === base.tab ? base.sub : null
  const workspace = cleanId(target.workspace ?? null)
  const docId = cleanId(target.doc_id ?? null)
  const run = cleanId(target.run_id ?? null)
  const dep = cleanId(target.dep ?? null)
  const worker = cleanId(target.worker_id ?? null)
  let drawer: Drawer | null = tab === base.tab ? base.drawer : null
  if (workspace && docId) drawer = { kind: 'document', workspace, id: docId }
  else if (run) drawer = { kind: 'run', id: run }
  else if (workspace) drawer = { kind: 'workspace', id: workspace }
  else if (dep) drawer = { kind: 'dependency', id: dep }
  else if (worker) drawer = { kind: 'worker', id: worker }
  return { tab, sub, drawer, logQuery: null }
}

/** Two drawers are the same entity (the URL can name only one). */
export const sameDrawer = (a: Drawer | null, b: Drawer | null) =>
  a === b || (!!a && !!b && a.kind === b.kind && a.id === b.id && (a.kind !== 'document' || (b.kind === 'document' && a.workspace === b.workspace)))
