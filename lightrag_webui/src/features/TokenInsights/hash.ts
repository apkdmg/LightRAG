import type { Scope } from './context'
import { PURPOSES, Purpose } from './meta'

/**
 * The Token Insights view in the URL hash, so an alert email, a colleague or
 * the Operations console can open it directly:
 *
 *   #/?insights=1&itab=explore&iws=<workspace>&idoc=<doc_id>
 *
 * `idoc` (spec FR-54) pre-filters Explore to one document's LLM calls; the
 * Operations Document drawer links here with it. Anything unknown or
 * malformed is dropped rather than trusted.
 */

export const INSIGHTS_TABS = ['overview', 'workspaces', 'explore', 'anomalies', 'work', 'forecast'] as const

const MAX_ID = 256

/** Workspace names and document ids: printable, no whitespace, bounded. */
export const cleanInsightsId = (raw: string | null | undefined): string | null => {
  if (!raw) return null
  const v = raw.trim()
  if (!v || v.length > MAX_ID) return null
  for (let i = 0; i < v.length; i += 1) {
    const c = v.charCodeAt(i)
    if (c <= 0x20 || c === 0x7f) return null
  }
  return /\s/.test(v) ? null : v
}

export type InsightsHash = {
  scope: Partial<Scope>
  /** A document whose calls Explore opens on (`idoc`); null when absent or malformed. */
  doc: string | null
}

export function parseInsightsHash(hash: string): InsightsHash {
  const query = hash.split('?')[1]
  if (!query) return { scope: {}, doc: null }
  const q = new URLSearchParams(query)
  if (q.get('insights') !== '1') return { scope: {}, doc: null }
  const out: Partial<Scope> = {}
  const tab = q.get('itab')
  if (tab && (INSIGHTS_TABS as readonly string[]).includes(tab)) out.tab = tab
  const purpose = q.get('ipurpose')
  if (purpose && (PURPOSES as readonly string[]).includes(purpose)) out.purpose = purpose as Purpose
  const ws = cleanInsightsId(q.get('iws'))
  if (ws) out.workspace = ws
  if (q.get('iday')) out.day = q.get('iday')
  const range = q.get('irange')
  if (range === 'custom' && q.get('istart') && q.get('iend')) {
    out.range = 'custom'
    out.start = q.get('istart') as string
    out.end = q.get('iend') as string
  } else if (range && !Number.isNaN(Number(range))) {
    out.range = Number(range)
  }
  const doc = cleanInsightsId(q.get('idoc'))
  // A document's calls live in Explore: idoc without a tab opens Explore.
  if (doc && !out.tab) out.tab = 'explore'
  return { scope: out, doc }
}

export function writeInsightsHash(scope: Scope, doc?: string | null): string {
  const q = new URLSearchParams({ insights: '1', itab: scope.tab, irange: String(scope.range) })
  if (scope.purpose) q.set('ipurpose', scope.purpose)
  if (scope.workspace) q.set('iws', scope.workspace)
  if (scope.day) q.set('iday', scope.day)
  if (scope.range === 'custom') {
    q.set('istart', scope.start)
    q.set('iend', scope.end)
  }
  const d = cleanInsightsId(doc)
  if (d && scope.tab === 'explore') q.set('idoc', d)
  return `#/?${q.toString()}`
}

/** Where another page links into Insights. */
export type InsightsTarget = {
  tab?: 'explore' | 'work' | 'overview'
  workspace?: string | null
  docId?: string | null
  purpose?: string | null
}

/** The hash for a link from another page; a document or purpose opens Explore. */
export function insightsLink(target: InsightsTarget = {}): string {
  const doc = cleanInsightsId(target.docId)
  const ws = cleanInsightsId(target.workspace)
  const purpose = target.purpose && (PURPOSES as readonly string[]).includes(target.purpose) ? target.purpose : null
  const q = new URLSearchParams({ insights: '1', itab: target.tab ?? 'explore' })
  if (purpose) q.set('ipurpose', purpose)
  if (ws) q.set('iws', ws)
  if (doc) {
    q.set('itab', 'explore')
    q.set('idoc', doc)
  }
  return `#/?${q.toString()}`
}
