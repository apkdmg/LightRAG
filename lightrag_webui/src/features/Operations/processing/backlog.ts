import type { Params } from '../api'

/**
 * Query state of the Backlog-by-workspace table (FR-12). The server pages,
 * searches and sorts in SQL; this module only turns the UI state into a
 * canonical query string, so equal views share one server cache entry and a
 * 3000-workspace host never ships more than one page.
 */

export const BACKLOG_SORTS = ['oldest', 'inflight', 'failed', 'name'] as const
export type BacklogSort = (typeof BACKLOG_SORTS)[number]

/** Filter chips, in the order they appear in the query string. */
export const BACKLOG_FILTERS = ['failed', 'stuck', 'email', 'paused'] as const
export type BacklogFilter = (typeof BACKLOG_FILTERS)[number]

export type SortDir = 'asc' | 'desc'

export const DEFAULT_PAGE_SIZE = 50
export const MAX_PAGE_SIZE = 200
/** The route's `q` limit: as long as the longest workspace ID. */
export const MAX_QUERY_LENGTH = 255

export type BacklogState = {
  q: string
  has: readonly BacklogFilter[]
  sort: BacklogSort
  dir: SortDir
  page: number
  pageSize: number
}

/**
 * The natural direction of each sort column: the oldest in-flight timestamp
 * first (ascending), the most in-flight or failed docs first, names A→Z.
 */
export const defaultDir = (sort: BacklogSort): SortDir => (sort === 'inflight' || sort === 'failed' ? 'desc' : 'asc')

export const DEFAULT_BACKLOG: BacklogState = { q: '', has: [], sort: 'oldest', dir: 'asc', page: 1, pageSize: DEFAULT_PAGE_SIZE }

const isSort = (v: unknown): v is BacklogSort => typeof v === 'string' && (BACKLOG_SORTS as readonly string[]).includes(v)

const clampInt = (v: unknown, min: number, max: number, fallback: number) => {
  const n = typeof v === 'number' ? v : Number(v)
  if (!Number.isFinite(n)) return fallback
  return Math.min(max, Math.max(min, Math.floor(n)))
}

/** Toggles one filter chip and keeps the canonical order. */
export function toggleFilter(has: readonly BacklogFilter[], filter: BacklogFilter): BacklogFilter[] {
  const set = new Set(has)
  if (set.has(filter)) set.delete(filter)
  else set.add(filter)
  return BACKLOG_FILTERS.filter((f) => set.has(f))
}

/** Clicking the active sort flips its direction; a new sort starts in its natural direction. */
export function nextSort(state: Pick<BacklogState, 'sort' | 'dir'>, sort: BacklogSort): { sort: BacklogSort; dir: SortDir } {
  if (state.sort === sort) return { sort, dir: state.dir === 'asc' ? 'desc' : 'asc' }
  return { sort, dir: defaultDir(sort) }
}

/**
 * Query parameters for GET /processing/workspaces. Unknown sorts fall back
 * to the default, unknown filters are dropped, the search is trimmed and
 * capped, the page size is capped at 200 and empty values are left out.
 */
export function backlogQuery(state: Partial<BacklogState>): Params {
  const sort = isSort(state.sort) ? state.sort : DEFAULT_BACKLOG.sort
  const dir: SortDir = state.dir === 'asc' || state.dir === 'desc' ? state.dir : defaultDir(sort)
  const has = BACKLOG_FILTERS.filter((f) => (state.has ?? []).includes(f))
  const q = (state.q ?? '').trim().slice(0, MAX_QUERY_LENGTH)
  const params: Params = {
    sort,
    dir,
    page: clampInt(state.page, 1, 1_000_000, 1),
    page_size: clampInt(state.pageSize, 1, MAX_PAGE_SIZE, DEFAULT_PAGE_SIZE)
  }
  if (q) params.q = q
  if (has.length) params.has = has.join(',')
  return params
}

/** The URL-encoded string the request will carry (also the cache key of a view). */
export function backlogQueryString(state: Partial<BacklogState>): string {
  const params = backlogQuery(state)
  const q = new URLSearchParams()
  for (const key of ['q', 'has', 'sort', 'dir', 'page', 'page_size']) {
    const v = params[key]
    if (v !== undefined && v !== null && v !== '') q.set(key, String(v))
  }
  return q.toString()
}

/** "1–50 of 1,312": the first and last row numbers of a page. */
export function pageRange(page: number, pageSize: number, total: number): { first: number; last: number; pages: number } {
  const size = Math.max(1, pageSize)
  const pages = Math.max(1, Math.ceil(Math.max(0, total) / size))
  const p = Math.min(Math.max(1, page), pages)
  if (total <= 0) return { first: 0, last: 0, pages }
  return { first: (p - 1) * size + 1, last: Math.min(total, p * size), pages }
}
