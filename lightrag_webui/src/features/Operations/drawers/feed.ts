import type { Feed } from '../api'

/** The drawer keeps the last 200 feed lines (the server sends at most 200 per call). */
export const FEED_CAP = 200

export type FeedState = {
  lines: string[]
  /** The offset to ask for next: every line before it has been shown. */
  next: number
  /** The last answer said the feed was wiped (a run started, a bundle was replaced, a doc deleted). */
  reset: boolean
}

export const EMPTY_FEED: FeedState = { lines: [], next: 0, reset: false }

/**
 * Merges one /feed answer into what the drawer shows. Without an offset the
 * server returns the newest lines (a fresh snapshot); with one it returns the
 * lines after it. When `history_messages` was wiped since the last call the
 * server says `reset` (or its total drops below our offset): the shown lines
 * are replaced instead of appended, so old and new runs never interleave.
 */
export function mergeFeed(prev: FeedState, resp: Feed, requestedOffset: number | null, cap = FEED_CAP): FeedState {
  const lines = Array.isArray(resp.lines) ? resp.lines.filter((l): l is string => typeof l === 'string') : []
  const total = typeof resp.total === 'number' && resp.total >= 0 ? resp.total : lines.length
  const wiped = !!resp.reset || (requestedOffset !== null && total < requestedOffset)
  if (requestedOffset === null || wiped) {
    const next = typeof resp.next_offset === 'number' && resp.next_offset >= 0 ? resp.next_offset : total
    return { lines: lines.slice(-cap), next, reset: wiped }
  }
  if (!lines.length) return prev.reset ? { ...prev, reset: false } : prev
  const next = typeof resp.next_offset === 'number' && resp.next_offset >= requestedOffset ? resp.next_offset : requestedOffset + lines.length
  return { lines: [...prev.lines, ...lines].slice(-cap), next, reset: false }
}
