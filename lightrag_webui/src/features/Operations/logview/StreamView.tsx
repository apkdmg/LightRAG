import { UIEvent, useEffect, useLayoutEffect, useRef, useState } from 'react'
import { useTranslation } from 'react-i18next'
import { ArrowUpIcon } from 'lucide-react'
import { cn } from '@/lib/utils'
import { LogPage, opsApi } from '../api'
import ErrorPanel from '../components/ErrorPanel'
import { LogRows } from '../components/LogRow'
import { StaleChip, Unavailable } from '../components/Panel'
import RefreshHairline from '../components/RefreshHairline'
import Skeleton from '../components/Skeleton'
import { OpsError, isAbort, toOpsError } from '../errors'
import { fmtClock, fmtCount } from '../format'
import { useStale } from '../hooks'
import {
  EMPTY_STREAM,
  FOLLOW_SECONDS,
  LOG_CAP,
  LOG_PAGE,
  LOG_PAGE_MAX,
  LogFilter,
  LogStream,
  anchoredScrollTop,
  applyFollow,
  applyHead,
  applyOlder,
  beforeParam,
  canLoadOlder,
  effectiveWindow,
  filterKey,
  followSince,
  isAtNewest,
  logsQuery,
  newestTs,
  textSearchWindow
} from '../logs'
import { useOpsShell } from '../context'
import { useOpsMotion } from '../motion'
import { useOpsPoll } from '../poll'

type Batch = { kind: 'head' | 'follow'; key: string; page: LogPage; limit: number }

type View = {
  stream: LogStream
  /** The poll answer the stream last took in (a new object means new data). */
  source: Batch | null
  /** How the rows last changed: decides the scroll correction. */
  change: 'head' | 'follow' | 'older' | 'none'
}

const EMPTY_VIEW: View = { stream: EMPTY_STREAM, source: null, change: 'none' }

/**
 * The stream view of the Logs tab (§2.5, §2.12, FR-21, FR-22). Newest first,
 * capped at 1000 DOM rows, older rows paged by the (ts, id) keyset.
 *
 * Follow polls every 5 s with `after_ts` = newest − 10 s and merges by id, so
 * rows that commit late are not skipped and the overlap never duplicates.
 * At the newest end a batch goes in with no list movement; new rows fade in
 * and carry a marker until the next batch. Scrolled away, the rows on screen
 * stay put (one scrollHeight measure, one scrollTop correction; Safari has no
 * overflow-anchor) and the "↑ N new" pill counts what arrived. Follow off,
 * "⏸ auto", a hidden tab and reduced motion all stop the polling or the fade.
 */
export default function StreamView({
  filter,
  follow,
  onCapture
}: {
  filter: LogFilter
  follow: boolean
  /** "Capture INFO for 60 min…" in the empty state (§2.5), for operators while no capture runs. */
  onCapture?: () => void
}) {
  const { t } = useTranslation()
  const { paused } = useOpsMotion()
  const textWindow = textSearchWindow(useOpsShell().status?.features)
  // The window a text search may cover is part of the query: a change (the index became valid) starts over.
  const key = `${filterKey(filter)}|${textWindow}`
  const [view, setView] = useState<View>(EMPTY_VIEW)
  const [atNewest, setAtNewest] = useState(true)
  const [older, setOlder] = useState<{ key: string; loading: boolean; error: OpsError | null }>({ key, loading: false, error: null })
  const scrollRef = useRef<HTMLDivElement | null>(null)
  const snap = useRef({ scrollTop: 0, scrollHeight: 0 })

  const current = view.stream.key === key ? view.stream : null
  const poll = useOpsPoll<Batch>(
    async (signal) => {
      // Follow off: a timer armed before the switch must not add rows (Follow is the pause control).
      if (!follow && current?.items.length && view.source) return view.source
      // With rows for this filter, ask only for what is newer; otherwise the newest page.
      const since = current?.items.length ? followSince(newestTs(current.items)) : null
      if (since) {
        const page = await opsApi.logs({ ...logsQuery(filter, Date.now(), LOG_PAGE_MAX, textWindow), after_ts: since }, signal)
        return { kind: 'follow', key, page, limit: LOG_PAGE_MAX }
      }
      return { kind: 'head', key, page: await opsApi.logs(logsQuery(filter, Date.now(), LOG_PAGE, textWindow), signal), limit: LOG_PAGE }
    },
    [key],
    follow ? FOLLOW_SECONDS : undefined
  )

  // Follow switched on: poll now (the interval only re-arms after a request).
  const reload = poll.reload
  const wasFollowing = useRef(follow)
  useEffect(() => {
    if (follow && !wasFollowing.current) reload()
    wasFollowing.current = follow
  }, [follow, reload])

  // Take in a new poll answer once, during render (no extra commit).
  const batch = poll.data
  if (batch && batch !== view.source && batch.key === key) {
    const s = view.stream
    // A new filter starts over, and so does a full follow page (it may hide a gap). The first
    // rows into an empty list are merged like a follow batch, so they are marked as new.
    const restart = s.key !== key || batch.page.items.length >= batch.limit || (batch.kind === 'head' && s.items.length > 0)
    const next = restart ? applyHead(key, batch.page) : applyFollow(s, batch.page, { baseline: poll.baseline, atNewest, limit: batch.limit })
    setView({ stream: next, source: batch, change: next === s ? 'none' : restart ? 'head' : 'follow' })
    // A new list starts at the top.
    if (restart && !atNewest) setAtNewest(true)
  }

  // After rows went in: keep the reader's place, or go to the top for a new filter.
  useLayoutEffect(() => {
    const el = scrollRef.current
    if (!el) return
    if (view.change === 'head') el.scrollTop = 0
    else if (view.change === 'follow' && !isAtNewest(snap.current.scrollTop)) {
      const top = anchoredScrollTop(snap.current, el.scrollHeight)
      if (top !== el.scrollTop) el.scrollTop = top
    }
    snap.current = { scrollTop: el.scrollTop, scrollHeight: el.scrollHeight }
  }, [view])

  const onScroll = (e: UIEvent<HTMLDivElement>) => {
    const el = e.currentTarget
    snap.current = { scrollTop: el.scrollTop, scrollHeight: el.scrollHeight }
    const at = isAtNewest(el.scrollTop)
    if (at !== atNewest) setAtNewest(at)
    if (at && view.stream.pending) setView((v) => ({ ...v, stream: { ...v.stream, pending: 0 }, change: 'none' }))
  }

  const toNewest = () => {
    const el = scrollRef.current
    if (el) el.scrollTop = 0
    setAtNewest(true)
    setView((v) => ({ ...v, stream: { ...v.stream, pending: 0 }, change: 'none' }))
  }

  const olderState = older.key === key ? older : { key, loading: false, error: null }
  const loadOlder = async () => {
    const s = current
    if (!s || !canLoadOlder(s) || olderState.loading) return
    setOlder({ key, loading: true, error: null })
    try {
      const page = await opsApi.logs({ ...logsQuery(filter, Date.now(), LOG_PAGE, textWindow), before: beforeParam(s.older) })
      setView((v) => (v.stream.key === key ? { ...v, stream: applyOlder(v.stream, page), change: 'older' } : v))
      setOlder((o) => (o.key === key ? { key, loading: false, error: null } : o))
    } catch (err) {
      if (isAbort(err)) return
      setOlder((o) => (o.key === key ? { key, loading: false, error: toOpsError(err) } : o))
    }
  }

  const { stale, ageSeconds } = useStale(follow ? poll.updatedAt : null, FOLLOW_SECONDS)
  const items = current?.items ?? []
  const showStale = follow && !paused && !!current && (stale || !!poll.error)
  const win = effectiveWindow(filter, textWindow)

  if (!current) {
    if (poll.error) return <ErrorPanel error={poll.error} onRetry={poll.reload} />
    return (
      <div className="flex flex-col gap-1.5" aria-busy="true">
        <Skeleton className="h-8 w-full" />
        <Skeleton className="h-8 w-full" />
        <Skeleton className="h-8 w-full" />
      </div>
    )
  }
  if (current.unavailable) return <Unavailable />

  let modeText: string | null = null
  if (follow) modeText = paused ? t('operations.logs.followPaused') : t('operations.logs.following', { seconds: FOLLOW_SECONDS })
  else if (poll.updatedAt) modeText = t('operations.logs.snapshot', { time: fmtClock(poll.updatedAt) })

  return (
    <div className={cn('relative flex flex-col gap-2', showStale && 'ops-stale')} aria-busy={poll.slow || undefined}>
      <RefreshHairline slow={poll.slow} />
      {poll.error ? <ErrorPanel error={poll.error} onRetry={poll.reload} compact /> : null}
      <div className="flex flex-wrap items-center justify-between gap-2 text-xs">
        <span className="text-muted-foreground">{modeText}</span>
        {showStale ? <StaleChip ageSeconds={ageSeconds} /> : null}
      </div>
      <div className="relative">
        {/* The pill is the second aria-live region of the page (§2.2): it says what motion only hints at. */}
        <div aria-live="polite" className="pointer-events-none absolute inset-x-0 top-2 z-10 flex justify-center">
          {current.pending > 0 ? (
            <button
              type="button"
              onClick={toNewest}
              className="ops-fade-in bg-primary text-primary-foreground pointer-events-auto inline-flex min-h-8 items-center gap-1.5 rounded-full px-3 text-[13px] shadow-md"
            >
              <ArrowUpIcon className="size-3.5" aria-hidden="true" />
              {t('operations.logs.newRows', { count: current.pending })}
            </button>
          ) : null}
        </div>
        <div
          ref={scrollRef}
          onScroll={onScroll}
          className="ops-log-scroll bg-card overflow-y-auto rounded-[10px] border"
          tabIndex={0}
          role="region"
          aria-label={t('operations.logs.listLabel')}
        >
          <div className="ops-log-head text-muted-foreground bg-card sticky top-0 z-[1] border-b text-xs" aria-hidden="true">
            <span>{t('operations.logs.col.time')}</span>
            <span>{t('operations.logs.col.level')}</span>
            <span>{t('operations.logs.col.pid')}</span>
            <span>{t('operations.logs.col.source')}</span>
            <span>{t('operations.logs.col.workspace')}</span>
            <span>{t('operations.logs.col.message')}</span>
          </div>
          {items.length ? (
            <LogRows items={items} fresh={current.fresh} />
          ) : (
            <div className="flex flex-col gap-1 p-4 text-sm">
              <p className="m-0">{t('operations.logs.empty', { level: filter.level })}</p>
              <p className="text-muted-foreground m-0 text-xs">{t('operations.logs.emptyHint')}</p>
              {onCapture && filter.level !== 'INFO' ? (
                <button type="button" className="hover:bg-muted mt-1 min-h-8 self-start rounded-lg border px-2.5 text-[13px]" onClick={onCapture}>
                  {t('operations.capture.emptyAction')}
                </button>
              ) : null}
            </div>
          )}
          {items.length ? (
            <div className="flex flex-col items-center gap-1.5 border-t p-2.5 text-xs">
              {olderState.error ? <ErrorPanel error={olderState.error} onRetry={loadOlder} compact /> : null}
              {canLoadOlder(current) ? (
                <button
                  type="button"
                  className="hover:bg-muted min-h-8 rounded-lg border px-3 text-[13px] disabled:opacity-50"
                  disabled={olderState.loading}
                  aria-busy={olderState.loading || undefined}
                  onClick={loadOlder}
                >
                  {olderState.loading ? t('operations.logs.loading') : t('operations.logs.loadOlder')}
                </button>
              ) : current.older && items.length >= LOG_CAP ? (
                <p className="text-muted-foreground m-0 text-center">{t('operations.logs.capped', { cap: fmtCount(LOG_CAP) })}</p>
              ) : (
                <p className="text-muted-foreground m-0">{t('operations.logs.end', { window: t(`operations.logs.window.${win}`) })}</p>
              )}
              <p className="text-muted-foreground m-0">{t('operations.logs.footer', { count: fmtCount(items.length) })}</p>
            </div>
          ) : null}
        </div>
      </div>
    </div>
  )
}
