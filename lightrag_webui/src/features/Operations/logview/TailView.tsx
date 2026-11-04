import { UIEvent, memo, useEffect, useLayoutEffect, useRef, useState } from 'react'
import { useTranslation } from 'react-i18next'
import { ArrowUpIcon, PauseIcon, PlayIcon } from 'lucide-react'
import { Seg } from '@/components/admin'
import { cn } from '@/lib/utils'
import { LogTail, TailLine, opsApi } from '../api'
import ErrorPanel from '../components/ErrorPanel'
import { LevelChip } from '../components/LogRow'
import OpsLink from '../components/OpsLink'
import { StaleChip, Unavailable } from '../components/Panel'
import { RedactedText } from '../components/RedactedSpan'
import RefreshHairline from '../components/RefreshHairline'
import Skeleton from '../components/Skeleton'
import { fmtClock, fmtCount, shortId, utcTitle } from '../format'
import { useStale } from '../hooks'
import { anchoredScrollTop, firstLine, fmtLogTime, isAtNewest, logsView, pivotFilter, sourceLabel } from '../logs'
import { useOpsMotion } from '../motion'
import { useOpsPoll } from '../poll'
import { EMPTY_TAIL, TAIL_CAP, TAIL_LEVELS, TAIL_SECONDS, TailLevel, TailState, mergeTail, tailKey, tailQuery, takesBatch } from './tail'

type Batch = { key: string; tail: LogTail }

const ctxLink = 'font-mono underline-offset-2 hover:underline'

/**
 * One tail line: the stream row's columns, without a drawer (a line read
 * from a file has no stored entry to open). Its request and run IDs pivot
 * like the stream's. A line from the latest batch carries the 2 px marker
 * and fades in over 150 ms; under reduced motion only the marker stays.
 */
const TailRow = memo(function TailRow({ line, fresh, reduced }: { line: TailLine; fresh: boolean; reduced: boolean }) {
  const { t } = useTranslation()
  return (
    <li data-new={fresh ? '' : undefined} className={cn('ops-log-row', fresh && !reduced && 'ops-log-enter')}>
      <div className="ops-log-main">
        <span className="ops-log-time font-mono text-xs whitespace-nowrap" title={utcTitle(line.ts)}>
          {fmtLogTime(line.ts)}
        </span>
        <span className="ops-log-level">
          <LevelChip level={line.level} />
        </span>
        <span className="ops-log-pid text-muted-foreground font-mono text-xs" title={t('operations.logs.col.pid')}>
          {line.pid ?? '–'}
        </span>
        <span className="ops-log-src truncate font-mono text-xs" title={line.logger || undefined}>
          {sourceLabel(line)}
        </span>
        <span className="ops-log-ws truncate text-xs" title={line.workspace ?? undefined}>
          {line.workspace ?? '–'}
        </span>
        <span className="ops-log-msg min-w-0 text-[13px] break-words">
          <RedactedText text={firstLine(line.msg)} />
          {line.exc_type ? <span className="text-muted-foreground ms-1.5 text-xs whitespace-nowrap">▸ {line.exc_type}</span> : null}
        </span>
      </div>
      {line.request_id || line.run_id ? (
        <div className="text-muted-foreground flex flex-wrap gap-x-2 gap-y-0.5 px-2.5 pb-1.5 text-xs">
          {line.run_id ? (
            <OpsLink to={logsView(pivotFilter('run_id', line.run_id))} className={ctxLink} title={line.run_id}>
              {t('operations.logs.ctx.run', { id: shortId(line.run_id, 6) })}
            </OpsLink>
          ) : null}
          {line.run_id && line.request_id ? <span aria-hidden="true">·</span> : null}
          {line.request_id ? (
            <OpsLink to={logsView(pivotFilter('request_id', line.request_id))} className={ctxLink} title={line.request_id}>
              {t('operations.logs.ctx.request', { id: shortId(line.request_id, 8) })}
            </OpsLink>
          ) : null}
        </div>
      ) : null}
    </li>
  )
})

/** The rows of a tail state, newest first; the motion setting is read once for the batch. */
export function TailRows({ items, fresh }: { items: TailLine[]; fresh: ReadonlySet<string> }) {
  const { reduced } = useOpsMotion()
  return (
    <ol className="m-0 list-none p-0">
      {items.map((line) => (
        <TailRow key={line.key} line={line} fresh={fresh.has(line.key)} reduced={reduced} />
      ))}
    </ol>
  )
}

/**
 * Live tail (§3.8.6): the newest lines from every live process, INFO
 * included, read from the redacted per-process JSONL files and merged by
 * time on the server. It polls every 5 s while following; Pause stops the
 * polling (and so does "⏸ auto" and a hidden tab). New lines go in at the
 * top with no list movement; scrolled away, the lines on screen stay put and
 * the "↑ N new" pill counts what arrived. At most 500 lines stay on the
 * page. Only level and module narrow it: the other filters need the stored
 * log (Stream).
 */
export default function TailView({ module, enabled }: { module: string; enabled: boolean | null }) {
  const { t } = useTranslation()
  const { paused: autoPaused } = useOpsMotion()
  const [level, setLevel] = useState<TailLevel>('INFO')
  const [following, setFollowing] = useState(true)
  const key = tailKey(level, module)
  const [view, setView] = useState<{ state: TailState; source: Batch | null; moved: boolean }>({ state: EMPTY_TAIL, source: null, moved: false })
  const [atNewest, setAtNewest] = useState(true)
  const scrollRef = useRef<HTMLDivElement | null>(null)
  const snap = useRef({ scrollTop: 0, scrollHeight: 0 })

  const poll = useOpsPoll<Batch>(
    async (signal) => ({ key, tail: await opsApi.logTail(tailQuery(level, module), signal) }),
    [key],
    following ? TAIL_SECONDS : undefined,
    { enabled: enabled !== false }
  )

  // Resumed: read now (the interval only re-arms after a request).
  const reload = poll.reload
  const wasFollowing = useRef(following)
  useEffect(() => {
    if (following && !wasFollowing.current) reload()
    wasFollowing.current = following
  }, [following, reload])

  // Take in a poll answer once, during render. Paused, an answer already on its way is not shown.
  const batch = poll.data
  if (batch && batch !== view.source && takesBatch(following, view.state.key, batch.key, key)) {
    const next = mergeTail(view.state, key, batch.tail, { baseline: poll.baseline, atNewest })
    setView({ state: next, source: batch, moved: next !== view.state })
  }

  // After lines went in above the reader: keep the lines on screen where they were.
  useLayoutEffect(() => {
    const el = scrollRef.current
    if (!el) return
    if (view.moved && !isAtNewest(snap.current.scrollTop)) {
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
    if (at && view.state.pending) setView((v) => ({ ...v, state: { ...v.state, pending: 0 }, moved: false }))
  }

  const toNewest = () => {
    const el = scrollRef.current
    if (el) el.scrollTop = 0
    setAtNewest(true)
    setView((v) => ({ ...v, state: { ...v.state, pending: 0 }, moved: false }))
  }

  const current = view.state.key === key ? view.state : null
  const { stale, ageSeconds } = useStale(following ? poll.updatedAt : null, TAIL_SECONDS)
  const off = enabled === false || current?.enabled === false

  const controls = (
    <div className="flex flex-wrap items-center gap-2">
      <Seg
        value={level}
        onChange={setLevel}
        label={t('operations.tail.levelLabel')}
        options={TAIL_LEVELS.map((l) => ({ value: l, label: t(`operations.tail.level.${l}`) }))}
      />
      <button
        type="button"
        aria-pressed={!following}
        onClick={() => setFollowing((f) => !f)}
        disabled={off}
        className={cn(
          'hover:bg-muted inline-flex min-h-8 items-center gap-1.5 rounded-lg border px-2.5 text-[13px] disabled:cursor-not-allowed disabled:opacity-50',
          !following && 'bg-muted font-medium'
        )}
      >
        {following ? <PauseIcon className="size-3.5" aria-hidden="true" /> : <PlayIcon className="size-3.5" aria-hidden="true" />}
        {following ? t('operations.tail.pause') : t('operations.tail.resume')}
      </button>
      {module ? <span className="text-muted-foreground text-xs">{t('operations.tail.module', { module })}</span> : null}
    </div>
  )

  if (off) {
    return (
      <div className="flex flex-col gap-2">
        {controls}
        <p className="text-muted-foreground m-0 rounded-lg border border-dashed px-3 py-2 text-[13px]" role="note">
          {t('operations.tail.off')}
        </p>
      </div>
    )
  }
  if (!current) {
    return (
      <div className="flex flex-col gap-2">
        {controls}
        {poll.error ? (
          <ErrorPanel error={poll.error} onRetry={poll.reload} />
        ) : (
          <div className="flex flex-col gap-1.5" aria-busy="true">
            <Skeleton className="h-8 w-full" />
            <Skeleton className="h-8 w-full" />
            <Skeleton className="h-8 w-full" />
          </div>
        )}
      </div>
    )
  }
  if (batch?.tail.available === false) return <Unavailable />

  const showStale = following && !autoPaused && (stale || !!poll.error)
  let modeText: string
  if (!following) modeText = poll.updatedAt ? t('operations.tail.pausedAt', { time: fmtClock(poll.updatedAt) }) : t('operations.tail.paused')
  else if (autoPaused) modeText = t('operations.logs.followPaused')
  else modeText = t('operations.tail.live', { seconds: TAIL_SECONDS })

  return (
    <div className={cn('relative flex flex-col gap-2', showStale && 'ops-stale')} aria-busy={poll.slow || undefined}>
      <RefreshHairline slow={poll.slow} />
      {controls}
      {poll.error ? <ErrorPanel error={poll.error} onRetry={poll.reload} compact /> : null}
      <div className="flex flex-wrap items-center justify-between gap-2 text-xs">
        <span className="text-muted-foreground">
          {modeText}
          {typeof current.files === 'number' ? ` · ${t('operations.tail.files', { count: current.files })}` : ''}
        </span>
        {showStale ? <StaleChip ageSeconds={ageSeconds} /> : null}
      </div>
      <div className="relative">
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
          aria-label={t('operations.tail.listLabel')}
        >
          <div className="ops-log-head text-muted-foreground bg-card sticky top-0 z-[1] border-b text-xs" aria-hidden="true">
            <span>{t('operations.logs.col.time')}</span>
            <span>{t('operations.logs.col.level')}</span>
            <span>{t('operations.logs.col.pid')}</span>
            <span>{t('operations.logs.col.source')}</span>
            <span>{t('operations.logs.col.workspace')}</span>
            <span>{t('operations.logs.col.message')}</span>
          </div>
          {current.items.length ? (
            <TailRows items={current.items} fresh={current.fresh} />
          ) : (
            <p className="text-muted-foreground m-0 p-4 text-sm">{t('operations.tail.empty', { level: t(`operations.tail.level.${level}`) })}</p>
          )}
          {current.items.length ? (
            <div className="text-muted-foreground flex flex-col items-center gap-1 border-t p-2.5 text-xs">
              {current.truncated ? <p className="m-0">{t('operations.tail.truncated')}</p> : null}
              <p className="m-0">
                {current.items.length >= TAIL_CAP
                  ? t('operations.tail.capped', { cap: fmtCount(TAIL_CAP) })
                  : t('operations.tail.footer', { count: fmtCount(current.items.length) })}
              </p>
            </div>
          ) : null}
        </div>
      </div>
      <p className="text-muted-foreground m-0 text-xs">{t('operations.tail.note')}</p>
    </div>
  )
}
