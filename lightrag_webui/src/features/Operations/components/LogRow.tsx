import { memo } from 'react'
import { useTranslation } from 'react-i18next'
import { cn } from '@/lib/utils'
import { LogEntry } from '../api'
import { useOpsNav } from '../context'
import { useOpsMotion } from '../motion'
import { shortId, utcTitle } from '../format'
import { firstLine, fmtLogTime, levelName, levelState, logsView, pivotFilter, sourceLabel } from '../logs'
import { lookOf } from '../severity'
import OpsLink from './OpsLink'
import { RedactedText } from './RedactedSpan'
import { ToneChip } from './StatusChip'

/** "ERROR" with the severity icon: level is never colour alone (§2.2). */
export function LevelChip({ level }: { level: number }) {
  const look = lookOf(levelState(level))
  return (
    <ToneChip tone={look.tone} icon={look.icon} className="px-1.5 font-mono text-[11px]">
      {levelName(level)}
    </ToneChip>
  )
}

const ctxLink = 'font-mono underline-offset-2 hover:underline'

/** The second line: the IDs a row carries, each a pivot (§2.5). */
function ContextLine({ entry, inDrawer }: { entry: LogEntry; inDrawer: boolean }) {
  const { t } = useTranslation()
  const nav = useOpsNav()
  const push = { push: inDrawer }
  const parts: React.ReactNode[] = []
  if (entry.run_id) {
    const run = entry.run_id
    parts.push(
      <OpsLink key="run" to={{ ...nav.view, drawer: { kind: 'run', id: run } }} onNavigate={() => nav.openDrawer({ kind: 'run', id: run }, push)} className={ctxLink}>
        {t('operations.logs.ctx.run', { id: shortId(run, 6) })}
      </OpsLink>
    )
  }
  if (entry.track_id) {
    parts.push(
      <OpsLink key="track" to={logsView(pivotFilter('track_id', entry.track_id))} className={ctxLink} title={entry.track_id}>
        {t('operations.logs.ctx.track', { id: shortId(entry.track_id, 14) })}
      </OpsLink>
    )
  }
  if (entry.doc_id) {
    const doc = entry.doc_id
    const ws = entry.workspace
    parts.push(
      ws ? (
        <OpsLink
          key="doc"
          to={{ ...nav.view, drawer: { kind: 'document', workspace: ws, id: doc } }}
          onNavigate={() => nav.openDrawer({ kind: 'document', workspace: ws, id: doc }, push)}
          className={ctxLink}
        >
          {t('operations.logs.ctx.doc', { id: shortId(doc, 10) })}
        </OpsLink>
      ) : (
        <span key="doc" className="font-mono">
          {t('operations.logs.ctx.doc', { id: shortId(doc, 10) })}
        </span>
      )
    )
  }
  if (entry.request_id) {
    parts.push(
      <OpsLink key="req" to={logsView(pivotFilter('request_id', entry.request_id))} className={ctxLink} title={entry.request_id}>
        {t('operations.logs.ctx.request', { id: shortId(entry.request_id, 8) })}
      </OpsLink>
    )
  }
  if (!parts.length) return null
  return (
    <div className="text-muted-foreground flex flex-wrap gap-x-2 gap-y-0.5 px-2.5 pb-1.5 text-xs">
      {parts.map((p, i) => (
        <span key={i} className="inline-flex items-center gap-2">
          {i ? <span aria-hidden="true">·</span> : null}
          {p}
        </span>
      ))}
    </div>
  )
}

/**
 * One log line (§2.5): time (local, UTC in the tooltip), level, pid, source,
 * workspace and the redacted message, with "▸ traceback" when one was
 * stored and "×N" for coalesced repeats. The line is a real link to the
 * entry drawer; the IDs below it are pivots.
 *
 * Motion (§2.12 "Follow-mode log rows"): a row from the latest follow batch
 * carries a 2 px marker until the next batch and fades in over 150 ms. No
 * transform, no layout animation, no will-change. Under reduced motion it
 * appears at once and only the marker stays.
 */
function LogRow({ entry, fresh = false, reduced = false, inDrawer = false, showWorkspace = true }: {
  entry: LogEntry
  /** Part of the latest follow batch. */
  fresh?: boolean
  reduced?: boolean
  /** Rendered in a drawer section: links push onto the breadcrumb. */
  inDrawer?: boolean
  showWorkspace?: boolean
}) {
  const { t } = useTranslation()
  const nav = useOpsNav()
  const open = () => nav.openDrawer({ kind: 'log', id: String(entry.id) }, { push: inDrawer })
  return (
    <li data-new={fresh ? '' : undefined} className={cn('ops-log-row', inDrawer && 'ops-log-compact', fresh && !reduced && 'ops-log-enter')}>
      <OpsLink to={{ ...nav.view, drawer: { kind: 'log', id: String(entry.id) } }} onNavigate={open} className={cn('ops-log-main', !showWorkspace && 'ops-log-main-nows')}>
        <span className="ops-log-time font-mono text-xs whitespace-nowrap" title={utcTitle(entry.ts)}>
          {fmtLogTime(entry.ts)}
        </span>
        <span className="ops-log-level">
          <LevelChip level={entry.level} />
        </span>
        <span className="ops-log-pid text-muted-foreground font-mono text-xs" title={t('operations.logs.col.pid')}>
          {entry.pid ?? '–'}
        </span>
        <span className="ops-log-src truncate font-mono text-xs" title={entry.logger || undefined}>
          {sourceLabel(entry)}
        </span>
        {showWorkspace ? (
          <span className="ops-log-ws truncate text-xs" title={entry.workspace ?? undefined}>
            {entry.workspace ?? '–'}
          </span>
        ) : null}
        <span className="ops-log-msg min-w-0 text-[13px] break-words">
          <RedactedText text={firstLine(entry.msg)} />
          {entry.has_exc ? <span className="text-muted-foreground ms-1.5 text-xs whitespace-nowrap">▸ {entry.exc_type ?? t('operations.logs.traceback')}</span> : null}
          {entry.repeat_count > 1 ? (
            <span className="text-muted-foreground ms-1.5 font-mono text-xs whitespace-nowrap" title={t('operations.logs.repeatHint')}>
              ×{entry.repeat_count}
            </span>
          ) : null}
        </span>
      </OpsLink>
      <ContextLine entry={entry} inDrawer={inDrawer} />
    </li>
  )
}

const MemoLogRow = memo(LogRow)
export default MemoLogRow

/**
 * A batch of rows, newest first. Reads the motion setting once for the
 * batch (not once per row): under reduced motion, fresh rows appear at once
 * with their marker and no fade.
 */
export function LogRows({
  items,
  fresh,
  inDrawer = false,
  showWorkspace = true,
  className
}: {
  items: LogEntry[]
  fresh?: ReadonlySet<number>
  inDrawer?: boolean
  showWorkspace?: boolean
  className?: string
}) {
  const { reduced } = useOpsMotion()
  return (
    <ol className={cn('m-0 list-none p-0', className)}>
      {items.map((e) => (
        <MemoLogRow key={e.id} entry={e} fresh={!!fresh?.has(e.id)} reduced={reduced} inDrawer={inDrawer} showWorkspace={showWorkspace} />
      ))}
    </ol>
  )
}
