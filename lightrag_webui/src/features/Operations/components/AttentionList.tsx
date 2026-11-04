import { FocusEvent, ReactNode, useState } from 'react'
import { AnimatePresence, motion } from 'motion/react'
import { useTranslation } from 'react-i18next'
import { ArrowRightIcon, BellIcon } from 'lucide-react'
import { cn } from '@/lib/utils'
import { AttentionItem } from '../api'
import { ageSeconds, fmtAge, fmtCount } from '../format'
import { OpsHash, viewForTarget } from '../hashState'
import { TWEEN, useOpsMotion, useRing, useRingToken } from '../motion'
import { attentionTarget, sortAttention } from '../overview'
import { lookOf } from '../severity'
import OpsLink from './OpsLink'
import { StateIcon } from './StatusChip'

type Row = { item: AttentionItem; resolved: boolean }

/** The rule's i18n title, with the server's own text winning when it sends one. */
function useItemText(item: AttentionItem) {
  const { t } = useTranslation()
  const params = { count: item.count ?? 0, countText: fmtCount(item.count ?? null), workspace: item.workspace ?? '' }
  const title = item.title || t(`operations.rule.${item.rule}.title`, { ...params, defaultValue: item.rule })
  const runbook = item.runbook || t(`operations.rule.${item.rule}.runbook`, { ...params, defaultValue: '' })
  return { title, runbook }
}

function AttentionRow({ row, base, now }: { row: Row; base: OpsHash; now: number | null }) {
  const { t } = useTranslation()
  const { item, resolved } = row
  const look = lookOf(item.severity)
  const ring = useRing<HTMLSpanElement>(useRingToken(`att:${item.id}`))
  const { title, runbook } = useItemText(item)
  const target = viewForTarget(base, attentionTarget(item))
  const since = item.since ? ageSeconds(item.since, now ?? undefined) : null
  const alertId = item.alert_id
  const link = (
    <OpsLink
      to={target}
      className={cn(
        'hover:bg-muted grid min-w-0 grid-cols-[auto_minmax(0,1fr)_auto] items-start gap-x-3 gap-y-0.5 rounded-lg px-2 py-2 text-left no-underline',
        alertId && 'flex-1',
        resolved && 'opacity-60'
      )}
    >
      <span className="inline-flex items-center gap-1.5 pt-px text-xs font-semibold whitespace-nowrap" style={{ color: `var(--ops-${look.tone})` }}>
        <span ref={ring} className="ops-ringable inline-flex rounded-full">
          <StateIcon icon={look.icon} />
        </span>
        {t(look.label)}
      </span>
      <span className="min-w-0 text-[13px] font-medium">
        {title}
        {item.workspace && !title.includes(item.workspace) ? <span className="text-muted-foreground font-normal"> · {item.workspace}</span> : null}
        {since !== null ? <span className="text-muted-foreground font-normal"> · {fmtAge(since)}</span> : null}
        {resolved ? <span className="text-muted-foreground font-normal"> · {t('operations.attention.resolved')}</span> : null}
        {item.acked_by ? <span className="text-muted-foreground font-normal"> · {t('operations.attention.ackedBy', { by: item.acked_by })}</span> : null}
        {item.muted ? <span className="text-muted-foreground font-normal"> · {t('operations.attention.muted')}</span> : null}
      </span>
      <ArrowRightIcon className="text-muted-foreground mt-0.5 size-3.5" aria-hidden="true" />
      {item.detail ? <span className="text-muted-foreground col-start-2 text-xs">{item.detail}</span> : null}
      {runbook ? <span className="col-start-2 text-xs">{t('operations.attention.next', { step: runbook })}</span> : null}
    </OpsLink>
  )
  if (!alertId) return link
  // The alert behind the item (PR-8): a sibling link, since links cannot nest.
  return (
    <div className="flex items-start gap-1">
      {link}
      <OpsLink
        to={{ ...base, drawer: { kind: 'alert', id: alertId } }}
        className="hover:bg-muted text-muted-foreground hover:text-foreground mt-1 inline-flex size-8 flex-none items-center justify-center rounded-lg"
        aria-label={t('operations.attention.openAlert', { title })}
        title={t('operations.attention.openAlert', { title })}
      >
        <BellIcon className="size-3.5" aria-hidden="true" />
      </OpsLink>
    </div>
  )
}

/**
 * "Needs attention" (§2.3, §2.12). Worst first. A new item enters at its
 * sorted position (y −8 → 0 with a fade); rows below shift by FLIP. A
 * resolved item collapses, unless the pointer or focus is in the list, in
 * which case it stays (marked resolved) until they leave. Baseline polls and
 * reduced motion insert and remove in place. Every row is a real link.
 */
export default function AttentionList({
  items,
  base,
  baseline,
  now,
  empty
}: {
  items: AttentionItem[]
  base: OpsHash
  baseline?: boolean
  /** When the data was confirmed (epoch ms), for "since" ages. */
  now?: number | null
  /** What was checked, said when nothing needs attention. */
  empty: ReactNode
}) {
  const { reduced, hidden } = useOpsMotion()
  const sorted = sortAttention(items)
  const instant = !!baseline || reduced || hidden

  // Rows removed while the pointer or focus is inside stay until both leave.
  const [inside, setInside] = useState({ pointer: false, focus: false })
  const holding = inside.pointer || inside.focus
  const [track, setTrack] = useState<{ ids: string; items: AttentionItem[]; held: AttentionItem[] }>(() => ({
    ids: sorted.map((a) => a.id).join('\n'),
    items: sorted,
    held: []
  }))
  const ids = sorted.map((a) => a.id).join('\n')
  if (track.ids !== ids) {
    const current = new Set(sorted.map((a) => a.id))
    const removed = holding ? track.items.filter((a) => !current.has(a.id)) : []
    setTrack({ ids, items: sorted, held: [...track.held.filter((a) => !current.has(a.id)), ...removed] })
  }
  const leave = (part: 'pointer' | 'focus') => {
    const next = { ...inside, [part]: false }
    setInside(next)
    if (!next.pointer && !next.focus) setTrack((tr) => (tr.held.length ? { ...tr, held: [] } : tr))
  }
  const rows: Row[] = [...sorted.map((item) => ({ item, resolved: false })), ...track.held.map((item) => ({ item, resolved: true }))]

  if (!rows.length) return <p className="text-muted-foreground m-0 text-sm">{empty}</p>

  const enter = instant ? false : { opacity: 0, y: -8 }
  return (
    <ul
      className="m-0 flex list-none flex-col p-0"
      onPointerEnter={() => setInside((v) => ({ ...v, pointer: true }))}
      onPointerLeave={() => leave('pointer')}
      onFocusCapture={() => setInside((v) => (v.focus ? v : { ...v, focus: true }))}
      onBlurCapture={(e: FocusEvent<HTMLUListElement>) => {
        if (!e.currentTarget.contains(e.relatedTarget as Node | null)) leave('focus')
      }}
    >
      <AnimatePresence initial={false} custom={instant}>
        {rows.map((row) => (
          <motion.li
            key={row.item.id}
            layout={instant ? false : 'position'}
            custom={instant}
            initial={enter}
            animate={{ opacity: 1, y: 0, transition: TWEEN.panel }}
            exit="gone"
            variants={{
              gone: (skip: boolean) =>
                skip ? { opacity: 0, transition: { duration: 0 } } : { opacity: 0, height: 0, transition: TWEEN.baseExit }
            }}
            transition={TWEEN.panel}
            className="overflow-hidden"
          >
            <AttentionRow row={row} base={base} now={now ?? null} />
          </motion.li>
        ))}
      </AnimatePresence>
    </ul>
  )
}
