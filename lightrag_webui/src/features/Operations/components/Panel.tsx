import { ReactNode } from 'react'
import { useTranslation } from 'react-i18next'
import { cn } from '@/lib/utils'
import { fmtAge, fmtCount } from '../format'
import { useStale } from '../hooks'
import { OpsPoll } from '../poll'
import { pageRange } from '../processing/backlog'
import ErrorPanel from './ErrorPanel'
import RefreshHairline from './RefreshHairline'
import Skeleton from './Skeleton'
import { ToneChip } from './StatusChip'

/** "Approximate (interim)": live-job data read before run tracking, or while the Manager does not answer. */
export function ApproxChip() {
  const { t } = useTranslation()
  return (
    <ToneChip tone="unknown" icon="dashed-circle" className="font-medium">
      <span title={t('operations.approximate.hint')}>{t('operations.approximate.label')}</span>
    </ToneChip>
  )
}

/** "Stale · updated 3 min ago" in the AA-safe stale colour, never green. */
export function StaleChip({ ageSeconds }: { ageSeconds: number | null }) {
  const { t } = useTranslation()
  return (
    <span className="ops-fade-in rounded-md border border-dashed px-1.5 py-0.5 text-xs text-[color:var(--ops-stale-fg)]">
      {t('operations.stale', { age: fmtAge(ageSeconds) })}
    </span>
  )
}

/** The "requires PostgreSQL KV storage" state of a view (§2.2 "Feature unavailable"). */
export function Unavailable() {
  const { t } = useTranslation()
  return <p className="text-muted-foreground m-0 text-sm">{t('operations.unavailableShort')}</p>
}

/**
 * One panel of the console (§2.2 panel states): a fixed-height skeleton on
 * first load, old data kept with a 2 px hairline while refreshing, dimmed with
 * a stale chip when older than twice its freshness, the mapped error with
 * Ref and Retry, and a plain note when the server has no Postgres.
 */
export function Panel<T>({
  poll,
  freshness,
  title,
  sub,
  actions,
  approximate,
  skeleton,
  className,
  children
}: {
  poll: OpsPoll<T>
  /** Seconds the source is fresh for (§3.6); stale after twice that. */
  freshness: number
  title?: ReactNode
  sub?: ReactNode
  actions?: ReactNode
  approximate?: boolean
  skeleton?: ReactNode
  className?: string
  children: (data: T) => ReactNode
}) {
  const { stale, ageSeconds } = useStale(poll.updatedAt, freshness)
  const isStale = !!poll.data && (stale || (!!poll.error && !!poll.data))
  const data = poll.data
  const unavailable = !!data && (data as { available?: boolean }).available === false
  return (
    <section
      aria-busy={poll.slow || undefined}
      className={cn('bg-card relative flex min-w-0 flex-col gap-3 rounded-[10px] border p-4', isStale && 'ops-stale', className)}
    >
      <RefreshHairline slow={poll.slow} />
      {title || actions || approximate || isStale ? (
        <div className="flex flex-wrap items-start justify-between gap-x-3 gap-y-1.5">
          <div className="min-w-0">
            {title ? <h2 className="m-0 text-[15px] font-semibold">{title}</h2> : null}
            {sub ? <p className="text-muted-foreground mt-0.5 mb-0 text-[13px]">{sub}</p> : null}
          </div>
          <div className="flex flex-wrap items-center gap-2">
            {approximate ? <ApproxChip /> : null}
            {isStale ? <StaleChip ageSeconds={ageSeconds} /> : null}
            {actions}
          </div>
        </div>
      ) : null}
      {poll.error && data ? <ErrorPanel error={poll.error} onRetry={poll.reload} compact /> : null}
      {poll.error && !data ? (
        <ErrorPanel error={poll.error} onRetry={poll.reload} />
      ) : !data ? (
        (skeleton ?? <Skeleton className="h-24 w-full" />)
      ) : unavailable ? (
        <Unavailable />
      ) : (
        children(data)
      )}
    </section>
  )
}

/** "‹ Prev · 1–50 of 1,312 · Next ›" for server-paged lists. */
export function Pager({ page, pageSize, total, onPage }: { page: number; pageSize: number; total: number; onPage: (page: number) => void }) {
  const { t } = useTranslation()
  const { first, last, pages } = pageRange(page, pageSize, total)
  if (total <= pageSize && page <= 1) return null
  return (
    <div className="flex flex-wrap items-center justify-end gap-2 text-xs">
      <button
        type="button"
        className="hover:bg-muted min-h-8 rounded-lg border px-3 text-[13px] disabled:opacity-50"
        disabled={page <= 1}
        onClick={() => onPage(page - 1)}
      >
        {t('operations.pager.previous')}
      </button>
      <span className="text-muted-foreground">{t('operations.pager.range', { first: fmtCount(first), last: fmtCount(last), total: fmtCount(total) })}</span>
      <button
        type="button"
        className="hover:bg-muted min-h-8 rounded-lg border px-3 text-[13px] disabled:opacity-50"
        disabled={page >= pages}
        onClick={() => onPage(page + 1)}
      >
        {t('operations.pager.next')}
      </button>
    </div>
  )
}
