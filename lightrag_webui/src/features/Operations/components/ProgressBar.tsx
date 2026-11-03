import { useState } from 'react'
import { useTranslation } from 'react-i18next'
import { fmtCount } from '../format'

/**
 * done/total with a scaleX bar from the left (§2.12). It tweens forward only:
 * a new run, a smaller total or fewer done docs jump. The "4/12" text is
 * always beside it, and the bar carries role="progressbar".
 */
export default function ProgressBar({
  done,
  total,
  tone
}: {
  done: number | null | undefined
  total: number | null | undefined
  tone?: 'serious' | 'crit' | 'paused'
}) {
  const { t } = useTranslation()
  const d = typeof done === 'number' && Number.isFinite(done) ? Math.max(0, done) : null
  const n = typeof total === 'number' && Number.isFinite(total) ? Math.max(0, total) : null
  // Forward-only: decided against the previous render's numbers.
  const [last, setLast] = useState({ d, n, jump: true })
  if (last.d !== d || last.n !== n) setLast({ d, n, jump: d === null || n === null || last.d === null || n !== last.n || d < last.d })
  if (d === null || n === null || n <= 0) return <span className="text-muted-foreground font-mono text-xs">{d !== null ? fmtCount(d) : '–'}</span>
  const ratio = Math.min(1, d / n)
  return (
    <span className="inline-flex items-center gap-1.5 whitespace-nowrap">
      <span
        className="ops-bar ops-dim"
        role="progressbar"
        aria-label={t('operations.active.progressLabel')}
        aria-valuemin={0}
        aria-valuemax={n}
        aria-valuenow={Math.min(d, n)}
        data-jump={last.jump ? '' : undefined}
        data-tone={tone}
      >
        <i style={{ transform: `scaleX(${ratio.toFixed(3)})` }} />
      </span>
      <span className="font-mono text-xs">
        {fmtCount(Math.min(d, n))}/{fmtCount(n)}
      </span>
    </span>
  )
}
