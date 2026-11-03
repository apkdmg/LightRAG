import { useEffect, useState } from 'react'
import { useTranslation } from 'react-i18next'
import { useOpsMotion } from '../motion'

const SWEEP_CAP_MS = 5000

/**
 * The 2 px bar at a panel's top while a request runs longer than 300 ms
 * (`slow` from useOpsPoll). The sweep stops at 5 s and turns into a static bar
 * with "Still loading · 6 s". Under reduced motion or "⏸ auto" it is static
 * from the start. Fast polls never show it.
 */
export default function RefreshHairline({ slow }: { slow: boolean }) {
  const { t } = useTranslation()
  const { reduced, paused } = useOpsMotion()
  const [elapsed, setElapsed] = useState(0)

  useEffect(() => {
    if (!slow) return
    const start = Date.now()
    const timer = setInterval(() => setElapsed(Date.now() - start), 1000)
    return () => {
      clearInterval(timer)
      setElapsed(0)
    }
  }, [slow])

  if (!slow) return null
  const capped = elapsed >= SWEEP_CAP_MS
  return (
    <>
      <span className="ops-hairline" aria-hidden="true" data-static={capped || reduced || paused ? '' : undefined}>
        <i />
      </span>
      {capped ? (
        <span className="text-muted-foreground text-xs">{t('operations.stillLoading', { seconds: Math.round(elapsed / 1000) })}</span>
      ) : null}
    </>
  )
}
