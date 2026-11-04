import { useState } from 'react'
import { useTranslation } from 'react-i18next'
import type { StripCell } from '../api'
import { cellLabel } from '../health/model'
import { useOpsMotion } from '../motion'

const keyOf = (cell: StripCell, i: number) => (cell.start !== null ? String(cell.start) : `i${i}`)

/**
 * The 24 h availability strip (§2.6, §2.12): 48 half-hour cells, oldest on
 * the left. Every cell carries its meaning in an aria-label and a non-colour
 * mark (failed cells are hatched, warnings half filled, unknown cells a dashed
 * outline). The first paint is final; afterwards a new half-hour cell fades in
 * at the right without the strip sliding, and a changed cell crossfades its
 * colour. Baseline polls replay nothing.
 */
export default function AvailabilityStrip({ cells, baseline, label }: { cells: StripCell[]; baseline?: boolean; label?: string }) {
  const { t } = useTranslation()
  const { reduced } = useOpsMotion()
  const keys = cells.map(keyOf)
  const sig = keys.join(',')
  // The keys seen so far, and which ones are new since the previous render (the "state from the previous render" pattern).
  const [mem, setMem] = useState(() => ({ sig, all: new Set(keys), fresh: new Set<string>() }))
  if (mem.sig !== sig) {
    const fresh = baseline ? new Set<string>() : new Set(keys.filter((k) => !mem.all.has(k)))
    setMem({ sig, all: new Set([...mem.all, ...keys]), fresh })
  }
  if (!cells.length) return null
  return (
    <span className="ops-dim ops-strip" role="group" aria-label={label ?? t('operations.strip.label')}>
      {cells.map((c, i) => {
        const key = keys[i]
        const text = cellLabel(c, (k, o) => t(k, o) as string)
        return (
          <span
            key={key}
            role="img"
            aria-label={text}
            title={text}
            data-s={c.status}
            className={!reduced && !baseline && mem.fresh.has(key) ? 'ops-cell ops-cell-new' : 'ops-cell'}
          />
        )
      })}
    </span>
  )
}
