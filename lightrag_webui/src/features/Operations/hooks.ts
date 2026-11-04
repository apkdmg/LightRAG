import { RefObject, useEffect, useState } from 'react'
import { useDocumentHidden } from './poll'
import { isStale } from './severity'

/** Re-renders a small part of the page every `ms`, so "updated 3 min ago" moves; idle while hidden. */
export function useNow(ms: number, enabled = true) {
  const hidden = useDocumentHidden()
  const [now, setNow] = useState(() => Date.now())
  useEffect(() => {
    if (hidden || !enabled) return
    const timer = setInterval(() => setNow(Date.now()), ms)
    return () => clearInterval(timer)
  }, [ms, hidden, enabled])
  return now
}

/** While stale, the "updated N ago" text moves at this pace. */
const STALE_TICK_MS = 15_000

/**
 * Whether data confirmed at `updatedAt` is stale: older than twice its
 * source's freshness (§2.12). A fresh panel costs one timer that fires at the
 * moment it would turn stale, not a re-render per tick; only a stale panel
 * ticks, to keep its "updated N ago" honest.
 */
export function useStale(updatedAt: number | null, freshnessSeconds: number): { stale: boolean; ageSeconds: number | null } {
  const [now, setNow] = useState(() => Date.now())
  const stale = updatedAt !== null && isStale(updatedAt, freshnessSeconds, now)
  useEffect(() => {
    if (updatedAt === null) return
    const turnsStaleIn = updatedAt + 2 * freshnessSeconds * 1000 - Date.now()
    const timer =
      turnsStaleIn > 0
        ? setTimeout(() => setNow(Date.now()), turnsStaleIn + 50)
        : setInterval(() => setNow(Date.now()), STALE_TICK_MS)
    return () => {
      clearTimeout(timer as ReturnType<typeof setTimeout>)
      clearInterval(timer as ReturnType<typeof setInterval>)
    }
  }, [updatedAt, freshnessSeconds, stale])
  return { stale, ageSeconds: updatedAt === null ? null : Math.max(0, (now - updatedAt) / 1000) }
}

/**
 * Whether at least `threshold` of the element is inside the viewport. Without
 * IntersectionObserver (server rendering, tests) it reports true, so a
 * missing API never hides live state; the first observer callback corrects it.
 */
export function useInView<T extends Element>(ref: RefObject<T | null>, threshold = 0.5): boolean {
  const [inView, setInView] = useState(true)
  useEffect(() => {
    const el = ref.current
    if (!el || typeof IntersectionObserver === 'undefined') return
    const io = new IntersectionObserver(([entry]) => setInView(!!entry && entry.intersectionRatio >= threshold), { threshold: [0, threshold, 1] })
    io.observe(el)
    return () => io.disconnect()
  }, [ref, threshold])
  return inView
}
