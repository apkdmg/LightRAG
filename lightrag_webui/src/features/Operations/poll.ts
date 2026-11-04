import { useCallback, useEffect, useRef, useState, useSyncExternalStore } from 'react'
import { OpsError, isAbort, toOpsError } from './errors'
import { useOpsPrefs } from './prefs'

/**
 * Polling for Operations panels (spec §3.10):
 * - only mounted (visible) panels poll; everything pauses while the browser
 *   tab is hidden and while "⏸ auto" is on
 * - refreshes align to a shared wall-clock tick, so panels with the same
 *   interval refresh together instead of drifting apart
 * - errors back off exponentially, and the old data stays on screen
 * - one request in flight per panel; a superseded request is aborted
 * - an unchanged payload keeps the previous object, so nothing re-renders
 *   or animates for data that did not change (§2.12 rule 1)
 * - the first poll after mount, after the tab becomes visible again, after
 *   resuming and after an error clears is a "baseline": views jump to it
 *   without replaying change animations (§2.12 rule 7)
 */

export const MAX_BACKOFF_SECONDS = 300
/** A request still running after this long shows the refresh hairline. */
export const SLOW_REQUEST_MS = 300

/** Seconds to wait after `failures` consecutive errors: base × 2^failures, capped at 5 min. */
export function backoffSeconds(baseSeconds: number, failures: number): number {
  const base = Math.max(1, baseSeconds)
  const n = Math.min(Math.max(0, Math.floor(failures)), 16)
  return Math.min(base * 2 ** n, Math.max(base, MAX_BACKOFF_SECONDS))
}

/** Milliseconds until the next wall-clock multiple of `seconds` (never 0: a full period at the boundary). */
export function alignedDelay(now: number, seconds: number): number {
  const period = Math.max(1, seconds) * 1000
  const rest = period - (now % period)
  return rest === 0 ? period : rest
}

/** Delay before the next poll: aligned to the shared tick when healthy, backed off after errors. */
export function nextPollDelay(now: number, seconds: number, failures: number): number {
  return failures > 0 ? backoffSeconds(seconds, failures) * 1000 : alignedDelay(now, seconds)
}

const IGNORED_TOP_LEVEL = new Set(['generated_at'])

const deepEqual = (a: unknown, b: unknown): boolean => {
  if (Object.is(a, b)) return true
  if (typeof a !== 'object' || typeof b !== 'object' || a === null || b === null) return false
  if (Array.isArray(a) !== Array.isArray(b)) return false
  if (Array.isArray(a)) {
    const bb = b as unknown[]
    return a.length === bb.length && a.every((v, i) => deepEqual(v, bb[i]))
  }
  const ka = Object.keys(a as object)
  const kb = Object.keys(b as object)
  if (ka.length !== kb.length) return false
  return ka.every((k) => Object.prototype.hasOwnProperty.call(b, k) && deepEqual((a as any)[k], (b as any)[k]))
}

/** Payload equality for polling: `generated_at` changes on every response and is not data. */
export function samePayload(a: unknown, b: unknown): boolean {
  if (a && b && typeof a === 'object' && typeof b === 'object' && !Array.isArray(a) && !Array.isArray(b)) {
    const strip = (o: object) => Object.fromEntries(Object.entries(o).filter(([k]) => !IGNORED_TOP_LEVEL.has(k)))
    return deepEqual(strip(a), strip(b))
  }
  return deepEqual(a, b)
}

/** The previous object when the new payload carries the same data, else the new one. */
export function keepIfSame<T>(previous: T | null, next: T): T {
  return previous !== null && samePayload(previous, next) ? previous : next
}

// ---------- document visibility (shared by every panel) ----------

const subscribeVisibility = (fn: () => void) => {
  if (typeof document === 'undefined') return () => {}
  document.addEventListener('visibilitychange', fn)
  return () => document.removeEventListener('visibilitychange', fn)
}
const hiddenSnapshot = () => (typeof document === 'undefined' ? false : document.hidden)

/** True while the browser tab is hidden; no panel polls then. */
export function useDocumentHidden(): boolean {
  return useSyncExternalStore(subscribeVisibility, hiddenSnapshot, () => false)
}

// ---------- the hook ----------

type TimerRef = { current: ReturnType<typeof setTimeout> | null }

const clearTimer = (ref: TimerRef) => {
  if (ref.current !== null) clearTimeout(ref.current)
  ref.current = null
}

export type OpsPoll<T> = {
  data: T | null
  error: OpsError | null
  /** No answer yet for the current inputs (first load or changed filters). */
  loading: boolean
  /** A request has been running longer than SLOW_REQUEST_MS. */
  slow: boolean
  /** When the shown data was last confirmed by the server (epoch ms). */
  updatedAt: number | null
  /** The shown data came from a baseline poll: render it without change animations. */
  baseline: boolean
  reload: () => void
}

export type OpsPollOptions = {
  /** false: do not fetch at all (for example a feature the server reports unavailable). */
  enabled?: boolean
}

export function useOpsPoll<T>(
  fetcher: (signal: AbortSignal) => Promise<T>,
  deps: unknown[],
  seconds?: number,
  { enabled = true }: OpsPollOptions = {}
): OpsPoll<T> {
  const { paused } = useOpsPrefs()
  const hidden = useDocumentHidden()
  const depsKey = JSON.stringify(deps)

  const [data, setData] = useState<T | null>(null)
  const [error, setError] = useState<OpsError | null>(null)
  const [loadedKey, setLoadedKey] = useState<string | null>(null)
  const [updatedAt, setUpdatedAt] = useState<number | null>(null)
  const [baseline, setBaseline] = useState(true)
  const [slow, setSlow] = useState(false)

  const fetcherRef = useRef(fetcher)
  const controllerRef = useRef<AbortController | null>(null)
  const timerRef = useRef<ReturnType<typeof setTimeout> | null>(null)
  const slowTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null)
  const failuresRef = useRef(0)
  const baselineNextRef = useRef(true)
  const requestRef = useRef(0)
  const activeRef = useRef(false) // polling allowed (enabled, visible, not paused)
  const keyRef = useRef(depsKey)
  const secondsRef = useRef(seconds)
  const loadRef = useRef<() => Promise<void>>(async () => {})

  useEffect(() => {
    fetcherRef.current = fetcher
    secondsRef.current = seconds
  })

  const load = useCallback(async () => {
    clearTimer(timerRef)
    controllerRef.current?.abort()
    const controller = new AbortController()
    controllerRef.current = controller
    const id = ++requestRef.current
    const key = keyRef.current
    clearTimer(slowTimerRef)
    slowTimerRef.current = setTimeout(() => {
      if (id === requestRef.current) setSlow(true)
    }, SLOW_REQUEST_MS)

    let succeeded = false
    try {
      const result = await fetcherRef.current(controller.signal)
      if (id !== requestRef.current) return
      succeeded = true
      failuresRef.current = 0
      const isBaseline = baselineNextRef.current
      baselineNextRef.current = false
      setData((previous) => keepIfSame(previous, result))
      setError(null)
      setBaseline(isBaseline)
      setUpdatedAt(Date.now())
      setLoadedKey(key)
    } catch (e) {
      if (id !== requestRef.current || isAbort(e)) return
      failuresRef.current += 1
      // The first good answer after an error is a baseline: nothing replays.
      baselineNextRef.current = true
      setError(toOpsError(e))
      setLoadedKey(key)
    } finally {
      if (id === requestRef.current) {
        clearTimer(slowTimerRef)
        setSlow(false)
        controllerRef.current = null
        const every = secondsRef.current
        if (activeRef.current && every) {
          const delay = nextPollDelay(Date.now(), every, succeeded ? 0 : failuresRef.current)
          // Through the ref: the next tick always runs the current loader.
          timerRef.current = setTimeout(() => void loadRef.current(), delay)
        }
      }
    }
  }, [])

  // New inputs: fetch now (also while paused, so a changed filter still shows its data once).
  useEffect(() => {
    keyRef.current = depsKey
    if (!enabled || hidden) return
    void load()
    return () => {
      clearTimer(timerRef)
      controllerRef.current?.abort()
    }
    // `hidden` is handled by the effect below; re-running here would double-fetch.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [depsKey, enabled, load])

  // Pause, resume and visibility. Coming back (visible again, resumed) starts
  // with a baseline poll; turning `enabled` on is the effect above's job.
  const wasLive = useRef<boolean | null>(null)
  useEffect(() => {
    const live = !paused && !hidden
    activeRef.current = enabled && live
    const previous = wasLive.current
    wasLive.current = live
    if (!enabled || !live) {
      clearTimer(timerRef)
      return
    }
    if (previous === false) {
      baselineNextRef.current = true
      void load()
    }
  }, [enabled, paused, hidden, load])

  useEffect(
    () => () => {
      requestRef.current += 1
      clearTimer(timerRef)
      clearTimer(slowTimerRef)
      controllerRef.current?.abort()
    },
    []
  )

  useEffect(() => {
    loadRef.current = load
  }, [load])

  const reload = useCallback(() => void load(), [load])
  return { data, error, loading: loadedKey !== depsKey, slow, updatedAt, baseline, reload }
}
