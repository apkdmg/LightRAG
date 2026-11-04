import { createContext, useContext, useEffect, useLayoutEffect, useRef, useState, useSyncExternalStore } from 'react'
import { animate, useReducedMotion } from 'motion/react'
import { useDocumentHidden } from './poll'
import { useOpsPrefs } from './prefs'
import { rankOf } from './severity'

/**
 * Motion tokens for the Operations console, mirroring the CSS variables in
 * ops.css (spec §2.12). Ops-owned Motion transitions always pass a tween with
 * an explicit duration and ease: a missing transition would fall back to
 * Motion's default spring, which overshoots, and an overshooting number reads
 * as a wrong number.
 */

export const DUR = {
  fast: 0.15,
  base: 0.2,
  panel: 0.24,
  value: 0.3,
  ring: 0.9,
  breathe: 2
} as const

export const EASE = {
  out: 'easeOut',
  in: 'easeIn',
  loop: 'easeInOut',
  flow: 'linear'
} as const

export const TWEEN = {
  fast: { type: 'tween', duration: DUR.fast, ease: EASE.out },
  base: { type: 'tween', duration: DUR.base, ease: EASE.out },
  baseExit: { type: 'tween', duration: DUR.base, ease: EASE.in },
  panel: { type: 'tween', duration: DUR.panel, ease: EASE.out },
  panelExit: { type: 'tween', duration: DUR.panel, ease: EASE.in },
  value: { type: 'tween', duration: DUR.value, ease: EASE.out },
  /** Every fade under reduced motion (the fallback column of §2.12). */
  reducedFade: { type: 'tween', duration: DUR.fast, ease: 'linear' }
} as const

/** Same target at the same rank rings at most once per 10 minutes. */
export const RING_COOLDOWN_MS = 10 * 60 * 1000
/** A KPI delta line stays until the next change or for 15 minutes. */
export const DELTA_HOLD_MS = 15 * 60 * 1000
/** The KPI tint is held for 1 s, then fades over --ops-dur-value. */
export const TINT_HOLD_MS = 1000

/** Every ops chart spreads this: monitoring charts never redraw-animate on a poll. */
export const OPS_CHART = { isAnimationActive: false } as const

export type OpsMotion = {
  /** OS prefers-reduced-motion, or the page's "Reduce motion" toggle. */
  reduced: boolean
  /** The OS setting alone; the page toggle cannot switch motion back on while it is set. */
  osReduced: boolean
  /** "⏸ auto" is on: loops stop because frozen data cannot vouch that work is still running. */
  paused: boolean
  /** The browser tab is hidden. */
  hidden: boolean
}

export function useOpsMotion(): OpsMotion {
  const { paused, reduceMotion } = useOpsPrefs()
  const osReduced = useReducedMotion()
  const hidden = useDocumentHidden()
  return { reduced: !!osReduced || reduceMotion, osReduced: !!osReduced, paused, hidden }
}

// ---------- pure decision helpers (motion.test.ts) ----------

export type MotionCtx = {
  /** The data came from a baseline poll (after mount, a hidden tab, a resume or an error). */
  baseline?: boolean
  reduced?: boolean
  hidden?: boolean
  /** False on first paint: there is nothing to animate from. */
  hasPrevious?: boolean
}

/** Rule 7: baseline polls, a hidden tab and first paint replay nothing. */
export const isBaseline = (ctx: MotionCtx) => !!ctx.baseline || !!ctx.hidden || ctx.hasPrevious === false

const finite = (n: unknown): n is number => typeof n === 'number' && Number.isFinite(n)

/**
 * Whether a KPI value should tween from `prev` to `next` (§2.12 "KPI value
 * change"). Only when the formatted text differs, never on a baseline, under
 * reduced motion, while hidden, or when the unit changes (MB → GB jumps).
 */
export function shouldTween(
  prev: number | null | undefined,
  next: number | null | undefined,
  ctx: MotionCtx & { format: (n: number) => string; unit?: (n: number) => string | null }
): boolean {
  if (!finite(prev) || !finite(next)) return false
  if (isBaseline({ ...ctx, hasPrevious: ctx.hasPrevious ?? true }) || ctx.reduced) return false
  if (ctx.format(prev) === ctx.format(next)) return false
  if (ctx.unit && ctx.unit(prev) !== ctx.unit(next)) return false
  return true
}

/** The tile tint plays only when the severity changes, so a busy Backlog tweens but never flashes. */
export function shouldTint(prevState: string | null | undefined, nextState: string | null | undefined, ctx: MotionCtx = {}): boolean {
  if (isBaseline({ ...ctx, hasPrevious: ctx.hasPrevious ?? true })) return false
  return rankOf(prevState) !== rankOf(nextState)
}

/** OK < Warning < Serious < Critical/Down. Improvements, Unknown and Maintenance are not a worsening. */
export const isWorsening = (prevState: string | null | undefined, nextState: string | null | undefined) =>
  rankOf(nextState) > rankOf(prevState)

export type FlowBucket = 'idle' | 'low' | 'medium' | 'high'

/**
 * Topology edge speed bucket from the edge's own 24 h p50/p90 (§2.12). An
 * edge with no baseline yet flows at the low speed while it carries traffic.
 */
export function flowBucket(rate: number | null | undefined, edge: { p50?: number | null; p90?: number | null } = {}): FlowBucket {
  if (!finite(rate) || rate <= 0) return 'idle'
  const p50 = finite(edge.p50) && edge.p50 > 0 ? edge.p50 : null
  const p90 = finite(edge.p90) && edge.p90 > 0 ? Math.max(edge.p90, p50 ?? 0) : null
  if (p50 === null) return 'low'
  if (rate < p50) return 'low'
  if (p90 === null || rate < p90) return 'medium'
  return 'high'
}

export type KpiDelta = { glyph: '▲' | '▼'; text: string }

/** "▲ +118" / "▼ −3" between two values, formatted like the tile; null when nothing changed. */
export function kpiDelta(prev: number | null | undefined, next: number | null | undefined, format: (n: number) => string): KpiDelta | null {
  if (!finite(prev) || !finite(next) || format(prev) === format(next)) return null
  const diff = next - prev
  return { glyph: diff > 0 ? '▲' : '▼', text: `${diff > 0 ? '+' : '−'}${format(Math.abs(diff))}` }
}

// ---------- the attention slot (rule 2) ----------

export type RingSource = 'chip' | 'attention' | 'node'
export type RingCandidate = { key: string; rank: number; source: RingSource }

/** Same-poll priority: the state chip first, then an attention item, then a topology node. */
export const RING_PRIORITY: Record<RingSource, number> = { chip: 0, attention: 1, node: 2 }

/**
 * One page-level slot for attention-grabbing rings. A ring that cannot get
 * the slot is dropped, not queued; the static state still shows. The same
 * target at the same rank rings at most once per cooldown. Components read
 * their ring token through useRingToken(), so a granted ring re-renders only
 * the element that rings.
 */
export class AttentionSlot {
  private busyUntil = 0
  private last = new Map<string, number>()
  private tokens = new Map<string, number>()
  private listeners = new Set<() => void>()

  constructor(
    private readonly ringMs = DUR.ring * 1000,
    private readonly cooldownMs = RING_COOLDOWN_MS
  ) {}

  /** Grants at most one of the candidates (by priority) and returns it, or null. */
  offer(candidates: RingCandidate[], now: number): RingCandidate | null {
    if (!candidates.length || now < this.busyUntil) return null
    const ordered = [...candidates].sort((a, b) => RING_PRIORITY[a.source] - RING_PRIORITY[b.source] || b.rank - a.rank)
    for (const c of ordered) {
      const id = `${c.key}@${c.rank}`
      const at = this.last.get(id)
      if (at !== undefined && now - at < this.cooldownMs) continue
      this.last.set(id, now)
      this.busyUntil = now + this.ringMs
      this.tokens.set(c.key, (this.tokens.get(c.key) ?? 0) + 1)
      this.listeners.forEach((fn) => fn())
      return c
    }
    return null
  }

  token = (key: string) => this.tokens.get(key) ?? 0

  subscribe = (fn: () => void) => {
    this.listeners.add(fn)
    return () => {
      this.listeners.delete(fn)
    }
  }
}

export const AttentionSlotContext = createContext<AttentionSlot | null>(null)

/** The page's attention slot (null outside the Operations shell, for example in tests). */
export const useAttentionSlot = () => useContext(AttentionSlotContext)

const NO_SLOT_SUBSCRIBE = () => () => {}

/** Increments each time the slot grants a ring to `key`; 0 means "never rang". */
export function useRingToken(key: string): number {
  const slot = useAttentionSlot()
  return useSyncExternalStore(
    slot ? slot.subscribe : NO_SLOT_SUBSCRIBE,
    () => (slot ? slot.token(key) : 0),
    () => 0
  )
}

/**
 * Plays one ring on the element when `token` increases: a ::after outline that
 * grows by at most 8 px per side. The per-axis scale comes from the element's
 * size via ResizeObserver, never from a layout read in an animation frame.
 * Two alternating classes restart the animation without a forced reflow.
 */
export function useRing<T extends HTMLElement>(token: number) {
  const ref = useRef<T | null>(null)
  const { reduced } = useOpsMotion()
  const seen = useRef(token)

  useEffect(() => {
    const el = ref.current
    if (!el || typeof ResizeObserver === 'undefined') return
    const ro = new ResizeObserver(([entry]) => {
      const box = entry.contentRect
      el.style.setProperty('--ring-sx', (1 + 16 / Math.max(box.width, 1)).toFixed(3))
      el.style.setProperty('--ring-sy', (1 + 16 / Math.max(box.height, 1)).toFixed(3))
    })
    ro.observe(el)
    return () => ro.disconnect()
  }, [])

  useEffect(() => {
    const el = ref.current
    const fresh = token > seen.current
    seen.current = token
    if (!el || !fresh || reduced) return
    const odd = token % 2 === 1
    el.classList.remove(odd ? 'ops-ring-b' : 'ops-ring-a')
    el.classList.add(odd ? 'ops-ring-a' : 'ops-ring-b')
    const timer = setTimeout(() => el.classList.remove('ops-ring-a', 'ops-ring-b'), DUR.ring * 1000 + 50)
    return () => clearTimeout(timer)
  }, [token, reduced])

  return ref
}

// ---------- KPI numbers ----------

export type Seen = {
  text: string
  value: number | null
  from: number | null
  tween: boolean
  token: number
  delta: (KpiDelta & { since: number | null }) | null
}

type NumberCtx = {
  format: (n: number) => string
  unit?: (n: number) => string | null
  baseline?: boolean
  since?: number | null
  reduced: boolean
  hidden: boolean
}

/** What useAnimatedNumber records for a change of the formatted value (pure, for tests). */
export function nextSeen(seen: Seen, value: number | null, ctx: NumberCtx): Seen {
  const text = value === null ? '–' : ctx.format(value)
  if (text === seen.text) return seen
  const tween = shouldTween(seen.value, value, ctx)
  const d = isBaseline(ctx) ? null : kpiDelta(seen.value, value, ctx.format)
  return { text, value, from: seen.value, tween, token: seen.token + 1, delta: d ? { ...d, since: ctx.since ?? null } : null }
}

type Animate = (
  from: number,
  to: number,
  opts: { onUpdate: (n: number) => void; onComplete: () => void }
) => { stop: () => void }

const runTween: Animate = (from, to, handlers) => animate(from, to, { ...TWEEN.value, ...handlers })

/**
 * The DOM side of useAnimatedNumber for one change (`seen.token`): tween the
 * hook's text node, or show `text` at once. Returns the cleanup, which only
 * stops the tween: React commits the next value's text into the span before
 * the parent's layout cleanup runs, so writing the old target there would
 * overwrite the new value.
 */
export function paintNumber(
  el: { textContent: string | null },
  seen: Pick<Seen, 'tween' | 'from' | 'value'>,
  text: string,
  format: (n: number) => string,
  run: Animate = runTween
): (() => void) | undefined {
  if (!seen.tween || seen.from === null || seen.value === null) {
    // A jump. An earlier tween wrote frames into the node: show the current value.
    if (el.textContent !== text) el.textContent = text
    return undefined
  }
  const target = seen.value
  el.textContent = format(seen.from)
  const controls = run(seen.from, target, {
    onUpdate: (n) => {
      el.textContent = format(n)
    },
    onComplete: () => {
      el.textContent = format(target)
    }
  })
  return () => controls.stop()
}

export type AnimatedNumber = {
  /** The final formatted value (first paint, and what the tween lands on). */
  text: string
  /** The same final value for screen readers (an sr-only span). */
  srText: string
  /** "▲ +118" since the previous value; cleared after DELTA_HOLD_MS or the next change. */
  delta: (KpiDelta & { since: number | null }) | null
}

/**
 * A KPI number that tweens from the old value to the new one when its
 * formatted text changes (§2.12). Values are final on first paint: there is
 * no count-up intro, unlike Token Insights' Counter. The tween writes into a
 * text node the hook owns, so neither a per-frame re-render nor a poll
 * re-render overwrites it mid-flight. Jumps under reduced motion, while
 * hidden, on a baseline poll and when the unit changes. `since` is when the
 * previous value was last confirmed (the previous poll), for the delta line.
 */
export function useAnimatedNumber(
  /** An aria-hidden span the caller renders `text` into; the hook writes tween frames into it. */
  ref: React.RefObject<HTMLSpanElement | null>,
  value: number | null | undefined,
  opts: { format: (n: number) => string; unit?: (n: number) => string | null; baseline?: boolean; since?: number | null }
): AnimatedNumber {
  const { reduced, hidden } = useOpsMotion()
  const v = finite(value) ? value : null
  const text = v === null ? '–' : opts.format(v)

  const [seen, setSeen] = useState<Seen>(() => ({ text, value: v, from: null, tween: false, token: 0, delta: null }))
  if (seen.text !== text) {
    // React's "state from the previous render" pattern: decide once per change, during render.
    setSeen(nextSeen(seen, v, { ...opts, reduced, hidden }))
  }

  useLayoutEffect(
    () => (ref.current ? paintNumber(ref.current, seen, text, opts.format) : undefined),
    // Keyed on the change token: a poll re-render with the same value must not restart the tween.
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [seen.token]
  )

  // The delta line expires after 15 min (WCAG-friendly text that does not linger forever).
  const [expired, setExpired] = useState(-1)
  useEffect(() => {
    if (!seen.delta) return
    const token = seen.token
    const timer = setTimeout(() => setExpired(token), DELTA_HOLD_MS)
    return () => clearTimeout(timer)
  }, [seen.delta, seen.token])

  return { text, srText: text, delta: expired === seen.token ? null : seen.delta }
}

/**
 * The KPI tint: on a severity change (not on a baseline) the element takes
 * the --ops-flash background for 1 s, then fades over 300 ms. It writes a
 * class directly, so it costs no React commit. Under reduced motion the
 * kill switch removes the fade, leaving the 1 s tint.
 */
export function useTint<T extends HTMLElement>(state: string | null | undefined, baseline: boolean | undefined) {
  const ref = useRef<T | null>(null)
  const previous = useRef(state)
  const { hidden } = useOpsMotion()
  useEffect(() => {
    const before = previous.current
    previous.current = state
    const el = ref.current
    if (!el || !shouldTint(before, state, { baseline, hidden })) return
    el.classList.add('ops-tint')
    const timer = setTimeout(() => el.classList.remove('ops-tint'), TINT_HOLD_MS)
    return () => {
      clearTimeout(timer)
      el.classList.remove('ops-tint')
    }
  }, [state, baseline, hidden])
  return ref
}

/**
 * Phase offset for a breathing dot: `document.timeline.currentTime % 2000`
 * taken when the dot mounts, so rows added later breathe in step with the
 * others instead of flickering out of phase.
 */
export function breathePhase(): number {
  if (typeof document === 'undefined') return 0
  const t = (document as Document & { timeline?: { currentTime?: number | null } }).timeline?.currentTime
  return typeof t === 'number' && Number.isFinite(t) ? Math.round(t % (DUR.breathe * 1000)) : 0
}
