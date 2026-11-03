import { useReducedMotion } from 'motion/react'
import { useDocumentHidden } from './poll'
import { useOpsPrefs } from './prefs'

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
  panel: { type: 'tween', duration: DUR.panel, ease: EASE.out },
  panelExit: { type: 'tween', duration: DUR.panel, ease: EASE.in },
  value: { type: 'tween', duration: DUR.value, ease: EASE.out }
} as const

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
