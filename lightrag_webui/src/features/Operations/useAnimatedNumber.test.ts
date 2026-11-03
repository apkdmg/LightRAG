import { describe, expect, test } from 'bun:test'
import { fmtCount } from './format'
import { Seen, nextSeen, paintNumber } from './motion'

const live = { format: fmtCount, reduced: false, hidden: false }
const first = (value: number | null): Seen => ({
  text: value === null ? '–' : fmtCount(value),
  value,
  from: null,
  tween: false,
  token: 0,
  delta: null
})

/** A stand-in for Motion's animate(): frames are driven by the test. */
function fakeAnimate() {
  const runs: { from: number; to: number; frame: (n: number) => void; done: () => void; stopped: boolean }[] = []
  const run = (from: number, to: number, h: { onUpdate: (n: number) => void; onComplete: () => void }) => {
    const entry = { from, to, frame: h.onUpdate, done: h.onComplete, stopped: false }
    runs.push(entry)
    return { stop: () => void (entry.stopped = true) }
  }
  return { run, runs }
}

describe('nextSeen (useAnimatedNumber decisions, §2.12)', () => {
  test('a changed value tweens and carries a delta with the previous poll time', () => {
    const s = nextSeen(first(10), 20, { ...live, since: 1000 })
    expect(s).toMatchObject({ text: '20', value: 20, from: 10, tween: true, token: 1 })
    expect(s.delta).toEqual({ glyph: '▲', text: '+10', since: 1000 })
  })

  test('jumps under reduced motion, while hidden and on a baseline poll', () => {
    expect(nextSeen(first(10), 20, { ...live, reduced: true }).tween).toBe(false)
    expect(nextSeen(first(10), 20, { ...live, hidden: true })).toMatchObject({ tween: false, delta: null })
    expect(nextSeen(first(10), 20, { ...live, baseline: true })).toMatchObject({ tween: false, delta: null })
  })

  test('the same formatted text is no change at all', () => {
    const s = first(10)
    expect(nextSeen(s, 10.2, live)).toBe(s)
  })

  test('a value that becomes unknown jumps to the dash', () => {
    expect(nextSeen(first(10), null, live)).toMatchObject({ text: '–', tween: false, value: null })
  })
})

describe('paintNumber (the text node the hook owns)', () => {
  test('a tween writes frames and lands on the target', () => {
    const el = { textContent: '20' as string | null }
    const { run, runs } = fakeAnimate()
    paintNumber(el, { tween: true, from: 10, value: 20 }, '20', fmtCount, run)
    expect(el.textContent).toBe('10')
    runs[0].frame(15)
    expect(el.textContent).toBe('15')
    runs[0].done()
    expect(el.textContent).toBe('20')
  })

  test('a jump after an interrupted tween shows the new value, not the old target', () => {
    // 10 -> 20 tweens; the next poll (a baseline) jumps to 35 mid-flight.
    const el = { textContent: '20' as string | null }
    const { run, runs } = fakeAnimate()
    const cleanup = paintNumber(el, { tween: true, from: 10, value: 20 }, '20', fmtCount, run)
    runs[0].frame(14)
    // React commits the new text into the span before the layout cleanup runs.
    el.textContent = '35'
    cleanup?.()
    expect(runs[0].stopped).toBe(true)
    expect(el.textContent).toBe('35')
    paintNumber(el, { tween: false, from: 20, value: 35 }, '35', fmtCount, run)
    expect(el.textContent).toBe('35')
    expect(runs).toHaveLength(1)
  })

  test('a jump repairs a node an earlier tween left behind', () => {
    const el = { textContent: '14' as string | null }
    expect(paintNumber(el, { tween: false, from: null, value: 35 }, '35', fmtCount)).toBeUndefined()
    expect(el.textContent).toBe('35')
    paintNumber(el, { tween: false, from: 35, value: null }, '–', fmtCount)
    expect(el.textContent).toBe('–')
  })
})
