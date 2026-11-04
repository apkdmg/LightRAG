import { describe, expect, test } from 'bun:test'
import { renderToString } from 'react-dom/server'
import { HIDDEN, REVEAL_SECONDS, RevealState, RevealTimer, RevealTimers } from './reveal'

// The stores behind these components read localStorage as they load.
const memory = new Map<string, string>()
Object.defineProperty(globalThis, 'localStorage', {
  value: {
    getItem: (k: string) => memory.get(k) ?? null,
    setItem: (k: string, v: string) => void memory.set(k, v),
    removeItem: (k: string) => void memory.delete(k)
  },
  configurable: true
})
const { RevealedValue } = await import('./components/MaskedValue')

/** Fake timers: intervals fire only when the test advances the clock. */
class FakeTimers implements RevealTimers {
  now = 0
  private next = 1
  private intervals = new Map<number, { fn: () => void; ms: number; at: number }>()

  setInterval = (fn: () => void, ms: number) => {
    const id = this.next++
    this.intervals.set(id, { fn, ms, at: this.now + ms })
    return id
  }

  clearInterval = (handle: unknown) => {
    this.intervals.delete(handle as number)
  }

  get active() {
    return this.intervals.size
  }

  advance(ms: number) {
    const end = this.now + ms
    for (;;) {
      let due: [number, { fn: () => void; ms: number; at: number }] | null = null
      for (const entry of this.intervals) if (entry[1].at <= end && (!due || entry[1].at < due[1].at)) due = entry
      if (!due) break
      this.now = due[1].at
      due[1].at += due[1].ms
      due[1].fn()
    }
    this.now = end
  }
}

const setup = () => {
  const timers = new FakeTimers()
  const states: RevealState[] = []
  const reveal = new RevealTimer((s) => states.push(s), timers)
  return { timers, states, reveal }
}

describe('RevealTimer (§2.9: revealed values hide after 60 s unless kept)', () => {
  test('shows for 60 s, counts down once a second, then hides by itself', () => {
    const { timers, states, reveal } = setup()
    reveal.start()
    expect(reveal.current).toEqual({ visible: true, remaining: REVEAL_SECONDS, kept: false })
    timers.advance(59_000)
    expect(reveal.current).toEqual({ visible: true, remaining: 1, kept: false })
    timers.advance(1_000)
    expect(reveal.current).toEqual(HIDDEN)
    expect(states[states.length - 1]).toEqual(HIDDEN)
    // The countdown stopped: nothing ticks after the value is gone.
    expect(timers.active).toBe(0)
    const count = states.length
    timers.advance(10_000)
    expect(states.length).toBe(count)
  })

  test('Keep visible stops the countdown; the value stays until hidden', () => {
    const { timers, reveal } = setup()
    reveal.start()
    timers.advance(30_000)
    reveal.keep()
    expect(reveal.current).toEqual({ visible: true, remaining: 30, kept: true })
    expect(timers.active).toBe(0)
    timers.advance(10 * 60_000)
    expect(reveal.current.visible).toBe(true)
    reveal.hide()
    expect(reveal.current).toEqual(HIDDEN)
  })

  test('Hide drops the value at once', () => {
    const { timers, reveal } = setup()
    reveal.start()
    timers.advance(5_000)
    reveal.hide()
    expect(reveal.current).toEqual(HIDDEN)
    expect(timers.active).toBe(0)
  })

  test('a second reveal restarts the full 60 s', () => {
    const { timers, reveal } = setup()
    reveal.start()
    timers.advance(50_000)
    reveal.start()
    expect(reveal.current.remaining).toBe(REVEAL_SECONDS)
    expect(timers.active).toBe(1)
    timers.advance(59_000)
    expect(reveal.current.visible).toBe(true)
    timers.advance(1_000)
    expect(reveal.current.visible).toBe(false)
  })

  test('unmount stops counting without another state change', () => {
    const { timers, states, reveal } = setup()
    reveal.start()
    const count = states.length
    reveal.dispose()
    expect(timers.active).toBe(0)
    timers.advance(120_000)
    expect(states.length).toBe(count)
  })

  test('Keep visible after hiding does nothing', () => {
    const { reveal, states } = setup()
    reveal.keep()
    expect(states).toEqual([])
    expect(reveal.current).toEqual(HIDDEN)
  })
})

describe('RevealedValue', () => {
  test('counting down: the value, "Hides in", Keep visible and Hide', () => {
    const html = renderToString(<RevealedValue value="Report_2024.pdf" state={{ visible: true, remaining: 42, kept: false }} onKeep={() => {}} onHide={() => {}} />)
    expect(html).toContain('Report_2024.pdf')
    expect(html).toContain('operations.reveal.hidesIn')
    expect(html).toContain('operations.reveal.keep')
    expect(html).toContain('operations.reveal.hide')
  })

  test('kept: no countdown and no Keep visible, Hide stays', () => {
    const html = renderToString(<RevealedValue value="Report_2024.pdf" state={{ visible: true, remaining: 42, kept: true }} onKeep={() => {}} onHide={() => {}} />)
    expect(html).toContain('operations.reveal.kept')
    expect(html).not.toContain('operations.reveal.hidesIn')
    expect(html).not.toContain('operations.reveal.keep<')
    expect(html).toContain('operations.reveal.hide')
  })
})
