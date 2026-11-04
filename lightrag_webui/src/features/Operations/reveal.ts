/**
 * The countdown behind a revealed value (§2.9, §3.9): a revealed file name or
 * error hides itself after 60 s unless the viewer chose "Keep visible". The
 * value itself lives only in the component that asked for it (never in a
 * store, the URL or browser storage); this controller only says when to
 * drop it. Timers are injected so the countdown is tested with fake timers.
 */

export const REVEAL_SECONDS = 60

export type RevealTimers = {
  setInterval: (fn: () => void, ms: number) => unknown
  clearInterval: (handle: unknown) => void
}

const browserTimers: RevealTimers = {
  setInterval: (fn, ms) => globalThis.setInterval(fn, ms),
  clearInterval: (handle) => globalThis.clearInterval(handle as ReturnType<typeof setInterval>)
}

export type RevealState = { visible: boolean; remaining: number; kept: boolean }

export const HIDDEN: RevealState = { visible: false, remaining: 0, kept: false }

/**
 * One reveal at a time: `start()` shows for 60 s, `keep()` stops the
 * countdown (the value stays until `hide()` or unmount), `hide()` drops it
 * now. `onChange` gets every state, so the component re-renders once a
 * second while counting and never after it hides.
 */
export class RevealTimer {
  private handle: unknown = null
  private state: RevealState = HIDDEN

  constructor(
    private readonly onChange: (state: RevealState) => void,
    private readonly timers: RevealTimers = browserTimers,
    private readonly seconds: number = REVEAL_SECONDS
  ) {}

  get current(): RevealState {
    return this.state
  }

  start() {
    this.stop()
    this.set({ visible: true, remaining: this.seconds, kept: false })
    this.handle = this.timers.setInterval(() => this.tick(), 1000)
  }

  keep() {
    if (!this.state.visible || this.state.kept) return
    this.stop()
    this.set({ ...this.state, kept: true })
  }

  hide() {
    this.stop()
    if (this.state.visible) this.set(HIDDEN)
  }

  /** Unmount: stop counting without a last state change. */
  dispose() {
    this.stop()
    this.state = HIDDEN
  }

  private tick() {
    const remaining = this.state.remaining - 1
    if (remaining <= 0) this.hide()
    else this.set({ ...this.state, remaining })
  }

  private stop() {
    if (this.handle !== null) this.timers.clearInterval(this.handle)
    this.handle = null
  }

  private set(state: RevealState) {
    this.state = state
    this.onChange(state)
  }
}
