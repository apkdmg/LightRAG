/**
 * Who may see the admin tabs (Insights, Operations).
 *
 * A JWT carries the role, so password logins know it at once. SSO logins keep
 * the JWT in an HTTP-only cookie and learn the role from the login metadata,
 * which can be missing (role null). Then the browser asks the server once with
 * GET /admin/ops/status: 200 means admin, 403 means not. Only those two answers
 * are remembered for the session; anything else (5xx, timeout, a server that
 * does not have the route yet) is retried with backoff and never cached, so a
 * transient outage cannot hide the tab for the rest of the session. The server
 * stays the authority: every admin request is checked again there.
 *
 * Pure logic lives here so it can be tested without React or axios.
 */

export type Capability = boolean | null // null: not known yet (probe pending)

export type AuthSnapshot = {
  isAuthenticated: boolean
  isGuestMode: boolean
  isSSOMode: boolean
  role: string | null
  username: string | null
}

export type Decision = 'admin' | 'not_admin' | 'probe'

/** What the browser already knows, before asking the server. */
export function decide(auth: AuthSnapshot): Decision {
  if (!auth.isAuthenticated || auth.isGuestMode) return 'not_admin'
  if (auth.role === 'admin') return 'admin'
  if (auth.role !== null) return 'not_admin'
  // Password and API-key sessions always carry the role in the JWT; only SSO can lack it.
  return auth.isSSOMode ? 'probe' : 'not_admin'
}

export type ProbeOutcome = { value: boolean; cache: true } | { value: null; cache: false }

/** Maps a probe result (an HTTP status, or no status for a network error) to an answer. */
export function classifyProbe(status: number | null | undefined): ProbeOutcome {
  if (status !== null && status !== undefined && status >= 200 && status < 300) return { value: true, cache: true }
  if (status === 403) return { value: false, cache: true }
  return { value: null, cache: false }
}

export const PROBE_BASE_DELAY_MS = 2_000
export const PROBE_MAX_DELAY_MS = 300_000

/** Delay before retry number `attempt` (0-based): 2 s, 4 s, 8 s … capped at 5 min. */
export function probeRetryDelay(attempt: number): number {
  const exp = Math.min(Math.max(0, Math.floor(attempt)), 20)
  return Math.min(PROBE_BASE_DELAY_MS * 2 ** exp, PROBE_MAX_DELAY_MS)
}

export const SESSION_CACHE_KEY = 'LIGHTRAG-OPS-ADMIN'

type StorageLike = Pick<Storage, 'getItem' | 'setItem' | 'removeItem'>

type Timers = {
  setTimeout: (fn: () => void, ms: number) => unknown
  clearTimeout: (handle: unknown) => void
}

export type CapabilityStoreOptions = {
  /** Resolves with the HTTP status; rejects with an error carrying `status` (or none for network errors). */
  probe: () => Promise<number>
  storage?: () => StorageLike | null
  timers?: Timers
}

const defaultTimers: Timers = {
  setTimeout: (fn, ms) => globalThis.setTimeout(fn, ms),
  clearTimeout: (handle) => globalThis.clearTimeout(handle as ReturnType<typeof setTimeout>)
}

/**
 * Holds the probe answer per user for this session. One store is shared by
 * every component, so the header and the tab panes never probe twice.
 */
export function createCapabilityStore({ probe, storage = () => null, timers = defaultTimers }: CapabilityStoreOptions) {
  const memory = new Map<string, boolean>()
  const listeners = new Set<() => void>()
  let inFlight: string | null = null
  let retryTimer: unknown = null
  let retryUser: string | null = null
  let attempt = 0
  let generation = 0

  const notify = () => listeners.forEach((fn) => fn())

  const readSession = (user: string): boolean | null => {
    try {
      const raw = storage()?.getItem(SESSION_CACHE_KEY)
      if (!raw) return null
      const parsed = JSON.parse(raw)
      return parsed && parsed.user === user && typeof parsed.admin === 'boolean' ? parsed.admin : null
    } catch {
      return null
    }
  }

  const writeSession = (user: string, admin: boolean) => {
    try {
      storage()?.setItem(SESSION_CACHE_KEY, JSON.stringify({ user, admin }))
    } catch {
      // Storage blocked: the in-memory answer still holds for this page.
    }
  }

  const cancelRetry = () => {
    if (retryTimer !== null) timers.clearTimeout(retryTimer)
    retryTimer = null
    retryUser = null
  }

  const cached = (user: string): boolean | null => {
    const hit = memory.get(user)
    if (hit !== undefined) return hit
    const stored = readSession(user)
    if (stored !== null) memory.set(user, stored)
    return stored
  }

  const run = (user: string) => {
    if (inFlight === user) return
    inFlight = user
    const gen = generation
    const settle = (status: number | null) => {
      if (gen !== generation) return
      inFlight = null
      const outcome = classifyProbe(status)
      if (outcome.cache) {
        attempt = 0
        memory.set(user, outcome.value)
        writeSession(user, outcome.value)
        notify()
        return
      }
      // Not a definite answer: try again later while someone still needs it.
      if (listeners.size === 0) return
      const delay = probeRetryDelay(attempt)
      attempt += 1
      cancelRetry()
      retryUser = user
      retryTimer = timers.setTimeout(() => {
        retryTimer = null
        retryUser = null
        if (listeners.size > 0) run(user)
      }, delay)
    }
    probe().then(
      (status) => settle(status),
      (err) => settle(typeof err?.status === 'number' ? err.status : null)
    )
  }

  return {
    /** The answer for this user: true/false when known, null while it is being asked. No side effects. */
    peek(auth: AuthSnapshot): Capability {
      const decision = decide(auth)
      if (decision === 'admin') return true
      if (decision === 'not_admin') return false
      return cached(auth.username ?? '')
    },
    /** Starts the probe when the answer is unknown and none is running or scheduled. */
    ensure(auth: AuthSnapshot) {
      if (decide(auth) !== 'probe') return
      const user = auth.username ?? ''
      if (cached(user) !== null || inFlight === user || retryUser === user) return
      run(user)
    },
    subscribe(fn: () => void) {
      listeners.add(fn)
      return () => {
        listeners.delete(fn)
        if (listeners.size === 0) cancelRetry()
      }
    },
    /** Forget every answer (logout): the next user starts fresh. */
    reset() {
      generation += 1
      memory.clear()
      inFlight = null
      attempt = 0
      cancelRetry()
      try {
        storage()?.removeItem(SESSION_CACHE_KEY)
      } catch {
        // Storage blocked: nothing was stored either.
      }
      notify()
    }
  }
}

export type CapabilityStore = ReturnType<typeof createCapabilityStore>
