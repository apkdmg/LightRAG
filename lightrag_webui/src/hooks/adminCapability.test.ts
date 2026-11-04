import { describe, expect, test } from 'bun:test'
import { AuthSnapshot, SESSION_CACHE_KEY, classifyProbe, createCapabilityStore, decide, probeRetryDelay } from './adminCapability'

const auth = (patch: Partial<AuthSnapshot> = {}): AuthSnapshot => ({
  isAuthenticated: true,
  isGuestMode: false,
  isSSOMode: true,
  role: null,
  username: 'opso',
  ...patch
})

const memoryStorage = () => {
  const data = new Map<string, string>()
  return {
    data,
    getItem: (k: string) => data.get(k) ?? null,
    setItem: (k: string, v: string) => void data.set(k, v),
    removeItem: (k: string) => void data.delete(k)
  }
}

/** Manual timers: tests decide when a scheduled retry fires. */
const manualTimers = () => {
  const pending: { fn: () => void; ms: number; id: number }[] = []
  let next = 1
  return {
    pending,
    setTimeout: (fn: () => void, ms: number) => {
      const id = next++
      pending.push({ fn, ms, id })
      return id
    },
    clearTimeout: (id: unknown) => {
      const i = pending.findIndex((p) => p.id === id)
      if (i >= 0) pending.splice(i, 1)
    },
    fire: () => pending.shift()?.fn()
  }
}

const flush = () => new Promise((resolve) => setTimeout(resolve, 0))
const httpError = (status: number) => Object.assign(new Error(String(status)), { status })

describe('decide (what the browser knows without asking)', () => {
  const cases: [string, Partial<AuthSnapshot>, string][] = [
    ['signed out', { isAuthenticated: false, role: 'admin' }, 'not_admin'],
    ['guest', { isGuestMode: true, role: 'guest' }, 'not_admin'],
    ['admin from the JWT', { isSSOMode: false, role: 'admin' }, 'admin'],
    ['SSO admin from login metadata', { role: 'admin' }, 'admin'],
    ['user role', { isSSOMode: false, role: 'user' }, 'not_admin'],
    ['SSO user role', { role: 'user' }, 'not_admin'],
    ['SSO without a role', { role: null }, 'probe'],
    ['password login without a role claim', { isSSOMode: false, role: null }, 'not_admin']
  ]
  for (const [name, patch, expected] of cases) {
    test(name, () => expect(decide(auth(patch))).toBe(expected as any))
  }
})

describe('classifyProbe (only 200 and 403 are definite)', () => {
  test.each([
    [200, true, true],
    [204, true, true],
    [403, false, true],
    [401, null, false],
    [404, null, false],
    [429, null, false],
    [500, null, false],
    [502, null, false],
    [503, null, false]
  ])('status %p → %p (cached: %p)', (status, value, cache) => {
    expect(classifyProbe(status)).toEqual({ value, cache } as any)
  })

  test('a network error or timeout (no status) is not cached', () => {
    expect(classifyProbe(null)).toEqual({ value: null, cache: false })
    expect(classifyProbe(undefined)).toEqual({ value: null, cache: false })
  })
})

describe('probeRetryDelay', () => {
  test('backs off exponentially up to five minutes', () => {
    expect(probeRetryDelay(0)).toBe(2_000)
    expect(probeRetryDelay(1)).toBe(4_000)
    expect(probeRetryDelay(5)).toBe(64_000)
    expect(probeRetryDelay(8)).toBe(300_000)
    expect(probeRetryDelay(100)).toBe(300_000)
  })
})

describe('createCapabilityStore', () => {
  test('known roles never probe', async () => {
    let calls = 0
    const store = createCapabilityStore({ probe: async () => ((calls += 1), 200) })
    store.ensure(auth({ role: 'admin' }))
    store.ensure(auth({ role: 'user' }))
    await flush()
    expect(calls).toBe(0)
    expect(store.peek(auth({ role: 'admin' }))).toBe(true)
    expect(store.peek(auth({ role: 'user' }))).toBe(false)
  })

  test('a 200 is cached in memory and sessionStorage; the probe runs once', async () => {
    let calls = 0
    const storage = memoryStorage()
    const store = createCapabilityStore({ probe: async () => ((calls += 1), 200), storage: () => storage })
    const off = store.subscribe(() => {})
    expect(store.peek(auth())).toBeNull()
    store.ensure(auth())
    store.ensure(auth())
    await flush()
    expect(calls).toBe(1)
    expect(store.peek(auth())).toBe(true)
    expect(JSON.parse(storage.data.get(SESSION_CACHE_KEY) as string)).toEqual({ user: 'opso', admin: true })
    store.ensure(auth())
    await flush()
    expect(calls).toBe(1)
    off()
  })

  test('a 403 is cached as false', async () => {
    const storage = memoryStorage()
    const store = createCapabilityStore({ probe: async () => Promise.reject(httpError(403)), storage: () => storage })
    const off = store.subscribe(() => {})
    store.ensure(auth())
    await flush()
    expect(store.peek(auth())).toBe(false)
    expect(storage.data.has(SESSION_CACHE_KEY)).toBe(true)
    off()
  })

  test('a 5xx is not cached and is retried with backoff', async () => {
    const timers = manualTimers()
    const storage = memoryStorage()
    const answers = [httpError(502), httpError(500)]
    let calls = 0
    const store = createCapabilityStore({
      probe: async () => {
        calls += 1
        const next = answers.shift()
        if (next) throw next
        return 200
      },
      storage: () => storage,
      timers
    })
    const off = store.subscribe(() => {})
    store.ensure(auth())
    await flush()
    expect(store.peek(auth())).toBeNull()
    expect(storage.data.has(SESSION_CACHE_KEY)).toBe(false)
    expect(timers.pending.map((p) => p.ms)).toEqual([2_000])
    // A scheduled retry is not doubled by another ensure().
    store.ensure(auth())
    await flush()
    expect(calls).toBe(1)
    timers.fire()
    await flush()
    expect(calls).toBe(2)
    expect(timers.pending.map((p) => p.ms)).toEqual([4_000])
    timers.fire()
    await flush()
    expect(calls).toBe(3)
    expect(store.peek(auth())).toBe(true)
    expect(timers.pending).toHaveLength(0)
    off()
  })

  test('a network error is retried and not cached', async () => {
    const timers = manualTimers()
    const store = createCapabilityStore({ probe: async () => Promise.reject(new Error('Network Error')), timers })
    const off = store.subscribe(() => {})
    store.ensure(auth())
    await flush()
    expect(store.peek(auth())).toBeNull()
    expect(timers.pending).toHaveLength(1)
    off()
    // Nobody is listening any more: the retry is dropped.
    expect(timers.pending).toHaveLength(0)
  })

  test('answers are per user, and reset forgets them', async () => {
    const storage = memoryStorage()
    const store = createCapabilityStore({ probe: async () => 200, storage: () => storage })
    const off = store.subscribe(() => {})
    store.ensure(auth())
    await flush()
    expect(store.peek(auth())).toBe(true)
    expect(store.peek(auth({ username: 'someone_else' }))).toBeNull()
    store.reset()
    expect(store.peek(auth())).toBeNull()
    expect(storage.data.has(SESSION_CACHE_KEY)).toBe(false)
    off()
  })

  test('a probe answer that arrives after reset is ignored', async () => {
    let resolveProbe: (status: number) => void = () => {}
    const store = createCapabilityStore({ probe: () => new Promise<number>((resolve) => (resolveProbe = resolve)) })
    const off = store.subscribe(() => {})
    store.ensure(auth())
    store.reset()
    resolveProbe(200)
    await flush()
    expect(store.peek(auth())).toBeNull()
    off()
  })

  test('a blocked sessionStorage still works from memory', async () => {
    const store = createCapabilityStore({
      probe: async () => 200,
      storage: () => ({
        getItem: () => {
          throw new Error('blocked')
        },
        setItem: () => {
          throw new Error('blocked')
        },
        removeItem: () => {
          throw new Error('blocked')
        }
      })
    })
    const off = store.subscribe(() => {})
    store.ensure(auth())
    await flush()
    expect(store.peek(auth())).toBe(true)
    off()
  })

  test('a session answer from an earlier page load is reused without probing', async () => {
    const storage = memoryStorage()
    storage.setItem(SESSION_CACHE_KEY, JSON.stringify({ user: 'opso', admin: true }))
    let calls = 0
    const store = createCapabilityStore({ probe: async () => ((calls += 1), 200), storage: () => storage })
    expect(store.peek(auth())).toBe(true)
    store.ensure(auth())
    await flush()
    expect(calls).toBe(0)
    // Another user's stored answer is not reused.
    expect(store.peek(auth({ username: 'other' }))).toBeNull()
  })
})
