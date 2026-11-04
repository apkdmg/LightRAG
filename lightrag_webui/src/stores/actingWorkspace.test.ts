import { beforeEach, describe, expect, test } from 'bun:test'

// The store reads sessionStorage as it loads; the mock is in place before the import.
const memory = new Map<string, string>()
const storage = {
  getItem: (k: string) => memory.get(k) ?? null,
  setItem: (k: string, v: string) => void memory.set(k, v),
  removeItem: (k: string) => void memory.delete(k)
}
Object.defineProperty(globalThis, 'sessionStorage', { value: storage, configurable: true })

const { ACTING_SESSION_KEY, actingHeaderFor, cleanWorkspace, isDocumentsCall, mayAct, readActing, useActingStore, writeActing } = await import('./actingWorkspace')

const admin = { isAuthenticated: true, username: 'ops@unimas.my', role: 'admin', isGuestMode: false }
const acting = { workspace: 'fsktm', user: 'ops@unimas.my', since: 1 }

beforeEach(() => {
  memory.clear()
  useActingStore.getState().exit()
})

describe('which calls carry X-Target-Workspace (OQ-13)', () => {
  test('every Documents API path, and nothing else', () => {
    for (const url of ['/documents', '/documents/paginated', '/documents/upload', '/documents?x=1', '/documents/track_status/abc', 'http://h:9621/documents/scan']) {
      expect(isDocumentsCall(url)).toBe(true)
    }
    for (const url of ['/documentsx', '/query', '/graphs', '/admin/ops/status', '/workspace/provider-config', '/health', '', null, undefined, 'http://h/query']) {
      expect(isDocumentsCall(url)).toBe(false)
    }
  })

  test('only for the admin who entered the mode', () => {
    expect(actingHeaderFor('/documents/paginated', acting, admin)).toEqual({ 'X-Target-Workspace': 'fsktm' })
    // SSO admins may carry no role in the browser; the mode was entered after the server said admin.
    expect(actingHeaderFor('/documents', acting, { ...admin, role: null })).toEqual({ 'X-Target-Workspace': 'fsktm' })
    expect(actingHeaderFor('/documents', acting, { ...admin, role: 'user' })).toBeNull()
    expect(actingHeaderFor('/documents', acting, { ...admin, username: 'someone.else' })).toBeNull()
    expect(actingHeaderFor('/documents', acting, { ...admin, isAuthenticated: false })).toBeNull()
    expect(actingHeaderFor('/documents', acting, { ...admin, isGuestMode: true })).toBeNull()
    expect(actingHeaderFor('/documents', null, admin)).toBeNull()
    expect(actingHeaderFor('/query', acting, admin)).toBeNull()
    expect(mayAct(acting, admin)).toBe(true)
  })

  test('workspace ids must be header-safe', () => {
    expect(cleanWorkspace(' fsktm ')).toBe('fsktm')
    expect(cleanWorkspace('a b')).toBeNull()
    expect(cleanWorkspace('bad\nid')).toBeNull()
    expect(cleanWorkspace('ws中')).toBeNull()
    expect(cleanWorkspace('x'.repeat(257))).toBeNull()
    expect(cleanWorkspace('')).toBeNull()
    expect(cleanWorkspace(42)).toBeNull()
  })
})

describe('entering and leaving the mode', () => {
  test('non-admins can never enter it', () => {
    const { start } = useActingStore.getState()
    expect(start('fsktm', { admin: false, user: 'u' })).toBe(false)
    expect(start('fsktm', { admin: null, user: 'u' })).toBe(false)
    expect(useActingStore.getState().acting).toBeNull()
    expect(memory.has(ACTING_SESSION_KEY)).toBe(false)
  })

  test('an admin enters it; it is kept for the tab and Exit clears it', () => {
    expect(useActingStore.getState().start('fsktm', { admin: true, user: 'ops@unimas.my' })).toBe(true)
    expect(useActingStore.getState().acting).toMatchObject({ workspace: 'fsktm', user: 'ops@unimas.my' })
    expect(readActing(storage)).toMatchObject({ workspace: 'fsktm', user: 'ops@unimas.my' })
    useActingStore.getState().exit()
    expect(useActingStore.getState().acting).toBeNull()
    expect(memory.has(ACTING_SESSION_KEY)).toBe(false)
  })

  test('an id the server could not take is refused', () => {
    expect(useActingStore.getState().start('two words', { admin: true, user: 'a' })).toBe(false)
    expect(useActingStore.getState().acting).toBeNull()
  })

  test('a stored mode is read back only when it is well formed', () => {
    writeActing(acting, storage)
    expect(readActing(storage)).toEqual(acting)
    memory.set(ACTING_SESSION_KEY, '{"workspace":"bad id"}')
    expect(readActing(storage)).toBeNull()
    memory.set(ACTING_SESSION_KEY, 'not json')
    expect(readActing(storage)).toBeNull()
    expect(readActing(null)).toBeNull()
  })

  test('blocked storage never throws', () => {
    const blocked = {
      getItem: () => {
        throw new Error('blocked')
      },
      setItem: () => {
        throw new Error('blocked')
      },
      removeItem: () => {
        throw new Error('blocked')
      }
    }
    expect(readActing(blocked)).toBeNull()
    expect(() => writeActing(acting, blocked)).not.toThrow()
    expect(() => writeActing(null, blocked)).not.toThrow()
  })
})
