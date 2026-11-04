import { beforeAll, describe, expect, test } from 'bun:test'
import { AxiosHeaders, InternalAxiosRequestConfig } from 'axios'

const storageMock = () => {
  const data = new Map<string, string>()
  return {
    getItem: (key: string) => data.get(key) ?? null,
    setItem: (key: string, value: string) => void data.set(key, value),
    removeItem: (key: string) => void data.delete(key),
    clear: () => data.clear()
  }
}

let api: typeof import('./lightrag')
let acting: typeof import('@/stores/actingWorkspace')
let auth: typeof import('@/stores/state')

beforeAll(async () => {
  Object.defineProperty(globalThis, 'localStorage', { value: storageMock(), configurable: true })
  Object.defineProperty(globalThis, 'sessionStorage', { value: storageMock(), configurable: true })
  api = await import('./lightrag')
  acting = await import('@/stores/actingWorkspace')
  auth = await import('@/stores/state')
})

/** Runs a request config through the client's request interceptors, as axios would. */
const intercept = async (url: string) => {
  let config = { url, headers: new AxiosHeaders() } as InternalAxiosRequestConfig
  const handlers = (api.axiosInstance.interceptors.request as unknown as { handlers: ({ fulfilled: (c: InternalAxiosRequestConfig) => InternalAxiosRequestConfig } | null)[] }).handlers
  for (const h of handlers) if (h) config = await h.fulfilled(config)
  return config.headers.get('X-Target-Workspace') ?? null
}

describe('the API client and the admin "act on workspace" mode (OQ-13)', () => {
  test('Documents calls carry the target workspace while an admin acts on it', async () => {
    auth.useAuthStore.setState({ isAuthenticated: true, isGuestMode: false, username: 'ops', role: 'admin' })
    expect(acting.useActingStore.getState().start('fsktm', { admin: true, user: 'ops' })).toBe(true)
    expect(await intercept('/documents/paginated')).toBe('fsktm')
    expect(await intercept('/documents/upload')).toBe('fsktm')
    // Everything else still goes to the admin's own workspace.
    expect(await intercept('/query')).toBeNull()
    expect(await intercept('/graphs')).toBeNull()
  })

  test('a non-admin session never sends it, even with a mode left in the store', async () => {
    auth.useAuthStore.setState({ isAuthenticated: true, isGuestMode: false, username: 'ops', role: 'user' })
    expect(await intercept('/documents/paginated')).toBeNull()
  })

  test('another user on the same browser does not inherit it', async () => {
    auth.useAuthStore.setState({ isAuthenticated: true, isGuestMode: false, username: 'someone', role: 'admin' })
    expect(await intercept('/documents/paginated')).toBeNull()
  })

  test('Exit clears it', async () => {
    auth.useAuthStore.setState({ isAuthenticated: true, isGuestMode: false, username: 'ops', role: 'admin' })
    acting.useActingStore.getState().exit()
    expect(await intercept('/documents/paginated')).toBeNull()
  })
})
