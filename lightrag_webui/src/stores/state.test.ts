import { beforeAll, beforeEach, describe, expect, test } from 'bun:test'

type StateModule = typeof import('./state')

const SSO_PLACEHOLDER = 'cookie-based-auth'

const storageMock = () => {
  const data = new Map<string, string>()

  return {
    getItem: (key: string) => data.get(key) ?? null,
    setItem: (key: string, value: string) => {
      data.set(key, value)
    },
    removeItem: (key: string) => {
      data.delete(key)
    },
    clear: () => {
      data.clear()
    }
  }
}

const jwt = (payload: Record<string, unknown>) =>
  ['header', btoa(JSON.stringify(payload)), 'signature'].join('.')

let stateModule: StateModule

const auth = () => stateModule.useAuthStore.getState()

const ssoLogin = (role?: string | null) =>
  auth().login(SSO_PLACEHOLDER, false, true, null, null, null, null, role)

beforeAll(async () => {
  Object.defineProperty(globalThis, 'localStorage', {
    value: storageMock(),
    configurable: true
  })
  Object.defineProperty(globalThis, 'sessionStorage', {
    value: storageMock(),
    configurable: true
  })

  stateModule = await import('./state')
})

beforeEach(() => {
  localStorage.clear()
})

describe('auth store role', () => {
  test('SSO login takes the role from the login metadata', () => {
    ssoLogin('admin')

    expect(auth().role).toBe('admin')
  })

  test('SSO role survives a page reload', () => {
    ssoLogin('admin')

    expect(stateModule.initAuthState().role).toBe('admin')
  })

  test('re-login without metadata keeps the stored SSO role', () => {
    ssoLogin('admin')
    ssoLogin()

    expect(auth().role).toBe('admin')
    expect(stateModule.initAuthState().role).toBe('admin')
  })

  test('SSO login with no role clears the previous user role', () => {
    ssoLogin('admin')
    ssoLogin(null)

    expect(auth().role).toBeNull()
    expect(stateModule.initAuthState().role).toBeNull()
  })

  test('a role in a renewed SSO token takes priority over the stored role', () => {
    ssoLogin('admin')
    localStorage.setItem('LIGHTRAG-API-TOKEN', jwt({ sub: 'someone', role: 'user' }))

    expect(stateModule.initAuthState().role).toBe('user')
  })

  test('password login reads the token and drops any stored SSO role', () => {
    ssoLogin('admin')
    auth().login(jwt({ sub: 'someone', role: 'user' }))

    expect(auth().role).toBe('user')
    expect(localStorage.getItem('LIGHTRAG-SSO-ROLE')).toBeNull()
  })

  test('a non-SSO placeholder token never picks up a stored role', () => {
    localStorage.setItem('LIGHTRAG-SSO-ROLE', 'admin')
    localStorage.setItem('LIGHTRAG-API-TOKEN', SSO_PLACEHOLDER)

    expect(stateModule.initAuthState().role).toBeNull()
  })

  test('logout clears the stored SSO role', () => {
    ssoLogin('admin')
    Object.defineProperty(globalThis, 'window', {
      value: { location: { href: '' } },
      configurable: true
    })

    try {
      auth().logout()
    } finally {
      delete (globalThis as { window?: unknown }).window
    }

    expect(auth().role).toBeNull()
    expect(localStorage.getItem('LIGHTRAG-SSO-ROLE')).toBeNull()
  })
})
