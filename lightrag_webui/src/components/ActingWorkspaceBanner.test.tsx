import { describe, expect, test } from 'bun:test'
import { renderToString } from 'react-dom/server'

const memory = new Map<string, string>()
const storage = {
  getItem: (k: string) => memory.get(k) ?? null,
  setItem: (k: string, v: string) => void memory.set(k, v),
  removeItem: (k: string) => void memory.delete(k)
}
Object.defineProperty(globalThis, 'localStorage', { value: storage, configurable: true })
Object.defineProperty(globalThis, 'sessionStorage', { value: storage, configurable: true })

const { ActingBanner, default: ActingWorkspaceBanner } = await import('./ActingWorkspaceBanner')
const { shownActingWorkspace, useActingStore } = await import('@/stores/actingWorkspace')

describe('"Acting on <ws>" banner (OQ-13)', () => {
  test('names the workspace and offers Exit, with no way to dismiss it otherwise', () => {
    const html = renderToString(<ActingBanner workspace="fsktm" onExit={() => {}} />)
    expect(html).toContain('role="region"')
    expect(html).toContain('data-acting-workspace="fsktm"')
    expect(html).toContain('admin.acting.title')
    expect(html).toContain('admin.acting.exit')
    expect(html).not.toContain('admin.acting.openDocuments')
  })

  test('from another tab it also offers the way back to Documents', () => {
    expect(renderToString(<ActingBanner workspace="fsktm" onExit={() => {}} onOpenDocuments={() => {}} />)).toContain('admin.acting.openDocuments')
  })

  test('appears on first paint: no entrance animation, no transform', () => {
    const html = renderToString(<ActingBanner workspace="fsktm" onExit={() => {}} />)
    expect(html).not.toMatch(/animate-|ops-fade-in|transform/)
  })

  test('always shown while the mode is on for this admin; gone after Exit, for another user, or once the server says not admin', () => {
    const acting = { workspace: 'fsktm', user: 'ops', since: 1 }
    const auth = { isAuthenticated: true, isGuestMode: false, username: 'ops', role: 'admin' }
    expect(shownActingWorkspace(acting, auth, true)).toBe('fsktm')
    // An SSO admin whose role the browser does not know yet still sees it while the server is asked.
    expect(shownActingWorkspace(acting, { ...auth, role: null }, null)).toBe('fsktm')
    expect(shownActingWorkspace(null, auth, true)).toBeNull()
    expect(shownActingWorkspace(acting, { ...auth, username: 'someone' }, true)).toBeNull()
    expect(shownActingWorkspace(acting, auth, false)).toBeNull()
    expect(shownActingWorkspace(acting, { ...auth, isAuthenticated: false }, true)).toBeNull()
  })

  test('with the mode off nothing renders', () => {
    useActingStore.getState().exit()
    expect(renderToString(<ActingWorkspaceBanner />)).toBe('')
  })
})
