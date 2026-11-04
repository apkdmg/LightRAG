import { describe, expect, test } from 'bun:test'

// The store reads localStorage as it loads; seed a remembered tab this build does not know.
const memory = new Map<string, string>()
memory.set('settings-storage', JSON.stringify({ state: { currentTab: 'retired-tab', theme: 'dark' }, version: 20 }))
const storage = {
  getItem: (k: string) => memory.get(k) ?? null,
  setItem: (k: string, v: string) => void memory.set(k, v),
  removeItem: (k: string) => void memory.delete(k)
}
Object.defineProperty(globalThis, 'localStorage', { value: storage, configurable: true })
Object.defineProperty(globalThis, 'sessionStorage', { value: storage, configurable: true })

const { sanitizeTab, useSettingsStore } = await import('./settings')

describe('remembered tab', () => {
  test('known tabs, including operations, pass through', () => {
    expect(sanitizeTab('operations')).toBe('operations')
    expect(sanitizeTab('insights')).toBe('insights')
    expect(sanitizeTab('knowledge-graph')).toBe('knowledge-graph')
  })

  test('unknown or malformed values fall back to documents', () => {
    expect(sanitizeTab('retired-tab')).toBe('documents')
    expect(sanitizeTab(undefined)).toBe('documents')
    expect(sanitizeTab(42)).toBe('documents')
  })

  test('a persisted unknown tab is replaced on load, other settings are kept', () => {
    const state = useSettingsStore.getState()
    expect(state.currentTab).toBe('documents')
    expect(state.theme).toBe('dark')
  })
})
