import { describe, expect, test } from 'bun:test'

import type { ServerProviderSlot, ServerProviderView } from '@/api/lightrag'
import { knownKeyPreview, sameBaseUrl } from './serverProviderKeys'

const slot = (overrides: Partial<ServerProviderSlot> & { envHost?: string }): ServerProviderSlot => {
  const { envHost = 'https://env.example/v1', ...rest } = overrides
  return {
    base_url: null,
    model: null,
    reasoning_effort: null,
    preset_id: null,
    api_key_set: false,
    api_key_preview: '',
    active: false,
    effective: {
      binding: 'openai',
      model: 'm',
      host: envHost,
      reasoning_effort: null,
      source: 'env',
      api_key_preview: '...envk'
    },
    env: { binding: 'openai', host: envHost, model: 'm', api_key_preview: '...envk' },
    ...rest
  }
}

const view = (overrides: Partial<ServerProviderView>): ServerProviderView => ({
  extraction: slot({}),
  query: slot({}),
  vision: slot({}),
  updated_at: null,
  updated_by: null,
  status: 'ok',
  impact: { total_workspaces: 0, own_provider: { extraction: [], query: [], vision: [] } },
  paused: [],
  ...overrides
})

describe('sameBaseUrl', () => {
  test('ignores trailing slashes', () => {
    expect(sameBaseUrl('https://a.example/v1/', 'https://a.example/v1')).toBe(true)
  })

  test('tells different hosts apart', () => {
    expect(sameBaseUrl('https://a.example/v1', 'https://b.example/v1')).toBe(false)
  })
})

describe('knownKeyPreview', () => {
  test('reuses the .env key for the .env host', () => {
    expect(knownKeyPreview(view({}), 'query', 'https://env.example/v1/')).toBe('...envk')
  })

  test('needs a new key for a new host', () => {
    expect(knownKeyPreview(view({}), 'query', 'https://new.example/v1')).toBeNull()
  })

  test('prefers a saved key in this slot over another slot', () => {
    const saved = (preview: string) =>
      slot({
        active: true,
        base_url: 'https://saved.example/v1',
        api_key_set: true,
        api_key_preview: preview
      })
    const current = view({ extraction: saved('...aaaa'), query: saved('...bbbb') })
    expect(knownKeyPreview(current, 'query', 'https://saved.example/v1')).toBe('...bbbb')
  })

  test('reuses a key saved in another slot for the same host', () => {
    const current = view({
      extraction: slot({
        active: true,
        base_url: 'https://saved.example/v1',
        api_key_set: true,
        api_key_preview: '...aaaa'
      })
    })
    expect(knownKeyPreview(current, 'vision', 'https://saved.example/v1')).toBe('...aaaa')
  })

  test('does not count a .env host with no key', () => {
    const noKey = slot({})
    noKey.env.api_key_preview = ''
    const current = view({ extraction: noKey, query: noKey, vision: noKey })
    expect(knownKeyPreview(current, 'query', 'https://env.example/v1')).toBeNull()
  })
})
