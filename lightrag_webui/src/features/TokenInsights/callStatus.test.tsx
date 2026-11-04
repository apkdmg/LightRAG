import { describe, expect, test } from 'bun:test'
import { renderToString } from 'react-dom/server'
import type { TFunction } from 'i18next'

const memory = new Map<string, string>()
Object.defineProperty(globalThis, 'localStorage', {
  value: {
    getItem: (k: string) => memory.get(k) ?? null,
    setItem: (k: string, v: string) => void memory.set(k, v),
    removeItem: (k: string) => void memory.delete(k)
  },
  configurable: true
})

const { callStatusText } = await import('./callStatus')
const { RequestCell } = await import('./callCells')

/** Keys as text, error kinds marked, so a test sees which key each part used. */
const t = ((key: string) => (key.startsWith('operations.errorKind.') ? `kind:${key.slice(21)}` : key)) as unknown as TFunction

describe('Explore per-call rows (error_kind, request_id)', () => {
  test('a failed call says what kind of failure the recorder classified', () => {
    expect(callStatusText(t, { status: 'error', error_kind: 'rate_limited' })).toBe('insights.explore.status.error · kind:rate_limited')
  })

  test('rows from before the column read as before', () => {
    expect(callStatusText(t, { status: 'ok' })).toBe('insights.explore.status.ok')
    expect(callStatusText(t, { status: 'ok', cache_hit: true, estimated: true })).toBe('insights.explore.cacheHit · insights.explore.estimated')
  })

  test('the request ID is short and opens its logs; none reads as a dash', () => {
    const html = renderToString(<RequestCell id="0123456789abcdef0123456789abcdef" />)
    expect(html).toContain('01234567…')
    expect(html).toContain('<button')
    expect(renderToString(<RequestCell id={null} />)).toContain('–')
  })
})
