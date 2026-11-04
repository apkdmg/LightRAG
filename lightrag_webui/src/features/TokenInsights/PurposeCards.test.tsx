import { describe, expect, test } from 'bun:test'
import { renderToString } from 'react-dom/server'

import { Ctx } from './context'
import { PURPOSES, isFarFromOthers } from './meta'

// The stores behind these components read localStorage as they load.
const memory = new Map<string, string>()
Object.defineProperty(globalThis, 'localStorage', {
  value: {
    getItem: (k: string) => memory.get(k) ?? null,
    setItem: (k: string, v: string) => void memory.set(k, v),
    removeItem: (k: string) => void memory.delete(k)
  },
  configurable: true
})
const { PurposeCards } = await import('./OverviewTab')
const { ViewBoundary } = await import('@/components/admin')

const purpose = (id: string, others_per_unit: number | null, others_ratio: number | null) => ({
  id,
  status: { code: 'learning', level: 0 },
  tokens: 0,
  share_tokens: 0,
  share_cost: 0,
  series: [0, 0],
  per_unit: null,
  per_unit_usual: null,
  others_per_unit,
  others_ratio,
  cache_rate: null,
  model: '',
  billing: 'system'
})

const render = (others_per_unit: number | null, others_ratio: number | null) =>
  renderToString(
    <Ctx.Provider value={{ set: () => {} } as any}>
      <PurposeCards
        data={{
          learning: { anomalies_ready: false },
          scope: { workspace: 'quiet' },
          purposes: PURPOSES.map((id) => purpose(id, others_per_unit, others_ratio))
        }}
      />
    </Ctx.Provider>
  )

describe('PurposeCards on a workspace page', () => {
  test('a workspace with no usage has no ratio, and still renders', () => {
    const html = render(1200, null)
    expect(html).toContain('insights.card.others')
    expect(html).not.toContain('insights.card.thisOne')
  })

  test('a workspace far from the others says so', () => {
    expect(render(1200, 2.5)).toContain('insights.card.thisOne')
    expect(render(1200, 0.5)).toContain('insights.card.thisOne')
    expect(render(1200, 1.1)).not.toContain('insights.card.thisOne')
  })
})

describe('isFarFromOthers', () => {
  test('null and undefined are not far, though null <= 0.8 in JavaScript', () => {
    expect(isFarFromOthers(null)).toBe(false)
    expect(isFarFromOthers(undefined)).toBe(false)
  })

  test('a ratio of 0 is far', () => {
    expect(isFarFromOthers(0)).toBe(true)
  })
})

describe('ViewBoundary', () => {
  const boundary = (resetKey: string, failed: boolean) => {
    const b = new ViewBoundary({ resetKey, children: 'view' })
    b.state = { failed }
    const updates: unknown[] = []
    b.setState = ((s: unknown) => updates.push(s)) as any
    return { b, updates }
  }

  test('an error marks the view failed', () => {
    expect(ViewBoundary.getDerivedStateFromError()).toEqual({ failed: true })
  })

  test('renders the view until it fails', () => {
    expect(boundary('a', false).b.render()).toBe('view')
    expect(boundary('a', true).b.render()).not.toBe('view')
  })

  test('tries again only when the view changes', () => {
    const same = boundary('a', true)
    same.b.componentDidUpdate({ resetKey: 'a' })
    expect(same.updates).toEqual([])
    const changed = boundary('b', true)
    changed.b.componentDidUpdate({ resetKey: 'a' })
    expect(changed.updates).toEqual([{ failed: false }])
  })
})
