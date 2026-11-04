import { describe, expect, test } from 'bun:test'
import { renderToString } from 'react-dom/server'

// The stores behind the feature components read localStorage as they load.
const memory = new Map<string, string>()
Object.defineProperty(globalThis, 'localStorage', {
  value: {
    getItem: (k: string) => memory.get(k) ?? null,
    setItem: (k: string, v: string) => void memory.set(k, v),
    removeItem: (k: string) => void memory.delete(k)
  },
  configurable: true
})

const admin = await import('@/components/admin')
const insightsUi = await import('@/features/TokenInsights/ui')
const { Card, Empty, Kv, Meter, RowButton, Seg, Spark, StatusIcon, TextButton, Tile, Title, ViewBoundary } = admin

describe('admin primitives (§2.11, promoted from Token Insights)', () => {
  test('one source: Token Insights keeps only what knows about purposes and money', () => {
    for (const name of ['Card', 'Title', 'Seg', 'Kv', 'Meter', 'Spark', 'Tile', 'Empty', 'TextButton', 'RowButton', 'StatusIcon', 'ViewBoundary']) {
      expect(name in admin).toBe(true)
      expect(name in insightsUi).toBe(false)
    }
    for (const name of ['Swatch', 'PurposeName', 'StatusLabel', 'SeverityLabel', 'Mix', 'Counter', 'fmtTokens', 'fmtMoney']) {
      expect(name in insightsUi).toBe(true)
    }
  })

  // The markup is what both consoles drew before the move: no visual change.
  test('Card and Title', () => {
    expect(renderToString(<Card className="x">c</Card>)).toBe('<div class="bg-card rounded-[10px] border p-4 x">c</div>')
    expect(renderToString(<Title sub="s">t</Title>)).toBe('<div class="mb-2"><h2 class="m-0 text-[15px] font-semibold">t</h2><p class="text-muted-foreground mt-0.5 text-[13px]">s</p></div>')
  })

  test('Seg: a labelled group of pressed buttons', () => {
    const html = renderToString(
      <Seg
        value="b"
        label="View"
        onChange={() => {}}
        options={[
          { value: 'a', label: 'A' },
          { value: 'b', label: 'B' },
          { value: 'c', label: 'C', disabled: true }
        ]}
      />
    )
    expect(html).toContain('role="group" aria-label="View"')
    expect(html).toContain('aria-pressed="true" class="inline-flex min-h-8 items-center gap-1.5 rounded-md px-2.5 py-1 text-[13px] transition-colors disabled:opacity-40 bg-background font-medium shadow-sm"')
    expect(html).toContain('disabled=""')
  })

  test('Kv, Meter, Spark, Tile, Empty, TextButton, RowButton, StatusIcon', () => {
    expect(renderToString(<Kv k="k">v</Kv>)).toBe('<div class="flex justify-between gap-3 text-xs"><span class="text-muted-foreground">k</span><span class="text-right font-semibold">v</span></div>')
    expect(renderToString(<Meter value={150} />)).toContain('width:100%')
    expect(renderToString(<Spark values={[1, 2]} color="red" />)).toContain('height:50%')
    expect(renderToString(<Tile label="L" value="9" note="n" hero />)).toContain('text-[34px]')
    expect(renderToString(<Empty>e</Empty>)).toBe('<p class="text-muted-foreground m-0 text-sm">e</p>')
    expect(renderToString(<TextButton onClick={() => {}}>go</TextButton>)).toContain('underline')
    expect(renderToString(<RowButton>r</RowButton>)).toContain('disabled=""')
    expect(renderToString(<StatusIcon path="M0 0" color="red" />)).toContain('stroke="red"')
  })

  test('ViewBoundary names its console in the error and recovers on a new key', () => {
    const b = new ViewBoundary({ resetKey: 'a', name: 'Operations', children: 'view' })
    expect(b.render()).toBe('view')
    b.state = { failed: true }
    expect(b.render()).not.toBe('view')
    const errors: unknown[][] = []
    const original = console.error
    console.error = (...args: unknown[]) => void errors.push(args)
    try {
      b.componentDidCatch(new Error('boom'))
    } finally {
      console.error = original
    }
    expect(errors[0][0]).toBe('Operations view failed:')
  })
})

describe('both consoles still render with the promoted primitives', () => {
  test('Token Insights', async () => {
    const { Ctx } = await import('@/features/TokenInsights/context')
    const { PurposeCards } = await import('@/features/TokenInsights/OverviewTab')
    const { PURPOSES } = await import('@/features/TokenInsights/meta')
    const purpose = (id: string) => ({ id, status: { code: 'normal', level: 0 }, tokens: 10, share_tokens: 0.5, share_cost: 0.5, series: [1, 2], per_unit: 1, per_unit_usual: 1, others_per_unit: null, others_ratio: null, cache_rate: null, model: 'm', billing: 'system' })
    const html = renderToString(
      <Ctx.Provider value={{ set: () => {} } as any}>
        <PurposeCards data={{ learning: { anomalies_ready: true }, scope: {}, purposes: PURPOSES.map(purpose) }} />
      </Ctx.Provider>
    )
    // Spark and Kv inside the purpose cards.
    expect(html).toContain('flex h-11 items-end gap-px border-b')
    expect(html).toContain('<div class="flex justify-between gap-3 text-xs"><span class="text-muted-foreground">insights.card.usually</span>')
  })

  test('Operations', async () => {
    const { OpsShellContext } = await import('@/features/Operations/context')
    const { ServerCard } = await import('@/features/Operations/OverviewTab')
    const html = renderToString(
      <OpsShellContext.Provider
        value={{
          status: { available: true, features: { durable: true }, process: { mode: 'uvicorn', expected_workers: 1, workers_live: 1 }, operator: true, version: '1', generated_at: '2026-11-04T10:00:00Z' },
          openInsights: () => {}
        }}
      >
        <ServerCard />
      </OpsShellContext.Provider>
    )
    expect(html).toContain('bg-card rounded-[10px] border p-4')
    expect(html).toContain('operations.system.title')
  })
})
