import { describe, expect, test } from 'bun:test'
import { insightsLink, parseInsightsHash, writeInsightsHash } from './hash'
import type { Scope } from './context'

const SCOPE: Scope = { tab: 'explore', purpose: '', workspace: 'fsktm', range: 30, start: '', end: '', unit: 'tokens', day: null }

describe('Insights hash: idoc (FR-54)', () => {
  test('idoc pre-filters Explore to one document, with its workspace', () => {
    const h = parseInsightsHash('#/?insights=1&itab=explore&iws=fsktm&idoc=doc-9a2f')
    expect(h.doc).toBe('doc-9a2f')
    expect(h.scope.workspace).toBe('fsktm')
    expect(h.scope.tab).toBe('explore')
  })

  test('idoc without a tab opens Explore; another tab is kept', () => {
    expect(parseInsightsHash('#/?insights=1&idoc=doc-1').scope.tab).toBe('explore')
    expect(parseInsightsHash('#/?insights=1&itab=work&idoc=doc-1').scope.tab).toBe('work')
  })

  test('a malformed idoc is dropped, never trusted', () => {
    expect(parseInsightsHash('#/?insights=1&idoc=a%20b').doc).toBeNull()
    expect(parseInsightsHash(`#/?insights=1&idoc=${'x'.repeat(257)}`).doc).toBeNull()
    expect(parseInsightsHash('#/?insights=1&idoc=').doc).toBeNull()
    expect(parseInsightsHash('#/?insights=1&idoc=%0A').doc).toBeNull()
  })

  test('without the insights marker nothing is read', () => {
    expect(parseInsightsHash('#/?ops=1&idoc=doc-1')).toEqual({ scope: {}, doc: null })
    expect(parseInsightsHash('#/')).toEqual({ scope: {}, doc: null })
  })

  test('the other keys still parse as before', () => {
    const h = parseInsightsHash('#/?insights=1&itab=overview&ipurpose=extract&irange=7&iday=2026-11-01')
    expect(h.scope).toEqual({ tab: 'overview', purpose: 'extract', range: 7, day: '2026-11-01' })
    expect(parseInsightsHash('#/?insights=1&ipurpose=nope&itab=nope').scope).toEqual({})
    expect(parseInsightsHash('#/?insights=1&irange=custom&istart=2026-10-01&iend=2026-10-31').scope).toEqual({ range: 'custom', start: '2026-10-01', end: '2026-10-31' })
  })

  test('writing keeps idoc only while Explore shows that document', () => {
    expect(writeInsightsHash(SCOPE, 'doc-1')).toBe('#/?insights=1&itab=explore&irange=30&iws=fsktm&idoc=doc-1')
    expect(writeInsightsHash({ ...SCOPE, tab: 'overview' }, 'doc-1')).not.toContain('idoc')
    expect(writeInsightsHash(SCOPE, null)).not.toContain('idoc')
    // Round trip.
    const back = parseInsightsHash(writeInsightsHash(SCOPE, 'doc-1'))
    expect(back.doc).toBe('doc-1')
    expect(back.scope.workspace).toBe('fsktm')
  })

  test('links from Operations: a document opens Explore on its calls', () => {
    expect(insightsLink({ workspace: 'fsktm', docId: 'doc-9a2f' })).toBe('#/?insights=1&itab=explore&iws=fsktm&idoc=doc-9a2f')
    expect(insightsLink({ tab: 'work' })).toBe('#/?insights=1&itab=work')
    expect(insightsLink({ purpose: 'extract' })).toBe('#/?insights=1&itab=explore&ipurpose=extract')
    expect(insightsLink({ purpose: 'not-a-purpose', workspace: 'bad id' })).toBe('#/?insights=1&itab=explore')
    const parsed = parseInsightsHash(insightsLink({ workspace: 'ws/with?chars&x=1', docId: 'd' }))
    expect(parsed.scope.workspace).toBe('ws/with?chars&x=1')
    expect(parsed.doc).toBe('d')
  })
})
