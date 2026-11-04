import { describe, expect, test } from 'bun:test'
import { DEFAULT_OPS_HASH, OpsHash, hasOpsMarker, readOpsHash, writeOpsHash } from './hashState'

describe('readOpsHash', () => {
  test('ignores hashes without the ops marker', () => {
    expect(readOpsHash('')).toBeNull()
    expect(readOpsHash('#/')).toBeNull()
    expect(readOpsHash('#/?insights=1&itab=overview')).toBeNull()
    expect(readOpsHash('#/?ops=0&otab=logs')).toBeNull()
    expect(hasOpsMarker('#/?insights=1')).toBe(false)
    expect(hasOpsMarker('#/?ops=1')).toBe(true)
  })

  test('defaults to the overview tab', () => {
    expect(readOpsHash('#/?ops=1')).toEqual(DEFAULT_OPS_HASH)
  })

  test('reads the tab and a sub-view that belongs to it', () => {
    expect(readOpsHash('#/?ops=1&otab=processing&osub=stuck')).toEqual({ ...DEFAULT_OPS_HASH, tab: 'processing', sub: 'stuck' })
  })

  test('drops unknown tabs and sub-views that belong to another tab', () => {
    expect(readOpsHash('#/?ops=1&otab=bogus')?.tab).toBe('overview')
    expect(readOpsHash('#/?ops=1&otab=logs&osub=stuck')?.sub).toBeNull()
    expect(readOpsHash('#/?ops=1&otab=overview&osub=active')?.sub).toBeNull()
  })

  test('reads one drawer, the first in key order', () => {
    expect(readOpsHash('#/?ops=1&ows=fsktm')?.drawer).toEqual({ kind: 'workspace', id: 'fsktm' })
    expect(readOpsHash('#/?ops=1&orun=7f3a&ows=fsktm')?.drawer).toEqual({ kind: 'workspace', id: 'fsktm' })
    expect(readOpsHash('#/?ops=1&oal=a1&owk=w9')?.drawer).toEqual({ kind: 'worker', id: 'w9' })
  })

  test('parses odoc as <workspace>/<doc id> and rejects malformed values', () => {
    expect(readOpsHash('#/?ops=1&odoc=fsktm%2Fdoc-9a2f')?.drawer).toEqual({ kind: 'document', workspace: 'fsktm', id: 'doc-9a2f' })
    expect(readOpsHash('#/?ops=1&odoc=doc-9a2f')?.drawer).toBeNull()
    expect(readOpsHash('#/?ops=1&odoc=%2Fdoc-9a2f')?.drawer).toBeNull()
    expect(readOpsHash('#/?ops=1&odoc=fsktm%2F')?.drawer).toBeNull()
  })

  test('rejects ids with whitespace, control characters or excessive length', () => {
    expect(readOpsHash('#/?ops=1&ows=a%20b')?.drawer).toBeNull()
    expect(readOpsHash('#/?ops=1&ows=a%0Ab')?.drawer).toBeNull()
    expect(readOpsHash(`#/?ops=1&ows=${'x'.repeat(257)}`)?.drawer).toBeNull()
    expect(readOpsHash(`#/?ops=1&ows=${'x'.repeat(256)}`)?.drawer).toEqual({ kind: 'workspace', id: 'x'.repeat(256) })
  })

  test('keeps the log query only when it is reasonably short', () => {
    expect(readOpsHash('#/?ops=1&otab=logs&olq=level%3DERROR')?.logQuery).toBe('level=ERROR')
    expect(readOpsHash(`#/?ops=1&olq=${'q'.repeat(2001)}`)?.logQuery).toBeNull()
  })
})

describe('writeOpsHash', () => {
  test('round-trips every field', () => {
    const states: OpsHash[] = [
      DEFAULT_OPS_HASH,
      { tab: 'processing', sub: 'backlog', drawer: { kind: 'workspace', id: 'user_example_edu' }, logQuery: null },
      { tab: 'logs', sub: 'grouped', drawer: { kind: 'log', id: '12345' }, logQuery: 'level=ERROR&run=7f3a' },
      { tab: 'health', sub: null, drawer: { kind: 'document', workspace: 'fsktm', id: 'doc-1' }, logQuery: null },
      { tab: 'activity', sub: 'audit', drawer: { kind: 'alert', id: 'al-1' }, logQuery: null }
    ]
    for (const state of states) {
      expect(readOpsHash(writeOpsHash(state))).toEqual(state)
    }
  })

  test('writes the marker and the router prefix, and omits empty fields', () => {
    expect(writeOpsHash(DEFAULT_OPS_HASH)).toBe('#/?ops=1&otab=overview')
  })

  test('drops a sub-view that does not belong to the tab', () => {
    expect(writeOpsHash({ ...DEFAULT_OPS_HASH, tab: 'logs', sub: 'stuck' })).toBe('#/?ops=1&otab=logs')
  })
})
