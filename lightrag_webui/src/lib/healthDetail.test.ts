import { describe, expect, test } from 'bun:test'
import { hasHealthDetail } from './healthDetail'

describe('hasHealthDetail', () => {
  test('a session answer carries the configuration block', () => {
    expect(hasHealthDetail({ status: 'healthy', configuration: { max_graph_nodes: '1000' }, pipeline_busy: false })).toBe(true)
  })

  test('an anonymous liveness answer does not, so the last detailed status is kept', () => {
    expect(hasHealthDetail({ status: 'healthy', version: '1.5.1' })).toBe(false)
    expect(hasHealthDetail({ status: 'healthy', configuration: null })).toBe(false)
    expect(hasHealthDetail({ status: 'healthy', configuration: [] })).toBe(false)
    expect(hasHealthDetail(null)).toBe(false)
  })
})
