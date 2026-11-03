import { describe, expect, test } from 'bun:test'
import { MAX_BACKOFF_SECONDS, alignedDelay, backoffSeconds, keepIfSame, nextPollDelay, samePayload } from './poll'

describe('backoff math', () => {
  test('doubles per consecutive failure and caps at five minutes', () => {
    expect(backoffSeconds(15, 0)).toBe(15)
    expect(backoffSeconds(15, 1)).toBe(30)
    expect(backoffSeconds(15, 2)).toBe(60)
    expect(backoffSeconds(15, 4)).toBe(240)
    expect(backoffSeconds(15, 5)).toBe(MAX_BACKOFF_SECONDS)
    expect(backoffSeconds(15, 1000)).toBe(MAX_BACKOFF_SECONDS)
  })

  test('never goes below the base interval, even when the base exceeds the cap', () => {
    expect(backoffSeconds(600, 3)).toBe(600)
    expect(backoffSeconds(0, 0)).toBe(1)
    expect(backoffSeconds(5, -2)).toBe(5)
  })
})

describe('shared tick', () => {
  test('aligns to the next wall-clock multiple of the interval', () => {
    expect(alignedDelay(12_000, 15)).toBe(3_000)
    expect(alignedDelay(14_999, 15)).toBe(1)
    expect(alignedDelay(61_500, 60)).toBe(58_500)
  })

  test('a call exactly on the boundary waits a full period, never 0', () => {
    expect(alignedDelay(30_000, 15)).toBe(15_000)
  })

  test('two panels with the same interval fire at the same instant', () => {
    const a = 1_000_123
    const b = 1_004_567
    expect(a + alignedDelay(a, 5)).toBe(b + alignedDelay(b, 5))
  })

  test('errors switch from the aligned tick to backoff', () => {
    expect(nextPollDelay(12_000, 15, 0)).toBe(3_000)
    expect(nextPollDelay(12_000, 15, 2)).toBe(60_000)
  })
})

describe('unchanged payloads keep their identity', () => {
  test('an identical payload returns the previous object', () => {
    const previous = { generated_at: '2026-11-03T12:00:00Z', process: { mode: 'uvicorn', workers_live: null }, items: [1, 2] }
    const next = { generated_at: '2026-11-03T12:01:00Z', process: { mode: 'uvicorn', workers_live: null }, items: [1, 2] }
    expect(keepIfSame(previous, next)).toBe(previous)
  })

  test('a changed payload returns the new object', () => {
    const previous = { generated_at: 'a', items: [1, 2] }
    const next = { generated_at: 'b', items: [1, 3] }
    expect(keepIfSame(previous, next)).toBe(next)
    expect(keepIfSame(null, next)).toBe(next)
  })

  test('only the top-level generated_at is ignored', () => {
    expect(samePayload({ a: { generated_at: 1 } }, { a: { generated_at: 2 } })).toBe(false)
    expect(samePayload({ a: 1 }, { a: 1, b: undefined })).toBe(false)
    expect(samePayload([1, { x: 2 }], [1, { x: 2 }])).toBe(true)
    expect(samePayload([1], { 0: 1 })).toBe(false)
    expect(samePayload(null, null)).toBe(true)
  })
})
