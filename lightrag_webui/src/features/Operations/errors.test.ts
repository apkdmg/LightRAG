import { describe, expect, test } from 'bun:test'
import { detailText, isRetryable, toOpsError } from './errors'

const apiError = (status: number, detail: unknown, requestId: string | null = 'req-1') =>
  Object.assign(new Error(`${status} X\n{}\n/admin/ops/status`), { status, detail, requestId })

describe('toOpsError', () => {
  test('403 maps to the admin-access message and keeps the request id', () => {
    const e = toOpsError(apiError(403, 'Admin privileges required'))
    expect(e.kind).toBe('forbidden')
    expect(e.messageKey).toBe('operations.error.forbidden')
    expect(e.requestId).toBe('req-1')
  })

  test('status codes map to kinds', () => {
    expect(toOpsError(apiError(404, 'Not Found')).kind).toBe('not_found')
    expect(toOpsError(apiError(409, 'x')).kind).toBe('conflict')
    expect(toOpsError(apiError(429, 'x')).kind).toBe('rate_limited')
    expect(toOpsError(apiError(503, 'x')).kind).toBe('unavailable')
    expect(toOpsError(apiError(504, 'x')).kind).toBe('timeout')
    expect(toOpsError(apiError(500, 'x')).kind).toBe('server')
    expect(toOpsError(apiError(422, 'x')).kind).toBe('bad_request')
  })

  test('a 5xx detail is not repeated; other details are shown as text', () => {
    expect(toOpsError(apiError(500, 'Internal Server Error')).detail).toBeNull()
    expect(toOpsError(apiError(404, 'Workspace not found')).detail).toBe('Workspace not found')
  })

  test('action refusals map to their own message, from a bare code or an object', () => {
    const a = toOpsError(apiError(409, 'lock_unavailable'))
    expect(a.refusal).toBe('lock_unavailable')
    expect(a.messageKey).toBe('operations.refusal.lock_unavailable')
    const b = toOpsError(apiError(409, { code: 'in_progress_by', message: 'opso is running "Run pipeline now"' }))
    expect(b.refusal).toBe('in_progress_by')
    expect(b.detail).toBe('opso is running "Run pipeline now"')
    expect(toOpsError(apiError(403, { reason: 'not_operator' })).messageKey).toBe('operations.refusal.not_operator')
  })

  test('network errors, timeouts and aborts without a response', () => {
    expect(toOpsError(Object.assign(new Error('Network Error'), { code: 'ERR_NETWORK' })).kind).toBe('network')
    expect(toOpsError(Object.assign(new Error('timeout of 30000ms exceeded'), { code: 'ECONNABORTED' })).kind).toBe('timeout')
    expect(toOpsError(Object.assign(new Error('canceled'), { name: 'CanceledError', code: 'ERR_CANCELED' })).kind).toBe('aborted')
    expect(toOpsError(new DOMException('Aborted', 'AbortError')).kind).toBe('aborted')
    expect(toOpsError(new Error('Authentication required')).kind).toBe('unauthorized')
    expect(toOpsError('weird').kind).toBe('unknown')
    expect(toOpsError(null).requestId).toBeNull()
  })

  test('falls back to a raw axios response when the interceptor did not run', () => {
    const e = toOpsError({ response: { status: 500, data: { detail: 'Internal Server Error', request_id: 'abc' } } })
    expect(e.kind).toBe('server')
    expect(e.requestId).toBe('abc')
  })

  test('retryable kinds', () => {
    expect(isRetryable(toOpsError(apiError(503, 'x')))).toBe(true)
    expect(isRetryable(toOpsError(apiError(500, 'x')))).toBe(true)
    expect(isRetryable(toOpsError(apiError(403, 'x')))).toBe(false)
    expect(isRetryable(toOpsError(apiError(404, 'x')))).toBe(false)
  })
})

describe('detailText', () => {
  test('normalises FastAPI detail shapes', () => {
    expect(detailText('  plain  ')).toBe('plain')
    expect(detailText([{ msg: 'field required' }, { msg: 'too long' }])).toBe('field required; too long')
    expect(detailText({ message: 'nope' })).toBe('nope')
    expect(detailText({})).toBeNull()
    expect(detailText('')).toBeNull()
    expect(detailText(42)).toBeNull()
  })

  test('caps very long details', () => {
    const text = detailText('x'.repeat(1000)) as string
    expect(text.length).toBe(300)
    expect(text.endsWith('…')).toBe(true)
  })
})
