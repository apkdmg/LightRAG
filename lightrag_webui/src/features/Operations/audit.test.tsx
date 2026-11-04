import { describe, expect, test } from 'bun:test'
import { renderToString } from 'react-dom/server'

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
const { actionSuggestions, auditActor, auditFilters, auditParams, auditRowState, dayStartIso, draftInvalid, EMPTY_AUDIT_DRAFT } = await import('./activity/audit')
const { AuditTable } = await import('./activity/AuditView')
const { normalizeAuditPage, normalizeAuditRow } = await import('./normalize')

describe('audit filters (§3.7 GET /audit)', () => {
  test('a picked day is sent as the UTC instant of that local midnight; "to" includes its whole day', () => {
    const from = dayStartIso('2026-11-03')!
    const to = dayStartIso('2026-11-04', 1)!
    expect(new Date(from).getTime()).toBe(new Date(2026, 10, 3).getTime())
    expect(new Date(to).getTime()).toBe(new Date(2026, 10, 5).getTime())
    expect(dayStartIso('03/11/2026')).toBeNull()
  })

  test('blank fields are left out and text is trimmed', () => {
    expect(auditFilters(EMPTY_AUDIT_DRAFT)).toEqual({})
    const f = auditFilters({ actor: ' opso ', action: 'cancel', from: '2026-11-01', to: '' })
    expect(f).toEqual({ actor: 'opso', action: 'cancel', from: dayStartIso('2026-11-01')! })
  })

  test('the keyset cursor rides along; the first page has none', () => {
    expect(auditParams({ ...EMPTY_AUDIT_DRAFT, actor: 'opso' }, null)).toEqual({ actor: 'opso', before: undefined })
    expect(auditParams(EMPTY_AUDIT_DRAFT, '2026-11-04T10:00:00+00:00,57')).toEqual({ before: '2026-11-04T10:00:00+00:00,57' })
  })

  test('a "to" day before the "from" day is caught before asking', () => {
    expect(draftInvalid({ ...EMPTY_AUDIT_DRAFT, from: '2026-11-04', to: '2026-11-03' })).toBe(true)
    expect(draftInvalid({ ...EMPTY_AUDIT_DRAFT, from: '2026-11-04', to: '2026-11-04' })).toBe(false)
  })
})

describe('audit rows', () => {
  test('normalised from the server: numeric ids as text, unknown fields dropped, rows without an action skipped', () => {
    const page = normalizeAuditPage({
      items: [
        { id: 57, ts: '2026-11-04T10:22:40Z', actor: 'opso', auth_kind: 'sso', action: 'cancel', workspace: 'fsktm', reason: 'user asked', phase: 'result', outcome: 'cancellation_requested', request_id: '9be2', params: { x: 1 } },
        { id: 58, ts: '2026-11-04T10:22:41Z' }
      ],
      next_before: '2026-11-04T10:22:40+00:00,57'
    })
    expect(page.items).toHaveLength(1)
    expect(page.items[0]).toMatchObject({ id: '57', actor: 'opso', action: 'cancel', workspace: 'fsktm', outcome: 'cancellation_requested' })
    expect(page.items[0]).not.toHaveProperty('params')
    expect(page.next_before).toBe('2026-11-04T10:22:40+00:00,57')
    expect(normalizeAuditPage(null)).toMatchObject({ items: [], next_before: null })
    expect(normalizeAuditRow({ id: 'x' })).toBeNull()
  })

  test('state: intent rows are pending, refusals warn, errors are critical', () => {
    expect(auditRowState({ phase: 'intent', outcome: null })).toBe('unknown')
    expect(auditRowState({ phase: 'result', outcome: 'released' })).toBe('ok')
    expect(auditRowState({ phase: 'result', outcome: 'ok' })).toBe('ok')
    expect(auditRowState({ phase: 'result', outcome: 'owner_alive' })).toBe('warning')
    expect(auditRowState({ phase: 'result', outcome: 'refused_busy' })).toBe('warning')
    expect(auditRowState({ phase: 'result', outcome: 'error' })).toBe('critical')
    expect(auditRowState({ phase: 'result', outcome: 'not_busy' })).toBe('unknown')
  })

  test('the actor names the impersonator; suggestions come from the rows seen', () => {
    expect(auditActor({ actor: 'alice', impersonated_by: 'opso' })).toBe('alice (opso)')
    expect(auditActor({ actor: null, impersonated_by: null })).toBe('–')
    expect(actionSuggestions([{ action: 'run' }, { action: 'cancel' }, { action: 'run' }])).toEqual(['cancel', 'run'])
  })

  test('the table shows time, actor, action, target, reason, outcome and ref; server text is escaped', () => {
    const html = renderToString(
      <AuditTable
        now={Date.parse('2026-11-04T12:00:00Z')}
        rows={[
          { id: '1', ts: '2026-11-04T10:22:40Z', actor: 'opso', auth_kind: 'sso', action: 'cancel', workspace: 'fsktm', reason: '=HYPERLINK("x")<b>', phase: 'intent', outcome: null, request_id: '9be2aa' },
          { id: '2', ts: '2026-11-04T10:22:41Z', actor: 'opso', action: 'cancel', workspace: 'fsktm', target: 'fsktm', phase: 'result', outcome: 'cancellation_requested', request_id: null }
        ]}
      />
    )
    expect(html).toContain('operations.audit.intent')
    // An outcome code without a translation reads as words.
    expect(html).toContain('cancellation requested')
    expect(html).toContain('fsktm')
    expect(html).toContain('9be2aa')
    expect(html).toContain('&lt;b&gt;')
    expect(html).not.toContain('<b>')
  })
})
