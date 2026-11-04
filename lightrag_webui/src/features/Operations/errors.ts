/**
 * Maps any error from an Operations request to something a panel can show:
 * a message key, the server's detail when it is safe text, and the request ID
 * for the "Ref:" line. The API interceptor attaches status/detail/requestId
 * to its errors (api/lightrag.ts toApiError); network errors arrive as raw
 * axios errors without a response.
 */

export type OpsErrorKind =
  | 'forbidden'
  | 'unauthorized'
  | 'not_found'
  | 'conflict'
  | 'rate_limited'
  | 'unavailable'
  | 'server'
  | 'bad_request'
  | 'timeout'
  | 'network'
  | 'aborted'
  | 'unknown'

/** Action refusals the server returns as a result or detail code (spec §2.10, §3.7). */
export const REFUSALS = [
  'in_progress_by',
  'lock_unavailable',
  'refused_busy',
  'refused_paused',
  'not_operator',
  'not_busy',
  'nothing_pending',
  'owner_alive',
  'cas_mismatch',
  'single_process',
  'rate_limited',
  'feature_disabled',
  'probes_disabled',
  'not_checkable',
  'probe_failed',
  // Admin actions (PR-7): the guards every POST passes before it acts (§3.9 step 1 and 3).
  'audit_unavailable',
  'admin_not_current',
  'impersonation_not_allowed',
  'client_header_required',
  'origin_not_allowed',
  'invalid_reason',
  'owner_unknown',
  'confirm_mismatch',
  'invalid_kind',
  'invalid_id',
  'invalid_workspace',
  'workspace_not_served',
  'queue_unavailable',
  'invalid_range'
] as const
export type Refusal = (typeof REFUSALS)[number]

export type OpsError = {
  kind: OpsErrorKind
  status: number | null
  /** i18n key under operations.error.* or operations.refusal.* */
  messageKey: string
  /** Server detail as plain text (already generic on the server; trimmed here). */
  detail: string | null
  requestId: string | null
  refusal: Refusal | null
}

const MAX_DETAIL = 300

const isRefusal = (v: unknown): v is Refusal => typeof v === 'string' && (REFUSALS as readonly string[]).includes(v)

/** FastAPI details come as a string, a list of validation items, or an object. */
export const detailText = (detail: unknown): string | null => {
  let text: string | null = null
  if (typeof detail === 'string') text = detail
  else if (Array.isArray(detail)) {
    const parts = detail
      .map((d) => (typeof d === 'string' ? d : d && typeof d === 'object' && typeof d.msg === 'string' ? d.msg : null))
      .filter((d): d is string => !!d)
    text = parts.length ? parts.join('; ') : null
  } else if (detail && typeof detail === 'object') {
    const d = detail as Record<string, unknown>
    const msg = d.message ?? d.msg ?? d.reason ?? d.code
    text = typeof msg === 'string' ? msg : null
  }
  if (!text) return null
  const trimmed = text.trim()
  if (!trimmed) return null
  return trimmed.length > MAX_DETAIL ? `${trimmed.slice(0, MAX_DETAIL - 1)}…` : trimmed
}

const refusalOf = (detail: unknown): Refusal | null => {
  if (isRefusal(detail)) return detail
  if (detail && typeof detail === 'object' && !Array.isArray(detail)) {
    const d = detail as Record<string, unknown>
    for (const key of ['code', 'result', 'reason']) {
      if (isRefusal(d[key])) return d[key] as Refusal
    }
  }
  return null
}

const kindOfStatus = (status: number): OpsErrorKind => {
  if (status === 401) return 'unauthorized'
  if (status === 403) return 'forbidden'
  if (status === 404) return 'not_found'
  if (status === 409) return 'conflict'
  if (status === 429) return 'rate_limited'
  if (status === 503) return 'unavailable'
  if (status === 504) return 'timeout'
  if (status >= 500) return 'server'
  if (status >= 400) return 'bad_request'
  return 'unknown'
}

/** True for a request the page cancelled itself (unmount, superseded poll). Never shown. */
export const isAbort = (err: any) =>
  !!err && (err.name === 'AbortError' || err.name === 'CanceledError' || err.code === 'ERR_CANCELED')

export function toOpsError(err: any): OpsError {
  const status: number | null =
    typeof err?.status === 'number' ? err.status : typeof err?.response?.status === 'number' ? err.response.status : null
  const rawDetail = err?.detail !== undefined ? err.detail : err?.response?.data?.detail
  const requestId: string | null =
    typeof err?.requestId === 'string' && err.requestId
      ? err.requestId
      : typeof err?.response?.data?.request_id === 'string'
        ? err.response.data.request_id
        : null

  let kind: OpsErrorKind
  if (isAbort(err)) kind = 'aborted'
  else if (status !== null) kind = kindOfStatus(status)
  else if (err?.code === 'ECONNABORTED' || err?.code === 'ETIMEDOUT') kind = 'timeout'
  else if (err?.code === 'ERR_NETWORK' || err?.message === 'Network Error') kind = 'network'
  // The interceptor answers a 401 by sending the user to the login page and rejecting with this message.
  else if (err?.message === 'Authentication required') kind = 'unauthorized'
  else kind = 'unknown'

  const refusal = refusalOf(rawDetail)
  const detail = detailText(rawDetail)
  return {
    kind,
    status,
    messageKey: refusal ? `operations.refusal.${refusal}` : `operations.error.${kind}`,
    // A 5xx detail is generic by contract, but it adds nothing to the mapped message.
    detail: kind === 'server' ? null : detail,
    requestId,
    refusal
  }
}

/** Whether retrying the same request can help (backoff applies), as opposed to a definite answer. */
export const isRetryable = (e: OpsError) =>
  e.kind === 'server' || e.kind === 'unavailable' || e.kind === 'timeout' || e.kind === 'network' || e.kind === 'rate_limited'
