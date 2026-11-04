import { axiosInstance } from '@/api/lightrag'

/**
 * Operations API client. Every route lives under /admin/ops behind the
 * server's router-wide admin check; responses carry `generated_at` and errors
 * are `{detail, request_id}` (spec §3.7). Requests go through the shared
 * axios instance so auth headers, token renewal and the error mapping apply.
 */

export const OPS_PREFIX = '/admin/ops'

export type Params = Record<string, string | number | boolean | undefined | null>

export const clean = (params: Params = {}) =>
  Object.fromEntries(Object.entries(params).filter(([, v]) => v !== undefined && v !== null && v !== ''))

export const get = async <T>(path: string, params?: Params, signal?: AbortSignal): Promise<T> =>
  (await axiosInstance.get(`${OPS_PREFIX}${path}`, { params: clean(params), signal })).data

export const post = async <T>(path: string, body?: unknown, signal?: AbortSignal): Promise<T> =>
  (await axiosInstance.post(`${OPS_PREFIX}${path}`, body ?? {}, { signal })).data

/** Feature switches reported by /status; each view shows "not available" instead of failing. */
export type OpsFeatures = {
  durable: boolean
  runs: boolean
  logs: boolean
  probes: boolean
  alerts: boolean
  insights: boolean
  release_busy: boolean
  gpu: boolean
}

export type OpsStatus = {
  /** Ops data needs PostgreSQL KV storage; false means views stay unavailable. */
  available: boolean
  features: Partial<OpsFeatures>
  process: {
    mode: 'uvicorn' | 'gunicorn' | string
    expected_workers: number | null
    /** Live workers from heartbeats; null until worker heartbeats exist. */
    workers_live: number | null
  }
  /** May use forceful actions (release busy, reveal, export); all admins when OPS_OPERATOR_ACCOUNTS is unset. */
  operator: boolean
  version: string | null
  generated_at: string
}

export const opsApi = {
  status: (signal?: AbortSignal) => get<OpsStatus>('/status', undefined, signal)
}
