import { useMemo } from 'react'
import { opsApi } from '../api'
import { useStale } from '../hooks'
import { FlowRates, PerfDeps, PerfHttp, PerfLlm, PerfWindow, flowRatesFrom } from '../performance/model'
import { useOpsPoll } from '../poll'

/** The current minute is re-read with the Performance cadence (§3.10). */
export const FLOW_REFRESH_SECONDS = 60
/** Each edge's 24 h p50/p90 moves slowly; the server caches the 24 h answers too. */
export const FLOW_BASELINE_SECONDS = 15 * 60

type Sources = { http: PerfHttp | null; deps: PerfDeps | null; llm: PerfLlm | null }

/**
 * One request per source, settled together: a source that fails or is
 * switched off (OPS_METRICS_ENABLED=false, no Postgres) leaves only its own
 * edges still. Every request is a server-side cached aggregate, never a
 * per-workspace call.
 */
async function readSources(window: PerfWindow, signal: AbortSignal, metrics: boolean): Promise<Sources> {
  const [http, deps, llm] = await Promise.allSettled([
    metrics ? opsApi.perfHttp(window, signal) : Promise.resolve(null),
    metrics ? opsApi.perfDeps(window, signal) : Promise.resolve(null),
    opsApi.perfLlm(window, 'purpose', signal)
  ])
  const value = <T>(r: PromiseSettledResult<T | null>) => (r.status === 'fulfilled' ? r.value : null)
  if (http.status === 'rejected' && deps.status === 'rejected' && llm.status === 'rejected') throw http.reason
  return { http: value(http), deps: value(deps), llm: value(llm) }
}

/**
 * Rates for the topology's flowing edges: the latest minute from the 15 m
 * answers (every 60 s) against thresholds from the 24 h answers (every
 * 15 min). Only polls while the Dependencies view is visible. `stale` turns
 * on when the current rates are older than twice their cadence or the last
 * refresh failed, which stops the dashes (§2.12 rule 3).
 */
export function useFlowRates(enabled: boolean, metrics: boolean): { rates: FlowRates | null; stale: boolean } {
  const current = useOpsPoll((signal) => readSources('15m', signal, metrics), [metrics], FLOW_REFRESH_SECONDS, { enabled })
  const baseline = useOpsPoll((signal) => readSources('24h', signal, metrics), [metrics], FLOW_BASELINE_SECONDS, { enabled })
  const { stale } = useStale(current.updatedAt, FLOW_REFRESH_SECONDS)
  const rates = useMemo(() => {
    if (!current.data || !current.updatedAt) return null
    // Bucket times are server times: measure "latest complete minute" on the server's clock when it says.
    const d = current.data
    const said = Date.parse(d.http?.generated_at ?? d.deps?.generated_at ?? d.llm?.generated_at ?? '')
    const r = flowRatesFrom(d, baseline.data ?? { http: null, deps: null, llm: null }, Number.isFinite(said) ? said : current.updatedAt)
    return Object.keys(r).length ? r : null
  }, [current.data, current.updatedAt, baseline.data])
  return { rates, stale: stale || !!current.error }
}
