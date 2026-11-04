import { useTranslation } from 'react-i18next'
import { opsApi } from '../api'
import { Panel } from '../components/Panel'
import { StatusChip } from '../components/StatusChip'
import { fmtCount } from '../format'
import { ChartBlock, SeriesTable, TimeBars } from '../performance/charts'
import { SeriesDef, spanEnding } from '../performance/chartKit'
import { SigninCounts, signinState } from '../performance/model'
import { MetricsOff } from '../performance/shared'
import { useOpsPoll } from '../poll'

/** The auth counters flush once a minute (§3.6 "Sign-in health"). */
const SIGNIN_SECONDS = 60
const DAY_MS = 24 * 3600_000

type Counter = 'sso_errors' | 'token_401' | 'lockouts'
const COUNTERS: Counter[] = ['sso_errors', 'token_401', 'lockouts']

function Counts({ data, label }: { data: SigninCounts | null; label: string }) {
  const { t } = useTranslation()
  return (
    <div className="flex min-w-0 flex-col gap-1.5 rounded-lg border p-3">
      <span className="text-muted-foreground text-xs font-medium">{label}</span>
      <dl className="m-0 grid grid-cols-3 gap-2">
        {COUNTERS.map((c) => (
          <div key={c} className="flex min-w-0 flex-col">
            <dt className="text-muted-foreground truncate text-xs">{t(`operations.signin.${c}`)}</dt>
            <dd className="m-0 font-mono text-lg font-semibold">{fmtCount(data ? data[c] : null)}</dd>
          </div>
        ))}
      </dl>
    </div>
  )
}

/**
 * Health › Sign-in (§2.6, FR-35): SSO callback errors, token validations
 * that ended in 401 and login-throttle lockouts, for the last 15 minutes and
 * the last 24 hours, from the `auth` metric counters (one request: the 24 h
 * answer carries its last 15 minutes). The counters record outcomes only,
 * never usernames. The state is the server's signin_failures_15m verdict
 * (default: warning above 5 SSO errors or 3 lockouts, serious above 20;
 * editable in the thresholds KV).
 */
export default function SigninView({ metricsOff }: { metricsOff: boolean }) {
  const { t } = useTranslation()
  const day = useOpsPoll((signal) => opsApi.signin('24h', signal), [], SIGNIN_SECONDS)
  const off = metricsOff || day.data?.enabled === false
  // The server's verdict carries any KV threshold overrides; the defaults are the fallback.
  const state = off || day.data?.available === false ? 'unknown' : (day.data?.state ?? signinState(day.data?.last_15m))
  const series: SeriesDef[] = COUNTERS.map((c, i) => ({ key: c, label: t(`operations.signin.${c}`), color: `var(--ops-s${i + 1})` }))
  return (
    <div className="flex flex-col gap-4">
      {off ? <MetricsOff /> : null}
      <Panel
        poll={day}
        freshness={SIGNIN_SECONDS}
        title={t('operations.signin.title')}
        sub={t('operations.signin.sub')}
        actions={<StatusChip state={state}>{t(`operations.signin.state.${state}`)}</StatusChip>}
      >
        {(data) => (
          <div className="grid grid-cols-1 gap-3 md:grid-cols-2">
            <Counts data={data.last_15m} label={t('operations.signin.last15m')} />
            <Counts data={data} label={t('operations.signin.last24h')} />
          </div>
        )}
      </Panel>
      <Panel poll={day} freshness={SIGNIN_SECONDS} title={t('operations.signin.chartTitle')}>
        {(data) => {
          const { from, to } = spanEnding(DAY_MS, data.generated_at, day.updatedAt)
          const rows = data.series.map((p) => ({ t: p.t, sso_errors: p.sso_errors, token_401: p.token_401, lockouts: p.lockouts }))
          const count = (v: number) => fmtCount(v)
          return (
            <ChartBlock
              title={t('operations.signin.perBucket')}
              series={series}
              empty={rows.length ? undefined : t('operations.signin.none')}
              chart={<TimeBars rows={rows} series={series} from={from} to={to} format={count} label={t('operations.signin.chartTitle')} />}
              table={<SeriesTable rows={rows} series={series} format={count} caption={t('operations.signin.chartTitle')} />}
            />
          )
        }}
      </Panel>
      <p className="text-muted-foreground m-0 text-xs">{t('operations.signin.privacy')}</p>
    </div>
  )
}
