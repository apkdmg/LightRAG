import { useTranslation } from 'react-i18next'
import { Card, Kv, Title } from '@/features/TokenInsights/ui'
import { OpsFeatures } from './api'
import ComingLater from './components/ComingLater'
import ProcessChip from './components/ProcessChip'
import { ToneChip } from './components/StatusChip'
import { useOpsShell } from './context'
import { fmtClock } from './format'
import { isMultiWorker } from './severity'

const FEATURE_ORDER: (keyof OpsFeatures)[] = ['durable', 'runs', 'logs', 'probes', 'alerts', 'release_busy', 'gpu', 'insights']

/** Why a feature is off: Insights can be switched off, release busy needs Gunicorn, the rest ships later. */
const offLabel = (key: keyof OpsFeatures, singleProcess: boolean) => {
  if (key === 'insights') return 'operations.feature.off'
  if (key === 'release_busy' && singleProcess) return 'operations.feature.multiWorkerOnly'
  return 'operations.feature.later'
}

/** What /status reports about this server, shown until the full overview ships. */
function ServerCard() {
  const { t } = useTranslation()
  const { status } = useOpsShell()
  if (!status) return null
  const features = status.features ?? {}
  const singleProcess = !isMultiWorker(status.process.mode)
  return (
    <Card className="flex max-w-[760px] flex-col gap-3">
      <Title sub={t('operations.system.checked', { time: fmtClock(status.generated_at) })}>{t('operations.system.title')}</Title>
      <div className="flex flex-col gap-1.5">
        <Kv k={t('operations.system.mode')}>
          <ProcessChip process={status.process} />
        </Kv>
        <Kv k={t('operations.system.workers')}>
          {typeof status.process.workers_live === 'number'
            ? t('operations.system.workersLive', { live: status.process.workers_live, expected: status.process.expected_workers ?? '–' })
            : t('operations.system.workersPending')}
        </Kv>
        <Kv k={t('operations.system.storage')}>
          {status.available ? (
            <ToneChip tone="ok" icon="check">{t('operations.system.storageOk')}</ToneChip>
          ) : (
            <ToneChip tone="unknown" icon="dashed-circle">{t('operations.system.storageMissing')}</ToneChip>
          )}
        </Kv>
        <Kv k={t('operations.system.access')}>{status.operator ? t('operations.system.operator') : t('operations.system.adminOnly')}</Kv>
        {status.version ? <Kv k={t('operations.system.version')}>{status.version}</Kv> : null}
      </div>
      <div>
        <h3 className="m-0 mb-1.5 text-[13px] font-semibold">{t('operations.system.features')}</h3>
        <ul className="m-0 grid list-none grid-cols-1 gap-1 p-0 sm:grid-cols-2">
          {FEATURE_ORDER.map((key) => (
            <li key={key} className="flex items-center justify-between gap-3 text-xs">
              <span className="text-muted-foreground">{t(`operations.feature.${key}`)}</span>
              {features[key] ? (
                <ToneChip tone="ok" icon="check">{t('operations.feature.on')}</ToneChip>
              ) : (
                <ToneChip tone="unknown" icon="dashed-circle">{t(offLabel(key, singleProcess))}</ToneChip>
              )}
            </li>
          ))}
        </ul>
      </div>
    </Card>
  )
}

export default function OverviewTab() {
  const { t } = useTranslation()
  return (
    <div className="flex flex-col gap-4">
      <ServerCard />
      <ComingLater title={t('operations.later.overview.title')} body={t('operations.later.overview.body')} />
    </div>
  )
}
