import { useTranslation } from 'react-i18next'
import { ServerIcon } from 'lucide-react'
import { OpsStatus } from '../api'
import { lookOf, toProcessMode, workersState } from '../severity'
import { ToneChip } from './StatusChip'

/**
 * "uvicorn · 1 process" or "gunicorn · 2/3 workers". In single-process mode
 * the multi-worker features (orphan detection, release busy) hide themselves.
 * The worker count gets a tone only once live heartbeats are reported.
 */
export default function ProcessChip({ process }: { process: OpsStatus['process'] }) {
  const { t } = useTranslation()
  const mode = toProcessMode(process.mode)
  const expected = typeof process.expected_workers === 'number' ? process.expected_workers : null
  const live = typeof process.workers_live === 'number' ? process.workers_live : null

  let text: string
  if (mode === 'uvicorn') text = t('operations.process.uvicorn')
  else if (mode === 'gunicorn' && expected !== null && live !== null) text = t('operations.process.gunicornLive', { live, expected })
  else if (mode === 'gunicorn' && expected !== null) text = t('operations.process.gunicorn', { count: expected })
  else text = t('operations.process.unknown')

  const state = mode === 'gunicorn' ? workersState(expected, live) : null
  if (state && state !== 'unknown') {
    const look = lookOf(state)
    return (
      <ToneChip tone={look.tone} icon={look.icon}>
        {text}
        <span className="sr-only"> ({t(look.label)})</span>
      </ToneChip>
    )
  }
  return (
    <span className="text-muted-foreground inline-flex items-center gap-1.5 rounded-md border px-2 py-0.5 text-xs font-medium">
      <ServerIcon className="size-3.5 flex-none" aria-hidden="true" />
      {text}
    </span>
  )
}
