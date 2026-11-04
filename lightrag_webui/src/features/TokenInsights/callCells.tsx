import { useTranslation } from 'react-i18next'
import { writeOpsHash } from '@/features/Operations/hashState'
import { logsView, pivotFilter } from '@/features/Operations/logs'
import { useSettingsStore } from '@/stores/settings'

/**
 * Opens the Operations Logs tab filtered to one request ID (the 7-day
 * window). The Operations chunk is loaded first, so it reads the link while
 * it renders, before this page's unmount resets the URL.
 */
async function openRequestLogs(id: string) {
  await import('@/features/Operations')
  window.history.replaceState(null, '', writeOpsHash(logsView(pivotFilter('request_id', id))))
  useSettingsStore.getState().setCurrentTab('operations')
}

/** The call's request ID, short, opening its log lines in Operations; "–" for calls without one (background work, older rows). */
export function RequestCell({ id }: { id?: string | null }) {
  const { t } = useTranslation()
  if (!id) return <span className="text-muted-foreground">–</span>
  return (
    <button
      type="button"
      className="hover:bg-muted min-h-7 rounded px-1 font-mono text-xs underline underline-offset-2"
      title={t('insights.explore.requestHint', { id })}
      onClick={() => void openRequestLogs(id)}
    >
      {id.length > 10 ? `${id.slice(0, 8)}…` : id}
    </button>
  )
}
