import { useTranslation } from 'react-i18next'
import { ShieldAlertIcon, XIcon } from 'lucide-react'
import { useActingWorkspace } from '@/hooks/useActingWorkspace'
import { useSettingsStore } from '@/stores/settings'

/**
 * "Acting on <ws>" (OQ-13): shown under the header on every tab while an
 * admin's Documents calls go to another workspace. It cannot be dismissed,
 * only ended with Exit, so the mode is never on without the admin seeing it.
 * No entrance animation: it is there on first paint (§2.12).
 */
export function ActingBanner({
  workspace,
  onExit,
  onOpenDocuments
}: {
  workspace: string
  onExit: () => void
  /** Offered when another tab is showing. */
  onOpenDocuments?: () => void
}) {
  const { t } = useTranslation()
  return (
    <div
      role="region"
      aria-label={t('admin.acting.label')}
      data-acting-workspace={workspace}
      className="flex flex-wrap items-center gap-x-3 gap-y-1.5 border-y-2 border-amber-500 bg-amber-100 px-4 py-1.5 text-[13px] text-amber-950 dark:border-amber-400 dark:bg-amber-950 dark:text-amber-50"
    >
      <ShieldAlertIcon className="size-4 flex-none" aria-hidden="true" />
      <p className="m-0 min-w-0 flex-1">
        <strong className="font-semibold">{t('admin.acting.title', { workspace })}</strong>{' '}
        <span>{t('admin.acting.body')}</span>
      </p>
      <span className="inline-flex flex-wrap gap-2">
        {onOpenDocuments ? (
          <button
            type="button"
            onClick={onOpenDocuments}
            className="min-h-7 rounded-md border border-amber-600 px-2.5 text-xs font-medium hover:bg-amber-200 dark:border-amber-300 dark:hover:bg-amber-900"
          >
            {t('admin.acting.openDocuments')}
          </button>
        ) : null}
        <button
          type="button"
          onClick={onExit}
          className="inline-flex min-h-7 items-center gap-1 rounded-md bg-amber-900 px-2.5 text-xs font-semibold text-amber-50 hover:bg-amber-800 dark:bg-amber-200 dark:text-amber-950 dark:hover:bg-amber-100"
        >
          <XIcon className="size-3.5" aria-hidden="true" />
          {t('admin.acting.exit')}
        </button>
      </span>
    </div>
  )
}

/** The banner wired to the acting store; renders nothing while Documents acts on the user's own workspace. */
export default function ActingWorkspaceBanner() {
  const { workspace, exit } = useActingWorkspace()
  const currentTab = useSettingsStore.use.currentTab()
  if (!workspace) return null
  return (
    <ActingBanner
      workspace={workspace}
      onExit={exit}
      onOpenDocuments={currentTab === 'documents' ? undefined : () => useSettingsStore.getState().setCurrentTab('documents')}
    />
  )
}
