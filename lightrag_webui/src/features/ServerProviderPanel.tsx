import { useCallback, useEffect, useState } from 'react'
import { useTranslation } from 'react-i18next'
import { toast } from 'sonner'
import { AlertTriangleIcon, InfoIcon, Undo2Icon } from 'lucide-react'

import { Alert, AlertDescription } from '@/components/ui/Alert'
import Button from '@/components/ui/Button'
import {
  Card,
  CardHeader,
  CardTitle,
  CardDescription,
  CardContent
} from '@/components/ui/Card'
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle
} from '@/components/ui/Dialog'
import {
  clearServerProvider,
  getServerProvider,
  type ProviderSlotKey,
  type ProviderSlotInput,
  type ServerProviderSlot,
  type ServerProviderView
} from '@/api/lightrag'
import { errorMessage } from '@/lib/utils'
import { PROVIDER_SLOT_META, getPreset, presetForBaseUrl } from '@/features/providerPresets'
import ProviderSlotFields, { type SlotDraft } from '@/features/ProviderSlotFields'
import OwnProviderWorkspacesDialog from '@/features/OwnProviderWorkspacesDialog'
import ServerProviderReviewDialog from '@/features/ServerProviderReviewDialog'
import { knownKeyPreview, sameBaseUrl } from '@/features/serverProviderKeys'

// The destructive color alone is too dark to read on the dark theme.
const DANGER_ALERT = 'border-red-500/50 text-red-600 dark:border-red-500/50 dark:text-red-400 [&>svg]:text-red-500'

/** Fields start on what's in use, so changing a model is one edit. */
function draftFromServerSlot(slot: ServerProviderSlot): SlotDraft {
  const preset = slot.preset_id
    ? getPreset(slot.preset_id)
    : presetForBaseUrl(slot.effective.host)
  return {
    presetId: preset.id,
    baseUrl: slot.effective.host ?? preset.baseUrl,
    model: slot.effective.model ?? '',
    apiKey: '',
    reasoning: slot.effective.reasoning_effort ?? ''
  }
}

interface ServerSlotCardProps {
  slotKey: ProviderSlotKey
  view: ServerProviderView
  onChanged: (next: ServerProviderView) => void
}

function ServerSlotCard({ slotKey, view, onChanged }: ServerSlotCardProps) {
  const { t } = useTranslation()
  const slot = view[slotKey]
  // Seeded once on mount; the parent remounts this card (via `key`) after each
  // change, so the draft re-seeds from the fresh view without an effect.
  const [draft, setDraft] = useState<SlotDraft>(() => draftFromServerSlot(slot))
  const [pending, setPending] = useState<ProviderSlotInput | null>(null)
  const [confirmingReturn, setConfirmingReturn] = useState(false)
  const [returning, setReturning] = useState(false)

  const baseUrl = draft.baseUrl.trim()
  const preview = knownKeyPreview(view, slotKey, baseUrl)
  const readOnly = view.status === 'no_secret'

  const unchanged =
    !draft.apiKey &&
    sameBaseUrl(baseUrl, slot.effective.host) &&
    draft.model.trim() === (slot.effective.model ?? '') &&
    draft.reasoning === (slot.effective.reasoning_effort ?? '')

  const handleReview = useCallback(() => {
    if (!baseUrl) {
      toast.error(t('providerSettings.errors.baseUrlRequired'))
      return
    }
    if (!draft.model.trim()) {
      toast.error(t('providerSettings.errors.modelRequired'))
      return
    }
    if (!draft.apiKey && !preview) {
      toast.error(t('providerSettings.server.keyRequiredForNewProvider'))
      return
    }
    setPending({
      base_url: baseUrl,
      model: draft.model.trim(),
      preset_id: draft.presetId,
      reasoning_effort: draft.reasoning, // '' => provider default
      ...(draft.apiKey ? { api_key: draft.apiKey } : {})
    })
  }, [baseUrl, draft, preview, t])

  const handleReturnToEnv = useCallback(async () => {
    setReturning(true)
    try {
      const next = await clearServerProvider(slotKey)
      toast.success(
        next.resumed
          ? t('providerSettings.server.returnedResumed', { count: next.resumed })
          : t('providerSettings.server.returned')
      )
      setConfirmingReturn(false)
      onChanged(next)
    } catch (err) {
      toast.error(errorMessage(err))
    } finally {
      setReturning(false)
    }
  }, [slotKey, onChanged, t])

  const fromWebui = slot.effective.source === 'webui'
  const total = view.impact.total_workspaces
  const meta = PROVIDER_SLOT_META.find((m) => m.key === slotKey)!

  return (
    <Card>
      <CardHeader>
        <CardTitle>{t(`providerSettings.${meta.titleKey}`)}</CardTitle>
        <CardDescription>
          {t(`providerSettings.${meta.descKey}`)}
          {slotKey === 'vision' && ` ${t('providerSettings.server.visionIncludesEmail')}`}
        </CardDescription>
      </CardHeader>
      <CardContent>
        <fieldset disabled={readOnly} className="flex flex-col gap-3">
          <span
            className={
              'inline-flex items-center gap-2 text-xs font-medium ' +
              (fromWebui ? 'text-emerald-500' : 'text-muted-foreground')
            }
          >
            <span
              className={'size-2 rounded-full ' + (fromWebui ? 'bg-emerald-500' : 'bg-zinc-400')}
            />
            {fromWebui
              ? t('providerSettings.server.sourceWebui')
              : t('providerSettings.server.sourceEnv')}
          </span>

          <p className="text-xs break-all text-muted-foreground">
            {t('providerSettings.effectiveLine', {
              host: slot.effective.host || '—',
              model: slot.effective.model || '—'
            })}
          </p>
          <p className="text-xs text-muted-foreground">
            {t('providerSettings.server.appliesTo', {
              affected: Math.max(total - view.impact.own_provider_by_slot[slotKey], 0),
              total
            })}
          </p>
          {fromWebui && (
            <p className="text-xs break-all text-muted-foreground">
              {t('providerSettings.server.envLine', {
                host: slot.env.host || '—',
                model: slot.env.model || '—'
              })}
            </p>
          )}

          <ProviderSlotFields
            slot={slotKey}
            draft={draft}
            setDraft={setDraft}
            apiKeyPlaceholder={
              preview
                ? t('providerSettings.apiKeyKeepPlaceholder', { preview })
                : t('providerSettings.server.keyRequiredPlaceholder')
            }
          />

          <div className="mt-1 flex flex-wrap gap-2">
            <Button onClick={handleReview} disabled={unchanged} size="sm">
              {t('providerSettings.server.review')}
            </Button>
            {fromWebui && (
              <Button onClick={() => setConfirmingReturn(true)} variant="outline" size="sm">
                <Undo2Icon className="size-4" />
                {t('providerSettings.server.returnToEnv')}
              </Button>
            )}
          </div>
        </fieldset>

        {pending && (
          <ServerProviderReviewDialog
            slotKey={slotKey}
            input={pending}
            view={view}
            onClose={() => setPending(null)}
            onApplied={onChanged}
          />
        )}

        <Dialog open={confirmingReturn} onOpenChange={setConfirmingReturn}>
          <DialogContent onCloseAutoFocus={(e) => e.preventDefault()}>
            <DialogHeader>
              <DialogTitle>{t('providerSettings.server.returnTitle')}</DialogTitle>
              <DialogDescription>
                {t('providerSettings.server.returnDescription', {
                  slot: t(`providerSettings.${meta.titleKey}`),
                  host: slot.env.host || '—',
                  model: slot.env.model || '—'
                })}
              </DialogDescription>
            </DialogHeader>
            <DialogFooter>
              <Button
                variant="outline"
                onClick={() => setConfirmingReturn(false)}
                disabled={returning}
              >
                {t('common.cancel')}
              </Button>
              <Button onClick={handleReturnToEnv} disabled={returning}>
                {t('providerSettings.server.returnToEnv')}
              </Button>
            </DialogFooter>
          </DialogContent>
        </Dialog>
      </CardContent>
    </Card>
  )
}

function pauseReason(kind: string | null, statusCode: number | null, t: (k: string, o?: any) => string) {
  return kind === 'budget_exhausted'
    ? t('providerSettings.server.pauseSpendLimit', { status: statusCode })
    : t('providerSettings.server.pauseInvalidKey', { status: statusCode })
}

interface ServerProviderPanelProps {
  onManageWorkspace: (workspace: string) => void
}

/** Admin-only: the provider every workspace without its own uses. */
export default function ServerProviderPanel({ onManageWorkspace }: ServerProviderPanelProps) {
  const { t } = useTranslation()
  const [showingOwn, setShowingOwn] = useState(false)
  const [view, setView] = useState<ServerProviderView | null>(null)
  const [revisions, setRevisions] = useState<Partial<Record<ProviderSlotKey, number>>>({})

  useEffect(() => {
    let cancelled = false
    getServerProvider()
      .then((next) => {
        if (!cancelled) setView(next)
      })
      .catch((err) => {
        toast.error(errorMessage(err) || t('providerSettings.errors.loadFailed'))
      })
    return () => {
      cancelled = true
    }
  }, [t])

  // Remounts only the card that changed, so what's typed in the others stays.
  const handleChanged = useCallback((slot: ProviderSlotKey, next: ServerProviderView) => {
    setView(next)
    setRevisions((r) => ({ ...r, [slot]: (r[slot] ?? 0) + 1 }))
  }, [])

  if (!view) {
    return <p className="text-sm text-muted-foreground">{t('providerSettings.loading')}</p>
  }

  const pauseReasons = view.paused.reasons
    .map((r) => pauseReason(r.kind, r.status_code, t))
    .join('; ')

  return (
    <>
      {view.status === 'unreadable' && (
        <Alert variant="destructive" className={DANGER_ALERT}>
          <AlertTriangleIcon className="size-4" />
          <AlertDescription>{t('providerSettings.server.unreadable')}</AlertDescription>
        </Alert>
      )}

      {view.status === 'no_secret' && (
        <Alert variant="destructive" className={DANGER_ALERT}>
          <AlertTriangleIcon className="size-4" />
          <AlertDescription>{t('providerSettings.server.noSecret')}</AlertDescription>
        </Alert>
      )}

      {view.paused.total > 0 && (
        <Alert variant="destructive" className={DANGER_ALERT}>
          <AlertTriangleIcon className="size-4" />
          <AlertDescription>
            {t('providerSettings.server.paused', {
              count: view.paused.total,
              reason: pauseReasons,
              since: view.paused.since
                ? new Date(view.paused.since * 1000).toLocaleString()
                : '—'
            })}
          </AlertDescription>
        </Alert>
      )}

      <Alert>
        <InfoIcon className="size-4" />
        <AlertDescription>
          <p>
            {t('providerSettings.server.impact', { count: view.impact.total_workspaces })}{' '}
            {t('providerSettings.server.notCovered')}
          </p>
          {view.impact.own_provider_total > 0 && (
            <button
              type="button"
              className="mt-1 cursor-pointer text-xs underline underline-offset-2"
              onClick={() => setShowingOwn(true)}
            >
              {t('providerSettings.server.viewOwn', { count: view.impact.own_provider_total })}
            </button>
          )}
        </AlertDescription>
      </Alert>

      {showingOwn && (
        <OwnProviderWorkspacesDialog
          onClose={() => setShowingOwn(false)}
          onManage={onManageWorkspace}
        />
      )}

      <div className="grid gap-4 lg:grid-cols-3">
        {PROVIDER_SLOT_META.map(({ key }) => (
          <ServerSlotCard
            key={`${key}-${revisions[key] ?? 0}`}
            slotKey={key}
            view={view}
            onChanged={(next) => handleChanged(key, next)}
          />
        ))}
      </div>

      {view.updated_at && (
        <p className="text-xs text-muted-foreground">
          {t('providerSettings.lastUpdated', {
            at: new Date(view.updated_at).toLocaleString(),
            by: view.updated_by || '—'
          })}
        </p>
      )}
    </>
  )
}
