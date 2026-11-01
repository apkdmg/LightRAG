import { useCallback, useEffect, useState } from 'react'
import { useTranslation } from 'react-i18next'
import { toast } from 'sonner'
import { CheckCircle2Icon, Loader2Icon, XCircleIcon } from 'lucide-react'

import Button from '@/components/ui/Button'
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle
} from '@/components/ui/Dialog'
import {
  testServerProvider,
  updateServerProvider,
  type ProviderSlotInput,
  type ProviderSlotKey,
  type ServerProviderTestResult,
  type ServerProviderView
} from '@/api/lightrag'
import { errorMessage } from '@/lib/utils'
import { presetForBaseUrl } from '@/features/providerPresets'
import { knownKeyPreview } from '@/features/serverProviderKeys'

const providerLabel = (baseUrl: string | null | undefined): string => {
  const preset = presetForBaseUrl(baseUrl)
  return preset.baseUrl ? preset.label : baseUrl || '—'
}

const maskKey = (key: string): string => `...${key.slice(-4)}`

interface ServerProviderReviewDialogProps {
  slotKey: ProviderSlotKey
  input: ProviderSlotInput
  view: ServerProviderView
  onClose: () => void
  onApplied: (next: ServerProviderView) => void
}

/** Shows what would change and tests the provider. Nothing is saved until Apply. */
export default function ServerProviderReviewDialog({
  slotKey,
  input,
  view,
  onClose,
  onApplied
}: ServerProviderReviewDialogProps) {
  const { t } = useTranslation()
  const [test, setTest] = useState<ServerProviderTestResult | null>(null)
  const [testError, setTestError] = useState<string | null>(null)
  const [applying, setApplying] = useState(false)

  const slot = view[slotKey]
  const slotTitle = t(`providerSettings.${slotKey}Title`)

  useEffect(() => {
    let cancelled = false
    testServerProvider(slotKey, input)
      .then((result) => {
        if (!cancelled) setTest(result)
      })
      .catch((err) => {
        if (!cancelled) setTestError(errorMessage(err))
      })
    return () => {
      cancelled = true
    }
  }, [slotKey, input])

  const handleApply = useCallback(async () => {
    setApplying(true)
    try {
      const next = await updateServerProvider(slotKey, input)
      toast.success(
        next.resumed
          ? t('providerSettings.server.appliedResumed', { count: next.resumed })
          : t('providerSettings.server.applied')
      )
      onClose()
      onApplied(next)
    } catch (err) {
      // The dialog and the form stay as they are, so nothing typed is lost.
      toast.error(errorMessage(err))
      setApplying(false)
    }
  }, [slotKey, input, onClose, onApplied, t])

  const newKeyPreview = input.api_key
    ? maskKey(input.api_key)
    : knownKeyPreview(view, slotKey, input.base_url ?? '')
  const affected = Math.max(
    view.impact.total_workspaces - view.impact.own_provider[slotKey].length,
    0
  )
  const pausedHere = view.paused.filter((p) => p.slot === slotKey).length

  const rows: [string, string, string][] = [
    [
      t('providerSettings.provider'),
      providerLabel(slot.effective.host),
      providerLabel(input.base_url)
    ],
    [t('providerSettings.model'), slot.effective.model || '—', input.model || '—'],
    [
      t('providerSettings.reasoning.label'),
      t(`providerSettings.reasoning.options.${slot.effective.reasoning_effort || 'default'}`),
      t(`providerSettings.reasoning.options.${input.reasoning_effort || 'default'}`)
    ],
    [
      t('providerSettings.apiKey'),
      slot.effective.api_key_preview || '—',
      `${newKeyPreview || '—'}${input.api_key ? ` ${t('providerSettings.server.newKey')}` : ''}`
    ]
  ]

  return (
    <Dialog open onOpenChange={(open) => !open && !applying && onClose()}>
      <DialogContent className="sm:max-w-xl" onCloseAutoFocus={(e) => e.preventDefault()}>
        <DialogHeader>
          <DialogTitle>{t('providerSettings.server.reviewTitle', { slot: slotTitle })}</DialogTitle>
          <DialogDescription>{t('providerSettings.server.reviewDescription')}</DialogDescription>
        </DialogHeader>

        <table className="w-full text-sm">
          <thead className="text-xs text-muted-foreground">
            <tr className="text-left">
              <th className="py-1 pr-3 font-medium" />
              <th className="py-1 pr-3 font-medium">{t('providerSettings.server.now')}</th>
              <th className="py-1 font-medium">{t('providerSettings.server.after')}</th>
            </tr>
          </thead>
          <tbody>
            {rows.map(([label, now, after]) => (
              <tr key={label} className="border-t border-border/40">
                <td className="py-1 pr-3 text-muted-foreground">{label}</td>
                <td className="py-1 pr-3 break-all">{now}</td>
                <td className={'py-1 break-all ' + (now !== after ? 'font-medium' : '')}>
                  {after}
                </td>
              </tr>
            ))}
          </tbody>
        </table>

        <ul className="flex flex-col gap-1 text-sm text-muted-foreground">
          <li>
            {affected === view.impact.total_workspaces
              ? t('providerSettings.server.reviewAffectsAll', { count: affected })
              : t('providerSettings.server.reviewAffectsSome', {
                affected,
                total: view.impact.total_workspaces
              })}
          </li>
          <li>{t('providerSettings.server.reviewBilling')}</li>
          <li>{t('providerSettings.server.reviewTiming')}</li>
          {pausedHere > 0 && (
            <li>{t('providerSettings.server.reviewResumes', { count: pausedHere })}</li>
          )}
        </ul>

        <div className="rounded-md border border-border/60 p-3 text-sm">
          {!test && !testError && (
            <p className="inline-flex items-center gap-2 text-muted-foreground">
              <Loader2Icon className="size-4 animate-spin" />
              {t('providerSettings.server.testing')}
            </p>
          )}
          {testError && (
            <p className="inline-flex items-start gap-2 break-words text-red-500">
              <XCircleIcon className="mt-0.5 size-4 shrink-0" />
              <span>{testError}</span>
            </p>
          )}
          {test?.ok && (
            <p className="inline-flex items-center gap-2 text-emerald-500">
              <CheckCircle2Icon className="size-4" />
              {t('providerSettings.server.testPassed', {
                seconds: (test.latency_ms / 1000).toFixed(1)
              })}
            </p>
          )}
          {test && !test.ok && (
            <div className="flex items-start gap-2 text-red-500">
              <XCircleIcon className="mt-0.5 size-4 shrink-0" />
              <div className="min-w-0 break-words">
                <p>
                  {t(`providerSettings.server.testOutcome.${test.outcome}`, {
                    status: test.status_code
                  })}
                </p>
                {test.message && <p className="mt-1 text-xs opacity-80">{test.message}</p>}
              </div>
            </div>
          )}
        </div>

        <DialogFooter>
          <Button variant="outline" onClick={onClose} disabled={applying}>
            {t('providerSettings.server.back')}
          </Button>
          {test && !test.ok && test.can_apply_anyway && (
            <Button variant="outline" onClick={handleApply} disabled={applying}>
              {t('providerSettings.server.applyAnyway')}
            </Button>
          )}
          {(!test || test.ok) && (
            <Button onClick={handleApply} disabled={!test?.ok || applying}>
              {applying && <Loader2Icon className="size-4 animate-spin" />}
              {t('providerSettings.server.apply')}
            </Button>
          )}
        </DialogFooter>
      </DialogContent>
    </Dialog>
  )
}
