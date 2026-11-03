import { useState, useEffect, useCallback } from 'react'
import { useTranslation } from 'react-i18next'
import { toast } from 'sonner'

import Button from '@/components/ui/Button'
import {
  Card,
  CardHeader,
  CardTitle,
  CardDescription,
  CardContent
} from '@/components/ui/Card'
import { SaveIcon, Trash2Icon } from 'lucide-react'

import {
  getProviderConfig,
  getServerProviderAvailability,
  updateProviderConfig,
  deleteProviderConfig,
  getEffectiveRoleConfig,
  type ProviderSlotMasked,
  type ProviderSlotKey,
  type EffectiveRolesResponse
} from '@/api/lightrag'
import DeleteWorkspaceCard from '@/components/DeleteWorkspaceCard'
import { useProvidersStore } from '@/stores/providers'
import { useAuthStore } from '@/stores/state'
import { errorMessage } from '@/lib/utils'
import { PROVIDER_SLOT_META, getPreset, presetForBaseUrl } from '@/features/providerPresets'
import ProviderSlotFields, { type SlotDraft } from '@/features/ProviderSlotFields'
import ManageWorkspaceSelector from '@/features/ManageWorkspaceSelector'
import ServerProviderPanel from '@/features/ServerProviderPanel'

function draftFromMasked(masked: ProviderSlotMasked): SlotDraft {
  const preset = masked.preset_id
    ? getPreset(masked.preset_id)
    : presetForBaseUrl(masked.base_url)
  return {
    presetId: preset.id,
    baseUrl: masked.base_url ?? preset.baseUrl,
    model: masked.model ?? '',
    apiKey: '',
    reasoning: masked.reasoning_effort ?? ''
  }
}

interface SlotFormProps {
  slot: ProviderSlotKey
  masked: ProviderSlotMasked
  targetWorkspace: string | null
  onSaved: () => void
}

function ProviderSlotForm({ slot, masked, targetWorkspace, onSaved }: SlotFormProps) {
  const { t } = useTranslation()
  // Seeded once on mount; the parent remounts this form (via `key`) after each
  // reload, so the draft re-seeds from the fresh masked config without an effect.
  const [draft, setDraft] = useState<SlotDraft>(() => draftFromMasked(masked))
  const [test, setTest] = useState(false)
  const [busy, setBusy] = useState(false)

  const handleSave = useCallback(async () => {
    if (!draft.baseUrl.trim()) {
      toast.error(t('providerSettings.errors.baseUrlRequired'))
      return
    }
    if (!draft.model.trim()) {
      toast.error(t('providerSettings.errors.modelRequired'))
      return
    }
    if (!draft.apiKey && !masked.api_key_set) {
      toast.error(t('providerSettings.errors.apiKeyRequired'))
      return
    }
    setBusy(true)
    try {
      await updateProviderConfig(
        {
          [slot]: {
            base_url: draft.baseUrl.trim(),
            model: draft.model.trim(),
            preset_id: draft.presetId,
            reasoning_effort: draft.reasoning, // '' => provider default
            ...(draft.apiKey ? { api_key: draft.apiKey } : {})
          }
        },
        test,
        targetWorkspace
      )
      toast.success(t('providerSettings.saved'))
      onSaved()
    } catch (err: any) {
      toast.error(err?.message ? errorMessage(err) : t('providerSettings.errors.saveFailed'))
    } finally {
      setBusy(false)
    }
  }, [draft, slot, test, masked.api_key_set, targetWorkspace, onSaved, t])

  const handleClear = useCallback(async () => {
    if (!window.confirm(t('providerSettings.clearConfirm'))) return
    setBusy(true)
    try {
      await deleteProviderConfig(slot, targetWorkspace)
      toast.success(t('providerSettings.cleared'))
      onSaved()
    } catch (err: any) {
      toast.error(err?.message ? errorMessage(err) : t('providerSettings.errors.clearFailed'))
    } finally {
      setBusy(false)
    }
  }, [slot, targetWorkspace, onSaved, t])

  return (
    <div className="flex flex-col gap-3">
      <div className="flex items-center justify-between">
        <span
          className={
            'inline-flex items-center gap-2 text-xs font-medium ' +
            (masked.active ? 'text-emerald-500' : 'text-muted-foreground')
          }
        >
          <span
            className={'size-2 rounded-full ' + (masked.active ? 'bg-emerald-500' : 'bg-zinc-400')}
          />
          {masked.active
            ? t('providerSettings.status.custom')
            : t('providerSettings.status.systemDefault')}
        </span>
      </div>

      {masked.effective && (
        <p className="text-xs text-muted-foreground">
          {t('providerSettings.effectiveLine', {
            host: masked.effective.host || '—',
            model: masked.effective.model || '—'
          })}
        </p>
      )}

      <ProviderSlotFields
        slot={slot}
        draft={draft}
        setDraft={setDraft}
        apiKeyPlaceholder={
          masked.api_key_set
            ? t('providerSettings.apiKeyKeepPlaceholder', { preview: masked.api_key_preview })
            : t('providerSettings.apiKeyPlaceholder')
        }
      />

      <label className="mt-1 inline-flex items-center gap-2 text-xs text-muted-foreground">
        <input type="checkbox" checked={test} onChange={(e) => setTest(e.target.checked)} />
        {t('providerSettings.testConnection')}
      </label>

      <div className="mt-1 flex gap-2">
        <Button onClick={handleSave} disabled={busy} size="sm">
          <SaveIcon className="size-4" />
          {t('providerSettings.save')}
        </Button>
        <Button onClick={handleClear} disabled={busy || !masked.active} variant="outline" size="sm">
          <Trash2Icon className="size-4" />
          {t('providerSettings.clear')}
        </Button>
      </div>
    </div>
  )
}

export default function ProviderSettings() {
  const { t } = useTranslation()
  const { role } = useAuthStore()
  const isAdmin = role === 'admin'
  const [targetWorkspace, setTargetWorkspace] = useState<string | null>(null)
  const [serverDefaultAvailable, setServerDefaultAvailable] = useState(false)
  const [serverDefaultSelected, setServerDefaultSelected] = useState(false)
  const config = useProvidersStore.use.config()
  const setConfig = useProvidersStore.use.setConfig()
  const [loading, setLoading] = useState(false)
  const [disabled, setDisabled] = useState(false)
  const [effective, setEffective] = useState<EffectiveRolesResponse | null>(null)
  const [reloadKey, setReloadKey] = useState(0)

  const reload = useCallback(async () => {
    setLoading(true)
    try {
      const data = await getProviderConfig(targetWorkspace)
      setConfig(data)
      setDisabled(false)
      setReloadKey((k) => k + 1)
      try {
        setEffective(await getEffectiveRoleConfig(targetWorkspace))
      } catch {
        setEffective(null)
      }
    } catch (err: any) {
      if (typeof err?.message === 'string' && err.message.includes('503')) {
        setDisabled(true)
        setConfig(null)
      } else {
        toast.error(err?.message ? errorMessage(err) : t('providerSettings.errors.loadFailed'))
      }
    } finally {
      setLoading(false)
    }
  }, [setConfig, targetWorkspace, t])

  useEffect(() => {
    // eslint-disable-next-line react-hooks/set-state-in-effect
    reload()
  }, [reload])

  useEffect(() => {
    if (!isAdmin) return
    let cancelled = false
    getServerProviderAvailability()
      .then(() => {
        if (!cancelled) setServerDefaultAvailable(true)
      })
      .catch(() => {
        // Not available on this server, or not a signed-in admin
      })
    return () => {
      cancelled = true
    }
  }, [isAdmin])

  return (
    <div className="mx-auto flex max-w-5xl flex-col gap-4 overflow-auto p-6">
      <div>
        <h1 className="text-xl font-bold">{t('providerSettings.title')}</h1>
        <p className="text-sm text-muted-foreground">
          {serverDefaultSelected
            ? t('providerSettings.server.subtitle')
            : t('providerSettings.subtitle')}
        </p>
      </div>

      {isAdmin && (
        <ManageWorkspaceSelector
          value={targetWorkspace}
          onChange={setTargetWorkspace}
          serverDefaultAvailable={serverDefaultAvailable}
          serverDefaultSelected={serverDefaultSelected}
          onServerDefaultSelectedChange={setServerDefaultSelected}
        />
      )}

      {serverDefaultSelected ? <ServerProviderPanel
        onManageWorkspace={(workspace) => {
          setServerDefaultSelected(false)
          setTargetWorkspace(workspace)
        }}
      /> : (
        <>
          {disabled && (
            <Card>
              <CardContent className="pt-6">
                <p className="text-sm text-muted-foreground">{t('providerSettings.disabledNotice')}</p>
              </CardContent>
            </Card>
          )}

          {config && (
            <div className="grid gap-4 lg:grid-cols-3">
              {PROVIDER_SLOT_META.map(({ key, titleKey, descKey }) => (
                <Card key={key}>
                  <CardHeader>
                    <CardTitle>{t(`providerSettings.${titleKey}`)}</CardTitle>
                    <CardDescription>{t(`providerSettings.${descKey}`)}</CardDescription>
                  </CardHeader>
                  <CardContent>
                    <ProviderSlotForm
                      key={`${key}-${reloadKey}`}
                      slot={key}
                      masked={config[key]}
                      targetWorkspace={targetWorkspace}
                      onSaved={reload}
                    />
                  </CardContent>
                </Card>
              ))}
            </div>
          )}

          {config?.updated_at && (
            <p className="text-xs text-muted-foreground">
              {t('providerSettings.lastUpdated', {
                at: new Date(config.updated_at).toLocaleString(),
                by: config.updated_by || '—'
              })}
            </p>
          )}

          {effective && (
            <details className="rounded-md border border-border/60 p-3 text-sm">
              <summary className="cursor-pointer font-medium">
                {t('providerSettings.effective.title')}
              </summary>
              <table className="mt-2 w-full text-xs">
                <thead className="text-muted-foreground">
                  <tr className="text-left">
                    <th className="py-1 pr-3 font-medium">{t('providerSettings.effective.role')}</th>
                    <th className="py-1 pr-3 font-medium">{t('providerSettings.effective.host')}</th>
                    <th className="py-1 pr-3 font-medium">{t('providerSettings.effective.model')}</th>
                    <th className="py-1 pr-3 font-medium">{t('providerSettings.reasoning.label')}</th>
                    <th className="py-1 font-medium">{t('providerSettings.effective.source')}</th>
                  </tr>
                </thead>
                <tbody>
                  {Object.entries(effective.roles).map(([roleName, cfg]) => (
                    <tr key={roleName} className="border-t border-border/40">
                      <td className="py-1 pr-3 font-mono">{roleName}</td>
                      <td className="py-1 pr-3 font-mono break-all">{cfg.host || '—'}</td>
                      <td className="py-1 pr-3 font-mono break-all">{cfg.model || '—'}</td>
                      <td className="py-1 pr-3 font-mono">{cfg.reasoning_effort || '—'}</td>
                      <td className="py-1">
                        <span
                          className={
                            cfg.source === 'custom' ? 'text-emerald-500' : 'text-muted-foreground'
                          }
                        >
                          {cfg.source === 'custom'
                            ? t('providerSettings.status.custom')
                            : t('providerSettings.status.systemDefault')}
                        </span>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </details>
          )}

          {loading && (
            <p className="text-sm text-muted-foreground">{t('providerSettings.loading')}</p>
          )}

          {/* Owner-only and never for a workspace an admin is managing */}
          {targetWorkspace === null && <DeleteWorkspaceCard />}
        </>
      )}
    </div>
  )
}
