import { useState, useEffect, useCallback } from 'react'
import { useTranslation } from 'react-i18next'
import { toast } from 'sonner'

import Button from '@/components/ui/Button'
import Input from '@/components/ui/Input'
import {
  Card,
  CardHeader,
  CardTitle,
  CardDescription,
  CardContent
} from '@/components/ui/Card'
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue
} from '@/components/ui/Select'
import { SaveIcon, Trash2Icon, ExternalLinkIcon } from 'lucide-react'

import {
  getProviderConfig,
  updateProviderConfig,
  deleteProviderConfig,
  getEffectiveRoleConfig,
  listAdminWorkspaces,
  type ProviderSlotMasked,
  type EffectiveRolesResponse,
  type AdminWorkspace
} from '@/api/lightrag'
import { useProvidersStore } from '@/stores/providers'
import { useAuthStore } from '@/stores/state'
import {
  PROVIDER_PRESETS,
  getPreset,
  presetForBaseUrl,
  type ProviderPreset
} from '@/features/providerPresets'

type SlotKey = 'llm' | 'vision'

interface SlotDraft {
  presetId: string
  baseUrl: string
  model: string
  apiKey: string
}

function draftFromMasked(masked: ProviderSlotMasked): SlotDraft {
  const preset = masked.preset_id
    ? getPreset(masked.preset_id)
    : presetForBaseUrl(masked.base_url)
  return {
    presetId: preset.id,
    baseUrl: masked.base_url ?? preset.baseUrl,
    model: masked.model ?? '',
    apiKey: ''
  }
}

interface SlotFormProps {
  slot: SlotKey
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

  const preset: ProviderPreset = getPreset(draft.presetId)
  const suggestedModels =
    slot === 'vision' ? preset.suggestedVisionModels : preset.suggestedModels
  const datalistId = `provider-models-${slot}`

  const onPresetChange = useCallback((id: string) => {
    setDraft((d) => {
      const p = getPreset(id)
      return {
        ...d,
        presetId: id,
        // Adopt the preset base URL unless it's the free-form "custom" preset.
        baseUrl: p.editableBaseUrl ? d.baseUrl : p.baseUrl
      }
    })
  }, [])

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
            // Send the key only when the user typed a new one (keep existing otherwise).
            ...(draft.apiKey ? { api_key: draft.apiKey } : {})
          }
        },
        test,
        targetWorkspace
      )
      toast.success(t('providerSettings.saved'))
      onSaved()
    } catch (err: any) {
      toast.error(err?.message || t('providerSettings.errors.saveFailed'))
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
      toast.error(err?.message || t('providerSettings.errors.clearFailed'))
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
            className={
              'size-2 rounded-full ' +
              (masked.active ? 'bg-emerald-500' : 'bg-zinc-400')
            }
          />
          {masked.active
            ? t('providerSettings.status.custom')
            : t('providerSettings.status.systemDefault')}
        </span>
      </div>

      {/* Effective provider actually in use — shown even for the system default. */}
      {masked.effective && (
        <p className="text-xs text-muted-foreground">
          {t('providerSettings.effectiveLine', {
            host: masked.effective.host || '—',
            model: masked.effective.model || '—'
          })}
        </p>
      )}

      {/* Provider preset */}
      <label className="text-sm font-medium">{t('providerSettings.provider')}</label>
      <Select value={draft.presetId} onValueChange={onPresetChange}>
        <SelectTrigger>
          <SelectValue />
        </SelectTrigger>
        <SelectContent>
          {PROVIDER_PRESETS.map((p) => (
            <SelectItem key={p.id} value={p.id}>
              {p.label}
            </SelectItem>
          ))}
        </SelectContent>
      </Select>

      {/* Base URL — read-only unless "custom" */}
      <label className="text-sm font-medium">{t('providerSettings.baseUrl')}</label>
      <Input
        value={draft.baseUrl}
        onChange={(e) => setDraft((d) => ({ ...d, baseUrl: e.target.value }))}
        placeholder="https://api.example.com/v1"
        readOnly={!preset.editableBaseUrl}
        className={!preset.editableBaseUrl ? 'opacity-70' : ''}
        autoComplete="off"
      />

      {/* Model (editable dropdown via datalist) */}
      <label className="text-sm font-medium">{t('providerSettings.model')}</label>
      <Input
        value={draft.model}
        onChange={(e) => setDraft((d) => ({ ...d, model: e.target.value }))}
        placeholder={t('providerSettings.modelPlaceholder')}
        list={suggestedModels.length ? datalistId : undefined}
        autoComplete="off"
      />
      {suggestedModels.length > 0 && (
        <datalist id={datalistId}>
          {suggestedModels.map((m) => (
            <option key={m} value={m} />
          ))}
        </datalist>
      )}

      {/* API key (password) */}
      <label className="text-sm font-medium">{t('providerSettings.apiKey')}</label>
      <Input
        type="password"
        value={draft.apiKey}
        onChange={(e) => setDraft((d) => ({ ...d, apiKey: e.target.value }))}
        placeholder={
          masked.api_key_set
            ? t('providerSettings.apiKeyKeepPlaceholder', { preview: masked.api_key_preview })
            : t('providerSettings.apiKeyPlaceholder')
        }
        autoComplete="off"
      />
      {preset.apiKeyHelpUrl && (
        <a
          href={preset.apiKeyHelpUrl}
          target="_blank"
          rel="noopener noreferrer"
          className="inline-flex items-center gap-1 text-xs text-emerald-500 hover:underline"
        >
          {t('providerSettings.getApiKey')}
          <ExternalLinkIcon className="size-3" />
        </a>
      )}

      <label className="mt-1 inline-flex items-center gap-2 text-xs text-muted-foreground">
        <input
          type="checkbox"
          checked={test}
          onChange={(e) => setTest(e.target.checked)}
        />
        {t('providerSettings.testConnection')}
      </label>

      <div className="mt-1 flex gap-2">
        <Button onClick={handleSave} disabled={busy} size="sm">
          <SaveIcon className="size-4" />
          {t('providerSettings.save')}
        </Button>
        <Button
          onClick={handleClear}
          disabled={busy || !masked.active}
          variant="outline"
          size="sm"
        >
          <Trash2Icon className="size-4" />
          {t('providerSettings.clear')}
        </Button>
      </div>
    </div>
  )
}

// Radix Select forbids empty-string item values, so use non-empty sentinels.
const SELF_WORKSPACE = '__self__'
const OTHER_WORKSPACE = '__other__'

interface ManageWorkspaceSelectorProps {
  value: string | null // null = the admin's own workspace
  onChange: (workspace: string | null) => void
}

/** Admin-only: pick whose workspace the provider settings below apply to. */
function ManageWorkspaceSelector({ value, onChange }: ManageWorkspaceSelectorProps) {
  const { t } = useTranslation()
  const [workspaces, setWorkspaces] = useState<AdminWorkspace[]>([])
  const [manual, setManual] = useState(false)
  const [manualValue, setManualValue] = useState('')

  useEffect(() => {
    let cancelled = false
    listAdminWorkspaces()
      .then((res) => {
        if (!cancelled) setWorkspaces(res.workspaces)
      })
      .catch((err: any) => {
        toast.error(err?.message || t('providerSettings.admin.loadWorkspacesFailed'))
      })
    return () => {
      cancelled = true
    }
  }, [t])

  const selectValue = value === null ? SELF_WORKSPACE : manual ? OTHER_WORKSPACE : value

  const handleSelect = (v: string) => {
    if (v === SELF_WORKSPACE) {
      setManual(false)
      onChange(null)
    } else if (v === OTHER_WORKSPACE) {
      setManual(true)
      onChange(manualValue.trim() || null)
    } else {
      setManual(false)
      onChange(v)
    }
  }

  return (
    <Card>
      <CardContent className="flex flex-col gap-2 pt-6">
        <label className="text-sm font-medium">
          {t('providerSettings.admin.manageWorkspace')}
        </label>
        <Select value={selectValue} onValueChange={handleSelect}>
          <SelectTrigger>
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value={SELF_WORKSPACE}>
              {t('providerSettings.admin.myWorkspace')}
            </SelectItem>
            {workspaces.map((w) => (
              <SelectItem key={w.workspace_id} value={w.workspace_id}>
                {w.owner_username} ({w.workspace_id})
              </SelectItem>
            ))}
            <SelectItem value={OTHER_WORKSPACE}>
              {t('providerSettings.admin.otherWorkspace')}
            </SelectItem>
          </SelectContent>
        </Select>
        {manual && (
          <Input
            value={manualValue}
            onChange={(e) => {
              setManualValue(e.target.value)
              onChange(e.target.value.trim() || null)
            }}
            placeholder={t('providerSettings.admin.otherWorkspacePlaceholder')}
            autoComplete="off"
          />
        )}
        {value && (
          <p className="text-xs text-amber-600 dark:text-amber-400">
            {t('providerSettings.admin.actingOn', { workspace: value })}
          </p>
        )}
      </CardContent>
    </Card>
  )
}

export default function ProviderSettings() {
  const { t } = useTranslation()
  const { role } = useAuthStore()
  const isAdmin = role === 'admin'
  const [targetWorkspace, setTargetWorkspace] = useState<string | null>(null)
  const config = useProvidersStore.use.config()
  const setConfig = useProvidersStore.use.setConfig()
  const [loading, setLoading] = useState(false)
  const [disabled, setDisabled] = useState(false)
  const [effective, setEffective] = useState<EffectiveRolesResponse | null>(null)
  // Bumped after each reload to remount the slot forms with fresh state.
  const [reloadKey, setReloadKey] = useState(0)

  const reload = useCallback(async () => {
    setLoading(true)
    try {
      const data = await getProviderConfig(targetWorkspace)
      setConfig(data)
      setDisabled(false)
      setReloadKey((k) => k + 1)
      // Best-effort ground-truth introspection; ignore failures.
      try {
        setEffective(await getEffectiveRoleConfig(targetWorkspace))
      } catch {
        setEffective(null)
      }
    } catch (err: any) {
      // 503 → feature not enabled on this server; show a notice rather than an error.
      if (typeof err?.message === 'string' && err.message.includes('503')) {
        setDisabled(true)
        setConfig(null)
      } else {
        toast.error(err?.message || t('providerSettings.errors.loadFailed'))
      }
    } finally {
      setLoading(false)
    }
  }, [setConfig, targetWorkspace, t])

  useEffect(() => {
    // eslint-disable-next-line react-hooks/set-state-in-effect
    reload()
  }, [reload])

  return (
    <div className="mx-auto flex max-w-3xl flex-col gap-4 overflow-auto p-6">
      <div>
        <h1 className="text-xl font-bold">{t('providerSettings.title')}</h1>
        <p className="text-sm text-muted-foreground">{t('providerSettings.subtitle')}</p>
      </div>

      {isAdmin && (
        <ManageWorkspaceSelector value={targetWorkspace} onChange={setTargetWorkspace} />
      )}

      {disabled && (
        <Card>
          <CardContent className="pt-6">
            <p className="text-sm text-muted-foreground">
              {t('providerSettings.disabledNotice')}
            </p>
          </CardContent>
        </Card>
      )}

      {config && (
        <div className="grid gap-4 md:grid-cols-2">
          <Card>
            <CardHeader>
              <CardTitle>{t('providerSettings.llmTitle')}</CardTitle>
              <CardDescription>{t('providerSettings.llmDescription')}</CardDescription>
            </CardHeader>
            <CardContent>
              <ProviderSlotForm
                key={`llm-${reloadKey}`}
                slot="llm"
                masked={config.llm}
                targetWorkspace={targetWorkspace}
                onSaved={reload}
              />
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle>{t('providerSettings.visionTitle')}</CardTitle>
              <CardDescription>{t('providerSettings.visionDescription')}</CardDescription>
            </CardHeader>
            <CardContent>
              <ProviderSlotForm
                key={`vision-${reloadKey}`}
                slot="vision"
                masked={config.vision}
                targetWorkspace={targetWorkspace}
                onSaved={reload}
              />
            </CardContent>
          </Card>
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

      {/* Ground-truth: exactly what each LLM role is calling right now. */}
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
                <th className="py-1 font-medium">{t('providerSettings.effective.source')}</th>
              </tr>
            </thead>
            <tbody>
              {Object.entries(effective.roles).map(([role, cfg]) => (
                <tr key={role} className="border-t border-border/40">
                  <td className="py-1 pr-3 font-mono">{role}</td>
                  <td className="py-1 pr-3 font-mono break-all">{cfg.host || '—'}</td>
                  <td className="py-1 pr-3 font-mono break-all">{cfg.model || '—'}</td>
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
    </div>
  )
}
