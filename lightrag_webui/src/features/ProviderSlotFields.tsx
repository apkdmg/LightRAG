import { useCallback, type Dispatch, type SetStateAction } from 'react'
import { useTranslation } from 'react-i18next'
import { ExternalLinkIcon } from 'lucide-react'

import Input from '@/components/ui/Input'
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue
} from '@/components/ui/Select'
import type { ProviderSlotKey } from '@/api/lightrag'
import { PROVIDER_PRESETS, getPreset, type ProviderPreset } from '@/features/providerPresets'

// 'default' = let the provider decide (stored as empty/None server-side).
const REASONING_OPTIONS = ['default', 'none', 'low', 'medium', 'high'] as const

export interface SlotDraft {
  presetId: string
  baseUrl: string
  model: string
  apiKey: string
  reasoning: string // '' = default, else none|low|medium|high
}

interface ProviderSlotFieldsProps {
  slot: ProviderSlotKey
  draft: SlotDraft
  setDraft: Dispatch<SetStateAction<SlotDraft>>
  apiKeyPlaceholder: string
}

/** The provider form fields shared by workspace providers and the server default. */
export default function ProviderSlotFields({
  slot,
  draft,
  setDraft,
  apiKeyPlaceholder
}: ProviderSlotFieldsProps) {
  const { t } = useTranslation()
  const preset: ProviderPreset = getPreset(draft.presetId)
  const suggestedModels = slot === 'vision' ? preset.suggestedVisionModels : preset.suggestedModels
  const datalistId = `provider-models-${slot}`

  const onPresetChange = useCallback(
    (id: string) => {
      setDraft((d) => {
        const p = getPreset(id)
        return { ...d, presetId: id, baseUrl: p.editableBaseUrl ? d.baseUrl : p.baseUrl }
      })
    },
    [setDraft]
  )

  return (
    <>
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

      <label className="text-sm font-medium">{t('providerSettings.baseUrl')}</label>
      <Input
        value={draft.baseUrl}
        onChange={(e) => setDraft((d) => ({ ...d, baseUrl: e.target.value }))}
        placeholder="https://api.example.com/v1"
        readOnly={!preset.editableBaseUrl}
        className={!preset.editableBaseUrl ? 'opacity-70' : ''}
        autoComplete="off"
      />

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

      <label className="text-sm font-medium">{t('providerSettings.reasoning.label')}</label>
      <Select
        value={draft.reasoning || 'default'}
        onValueChange={(v) => setDraft((d) => ({ ...d, reasoning: v === 'default' ? '' : v }))}
      >
        <SelectTrigger>
          <SelectValue />
        </SelectTrigger>
        <SelectContent>
          {REASONING_OPTIONS.map((opt) => (
            <SelectItem key={opt} value={opt}>
              {t(`providerSettings.reasoning.options.${opt}`)}
            </SelectItem>
          ))}
        </SelectContent>
      </Select>
      <p className="text-xs text-muted-foreground">{t('providerSettings.reasoning.help')}</p>

      <label className="text-sm font-medium">{t('providerSettings.apiKey')}</label>
      <Input
        type="password"
        value={draft.apiKey}
        onChange={(e) => setDraft((d) => ({ ...d, apiKey: e.target.value }))}
        placeholder={apiKeyPlaceholder}
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
    </>
  )
}
