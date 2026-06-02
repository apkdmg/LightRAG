/**
 * Provider presets for the per-workspace BYO LLM / Vision-LLM settings panel.
 *
 * Frontend-only convenience: a preset pre-fills the OpenAI-compatible `base_url`
 * and offers suggested model names so a non-technical owner only needs to paste
 * their API key. The backend stays fully provider-agnostic — it just stores the
 * resulting `{ base_url, api_key, model }`. The `preset_id` is round-tripped as
 * an opaque UI hint so the panel re-selects the right preset on reload.
 *
 * Suggested model lists are editable defaults (the form allows free-text), so
 * they remain usable even if a provider renames a model.
 */

export interface ProviderPreset {
  /** Stable identifier stored as `preset_id`. */
  id: string
  /** Display name. */
  label: string
  /** OpenAI-compatible base URL. Empty for the "custom" preset. */
  baseUrl: string
  /** Whether the user may edit the base URL (true only for "custom"). */
  editableBaseUrl: boolean
  /** Suggested text-LLM models. */
  suggestedModels: string[]
  /** Suggested vision-capable models (subset / multimodal). */
  suggestedVisionModels: string[]
  /** Where to obtain an API key (shown as a help link). */
  apiKeyHelpUrl?: string
}

export const CUSTOM_PRESET_ID = 'custom'

export const PROVIDER_PRESETS: ProviderPreset[] = [
  {
    id: 'gemini',
    label: 'Google Gemini',
    baseUrl: 'https://generativelanguage.googleapis.com/v1beta/openai/',
    editableBaseUrl: false,
    suggestedModels: ['gemini-2.5-flash', 'gemini-2.5-pro', 'gemini-2.0-flash'],
    suggestedVisionModels: ['gemini-2.5-flash', 'gemini-2.5-pro'],
    apiKeyHelpUrl: 'https://aistudio.google.com/app/apikey'
  },
  {
    id: 'openrouter',
    label: 'OpenRouter',
    baseUrl: 'https://openrouter.ai/api/v1',
    editableBaseUrl: false,
    suggestedModels: [
      'openai/gpt-4o-mini',
      'meta-llama/llama-3.3-70b-instruct',
      'google/gemini-2.5-flash'
    ],
    suggestedVisionModels: ['openai/gpt-4o-mini', 'google/gemini-2.5-flash'],
    apiKeyHelpUrl: 'https://openrouter.ai/keys'
  },
  {
    id: 'requesty',
    label: 'Requesty.ai',
    baseUrl: 'https://router.requesty.ai/v1',
    editableBaseUrl: false,
    suggestedModels: [
      'openai/gpt-4o-mini',
      'google/gemini-2.5-flash',
      'anthropic/claude-sonnet-4-5'
    ],
    suggestedVisionModels: ['openai/gpt-4o-mini', 'google/gemini-2.5-flash'],
    apiKeyHelpUrl: 'https://app.requesty.ai/api-keys'
  },
  {
    id: CUSTOM_PRESET_ID,
    label: 'Custom (OpenAI-compatible)',
    baseUrl: '',
    editableBaseUrl: true,
    suggestedModels: [],
    suggestedVisionModels: []
  }
]

export const getPreset = (id: string | null | undefined): ProviderPreset =>
  PROVIDER_PRESETS.find((p) => p.id === id) ??
  PROVIDER_PRESETS[PROVIDER_PRESETS.length - 1] // default to "custom"

/** Pick a preset from a stored base URL when no preset_id was recorded. */
export const presetForBaseUrl = (baseUrl: string | null | undefined): ProviderPreset => {
  if (baseUrl) {
    const match = PROVIDER_PRESETS.find((p) => p.baseUrl && p.baseUrl === baseUrl)
    if (match) return match
  }
  return getPreset(CUSTOM_PRESET_ID)
}
