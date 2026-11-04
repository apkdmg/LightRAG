import type { TFunction } from 'i18next'

/** A per-call row's fields these cells read (GET /admin/insights/calls). */
type CallRow = { status?: string | null; cache_hit?: boolean | null; estimated?: boolean | null; error_kind?: string | null }

/**
 * "Error · rate limited (429)": the call's status, and for a failed call the
 * error kind the recorder classified (classify.py, the same kinds the
 * Operations console shows). Calls recorded before the column existed have
 * no kind and read as before.
 */
export function callStatusText(t: TFunction, r: CallRow): string {
  const status = r.cache_hit ? t('insights.explore.cacheHit') : t(`insights.explore.status.${r.status}`, { defaultValue: r.status ?? '–' })
  const kind = r.error_kind ? t(`operations.errorKind.${r.error_kind}`, { defaultValue: r.error_kind.replace(/_/g, ' ') }) : null
  return [status, kind, r.estimated ? t('insights.explore.estimated') : null].filter(Boolean).join(' · ')
}
