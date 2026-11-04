import { useTranslation } from 'react-i18next'
import { OpsConfig } from '../api'
import { Panel } from '../components/Panel'
import { fmtClock } from '../format'
import { OpsPoll } from '../poll'
import { groupConfig } from './model'

/** A config value as text: objects one level deep as "key=value" pairs, never as raw JSON blobs. */
function valueText(v: unknown): string {
  if (v === null || v === undefined || v === '') return '–'
  if (typeof v === 'boolean') return v ? 'true' : 'false'
  if (typeof v === 'number' || typeof v === 'string') return String(v)
  if (Array.isArray(v)) return v.map(valueText).join(', ') || '–'
  if (typeof v === 'object') {
    const entries = Object.entries(v as Record<string, unknown>).filter(([, x]) => x !== null && x !== undefined && x !== '')
    return entries.length ? entries.map(([k, x]) => `${k}=${typeof x === 'object' ? JSON.stringify(x) : String(x)}`).join(', ') : '–'
  }
  return String(v)
}

/** Nested maps (per-role LLM config, storage workspaces) get one row per entry. */
function Rows({ name, value }: { name: string; value: unknown }) {
  if (value && typeof value === 'object' && !Array.isArray(value)) {
    const entries = Object.entries(value as Record<string, unknown>)
    if (entries.length && entries.every(([, v]) => v && typeof v === 'object' && !Array.isArray(v))) {
      return (
        <>
          {entries.map(([k, v]) => (
            <Row key={k} name={`${name}.${k}`} value={valueText(v)} />
          ))}
        </>
      )
    }
  }
  return <Row name={name} value={valueText(value)} />
}

function Row({ name, value }: { name: string; value: string }) {
  return (
    <div className="grid grid-cols-1 gap-x-4 border-b py-1.5 last:border-0 sm:grid-cols-[minmax(180px,260px)_minmax(0,1fr)]">
      <dt className="text-muted-foreground font-mono text-xs break-all">{name}</dt>
      <dd className="m-0 font-mono text-xs break-all">{value}</dd>
    </div>
  )
}

/**
 * Health › Config (FR-34): the server's configuration in the shape the status
 * dialog uses, with URLs stripped of userinfo and secrets redacted on the
 * server. Read-only; it changes only on restart, so it is fetched once.
 */
export default function ConfigView({ poll }: { poll: OpsPoll<OpsConfig> }) {
  const { t } = useTranslation()
  return (
    <Panel
      poll={poll}
      freshness={3600}
      title={t('operations.config.title')}
      sub={poll.updatedAt ? t('operations.config.sub', { time: fmtClock(poll.updatedAt) }) : undefined}
      actions={
        <button type="button" onClick={poll.reload} className="hover:bg-muted min-h-8 rounded-lg border px-2.5 text-[13px]">
          {t('operations.refresh')}
        </button>
      }
    >
      {(data) => {
        const groups = groupConfig(data.configuration)
        return (
          <div className="flex flex-col gap-4">
            {data.core_version || data.api_version ? (
              <p className="m-0 text-[13px]">{t('operations.config.versions', { core: data.core_version ?? '–', api: data.api_version ?? '–' })}</p>
            ) : null}
            {groups.length ? (
              groups.map((g) => (
                <section key={g.id} className="flex flex-col gap-1">
                  <h3 className="m-0 text-[13px] font-semibold">{t(`operations.config.group.${g.id}`)}</h3>
                  <dl className="m-0">
                    {g.entries.map(([k, v]) => (
                      <Rows key={k} name={k} value={v} />
                    ))}
                  </dl>
                </section>
              ))
            ) : (
              <p className="text-muted-foreground m-0 text-sm">{t('operations.config.empty')}</p>
            )}
            <p className="text-muted-foreground m-0 text-xs">{t('operations.config.note')}</p>
          </div>
        )
      }}
    </Panel>
  )
}
