import { useState } from 'react'
import { useTranslation } from 'react-i18next'
import { insightsApi } from './api'
import { WorkCharts } from './charts'
import { useData, useInsights } from './context'
import { INGEST, Purpose, QUESTIONS } from './meta'
import { Card, Empty, Mix, Seg, Title, fmtCount, fmtMoney, fmtPct, fmtTokens } from './ui'

export default function WorkTab() {
  const { t } = useTranslation()
  const { scope, params } = useInsights()
  const [side, setSide] = useState<'ingest' | 'questions'>(scope.purpose && QUESTIONS.includes(scope.purpose) ? 'questions' : 'ingest')
  const purpose = scope.purpose && (side === 'ingest' ? INGEST : QUESTIONS).includes(scope.purpose) ? scope.purpose : ''
  const { data, error, loading } = useData(
    () => insightsApi.work({ ...params, side, workspace: scope.workspace, purpose }),
    [JSON.stringify(params), side, scope.workspace, purpose]
  )
  if (error && !data) return <Empty>{error}</Empty>
  if (!data) return <Empty>{t('insights.loading')}</Empty>
  const images = purpose === 'vlm'
  const unit = side === 'ingest' ? (images ? t('insights.work.images') : t('insights.work.documents')) : t('insights.work.questions')
  const per = side === 'ingest' ? (images ? t('insights.work.perImage') : t('insights.work.perDocument')) : t('insights.work.perQuestion')
  return (
    <div className={`flex flex-col gap-4 ${loading ? 'opacity-60 transition-opacity' : 'transition-opacity'}`}>
      <Card>
        <div className="mb-3 flex flex-wrap items-start justify-between gap-2.5">
          <Title sub={t(`insights.work.${side}Sub`)}>{t(`insights.work.${side}Title`)}</Title>
          <Seg value={side} onChange={setSide} options={[{ value: 'ingest', label: t('insights.group.ingest.title') }, { value: 'questions', label: t('insights.group.questions.title') }]} />
        </div>
        <WorkCharts days={data.days} purposes={data.purposes as Purpose[]} unitLabel={unit} perLabel={per} />
      </Card>
      <Card>
        <Title sub={t(`insights.work.${side}TableSub`)}>{t('insights.work.bySource')}</Title>
        <div className="overflow-x-auto">
          <table className="w-full min-w-[640px] text-[13px]">
            <thead><tr className="text-muted-foreground border-b text-xs">
              <th className="p-2 text-left font-medium">{t('insights.explore.dim.source')}</th>
              <th className="w-[18%] p-2 text-left font-medium">{t('insights.workspaces.mix')}</th>
              <th className="p-2 text-right font-medium">{t('insights.tokens')}</th>
              <th className="p-2 text-right font-medium">{unit}</th>
              <th className="p-2 text-right font-medium">{per}</th>
              <th className="p-2 text-right font-medium">{t('insights.estCost')}</th>
              <th className="p-2 text-right font-medium">{t('insights.card.cache')}</th>
              <th className="p-2 text-right font-medium">{t('insights.work.wasted')}</th>
            </tr></thead>
            <tbody>
              {data.sources.map((s: any) => (
                <tr key={s.id} className="border-b last:border-0">
                  <td className="p-2">{t(`insights.source.${s.id}`, { defaultValue: s.id })}</td>
                  <td className="p-2"><Mix values={s.tokens} /></td>
                  <td className="p-2 text-right">{fmtTokens(s.total)}</td>
                  <td className="p-2 text-right">{fmtCount(s.units)}</td>
                  <td className="p-2 text-right">{fmtTokens(s.per_unit)}</td>
                  <td className="p-2 text-right">{fmtMoney(s.cost)}</td>
                  <td className="p-2 text-right">{fmtPct(s.cache_rate)}</td>
                  <td className="p-2 text-right">{fmtPct(s.wasted_share)}</td>
                </tr>
              ))}
            </tbody>
          </table>
          {!data.sources.length ? <Empty>{t('insights.empty')}</Empty> : null}
        </div>
      </Card>
    </div>
  )
}
