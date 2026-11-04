import { useEffect, useState } from 'react'
import { useTranslation } from 'react-i18next'
import { downloadFile, insightsApi } from './api'
import { ForecastChart } from './charts'
import { useData, useInsights } from './context'
import { PURPOSES } from './meta'
import { Card, Empty, Title } from '@/components/admin'
import { PurposeName, fmtChange, fmtDay, fmtMoney, fmtPct, fmtTokens } from './ui'

function Slider({ label, hint, value, min, max, step, shown, onChange }: { label: string; hint: string; value: number; min: number; max: number; step: number; shown: string; onChange: (v: number) => void }) {
  return (
    <label className="flex flex-col gap-1 text-[13px]">
      <span className="flex justify-between gap-2"><span>{label}</span><strong>{shown}</strong></span>
      <span className="text-muted-foreground text-xs">{hint}</span>
      <input type="range" min={min} max={max} step={step} value={value} onChange={(e) => onChange(Number(e.target.value))} className="min-h-7 w-full accent-[var(--ti-extract)]" />
    </label>
  )
}

export default function ForecastTab({ onOpenSettings }: { onOpenSettings: () => void }) {
  const { t } = useTranslation()
  const { scope } = useInsights()
  const [documents, setDocuments] = useState(0)
  const [questions, setQuestions] = useState(0)
  const [bulk, setBulk] = useState(0)
  const [swap, setSwap] = useState('')
  const [what, setWhat] = useState({ documents: 0, questions: 0, bulk: 0 })
  // Sliders move freely; the numbers follow once they settle.
  useEffect(() => {
    const timer = setTimeout(() => setWhat({ documents, questions, bulk }), 200)
    return () => clearTimeout(timer)
  }, [documents, questions, bulk])
  const { data: fc, error } = useData(
    () => insightsApi.forecast({ ...what, swap, purpose: scope.purpose }),
    [what.documents, what.questions, what.bulk, swap, scope.purpose]
  )
  if (error && !fc) return <Empty>{error}</Empty>
  if (!fc) return <Empty>{t('insights.loading')}</Empty>
  if (!fc.ready) {
    return (
      <Card className="flex max-w-[640px] flex-col gap-2.5">
        <Title sub={t('insights.forecast.learningSub')}>{t('insights.forecast.learning')}</Title>
        <div className="text-xs font-semibold">{t('insights.forecast.daysLeft', { count: fc.learning.forecast_in_days })}</div>
      </Card>
    )
  }
  const budget = fc.budget
  const name = scope.purpose ? t(`insights.purpose.${scope.purpose}`) : ''
  const tiles = [
    { label: t('insights.forecast.next30'), value: fmtTokens(fc.next30.tokens), note: t('insights.forecast.tileNote', { low: fmtTokens(fc.next30.low), high: fmtTokens(fc.next30.high), cost: fmtMoney(fc.next30.cost), byo: fmtMoney(fc.next30.byo_cost) }) },
    { label: t('insights.forecast.next90'), value: fmtTokens(fc.next90.tokens), note: t('insights.forecast.tileNote', { low: fmtTokens(fc.next90.low), high: fmtTokens(fc.next90.high), cost: fmtMoney(fc.next90.cost), byo: fmtMoney(fc.next90.byo_cost) }) },
    {
      label: t('insights.strip.budget'),
      value: budget?.set ? (budget.runs_out ? fmtDay(budget.runs_out) : t('insights.strip.later')) : t('insights.strip.noBudget'),
      note: budget?.set ? t('insights.forecast.budgetNote', { left: fmtMoney(budget.left), without: budget.runs_out_without_byo ? fmtDay(budget.runs_out_without_byo) : t('insights.strip.later') }) : ''
    },
    {
      label: t('insights.forecast.purchase'),
      value: budget?.set ? fmtMoney(budget.suggested_purchase) : '–',
      note: budget?.set ? t('insights.forecast.purchaseNote', { pct: fmtPct(fc.next90.cost ? fc.next90.byo_cost / fc.next90.cost : 0) }) : ''
    }
  ]
  return (
    <div className="flex flex-col gap-4">
      {scope.workspace ? <p className="text-muted-foreground m-0 text-[13px]">{t('insights.forecast.allWorkspaces')}</p> : null}
      {fc.learning.forecast_low_confidence ? <p className="text-muted-foreground m-0 text-[13px]">{t('insights.forecast.lowConfidence')}</p> : null}
      <div className="grid gap-3" style={{ gridTemplateColumns: 'repeat(auto-fit, minmax(min(210px, 100%), 1fr))' }}>
        {tiles.map((k) => (
          <Card key={k.label} className="p-3.5">
            <div className="text-muted-foreground text-[13px]">{k.label}</div>
            <div className="text-2xl leading-snug font-semibold tracking-tight">{k.value}</div>
            <div className="text-muted-foreground text-xs">{k.note}</div>
          </Card>
        ))}
      </div>
      {!budget?.set ? (
        <Card className="flex flex-wrap items-center justify-between gap-2.5">
          <span className="text-[13px]">{t('insights.forecast.setBudget')}</span>
          <button type="button" className="hover:bg-muted min-h-9 rounded-lg border px-3 text-[13px]" onClick={onOpenSettings}>{t('insights.settings.open')}</button>
        </Card>
      ) : null}
      <div className="grid items-start gap-4" style={{ gridTemplateColumns: 'repeat(auto-fit, minmax(min(420px, 100%), 1fr))' }}>
        <Card>
          <Title sub={t('insights.forecast.chartSub')}>{name ? t('insights.forecast.chartTitlePurpose', { name }) : t('insights.forecast.chartTitle')}</Title>
          <div className="mb-2.5 flex flex-wrap gap-x-4 gap-y-1 text-xs">
            <span className="inline-flex items-center gap-1.5"><svg width="22" height="6" aria-hidden="true"><path d="M0 3H22" stroke="var(--foreground)" strokeWidth="2" /></svg>{t('insights.forecast.allUsage')}</span>
            <span className="inline-flex items-center gap-1.5"><svg width="22" height="6" aria-hidden="true"><path d="M0 3H22" stroke="var(--muted-foreground)" strokeWidth="2" strokeDasharray="2 4" /></svg>{t('insights.forecast.systemOnly')}</span>
            <span className="inline-flex items-center gap-1.5"><svg width="22" height="6" aria-hidden="true"><path d="M0 3H22" stroke="var(--foreground)" strokeWidth="2" strokeDasharray="6 5" /></svg>{t('insights.forecast.projected')}</span>
            <span className="inline-flex items-center gap-1.5"><span className="inline-block h-2.5 w-[22px] rounded-sm" style={{ background: 'var(--muted-foreground)', opacity: 0.3 }} />{t('insights.forecast.range')}</span>
          </div>
          <ForecastChart forecast={fc} />
        </Card>
        <Card className="flex flex-col gap-4">
          <Title sub={t('insights.forecast.whatIfSub')}>{t('insights.forecast.whatIf')}</Title>
          <Slider label={t('insights.forecast.documents')} hint={t('insights.forecast.documentsHint')} min={-30} max={100} step={5} value={documents} shown={`${documents > 0 ? '+' : ''}${documents}%`} onChange={setDocuments} />
          <Slider label={t('insights.forecast.bulk')} hint={t('insights.forecast.bulkHint')} min={0} max={20000} step={500} value={bulk} shown={t('insights.forecast.bulkValue', { count: bulk })} onChange={setBulk} />
          <Slider label={t('insights.forecast.questions')} hint={t('insights.forecast.questionsHint')} min={-30} max={100} step={5} value={questions} shown={`${questions > 0 ? '+' : ''}${questions}%`} onChange={setQuestions} />
          <label className="flex flex-col gap-1 text-[13px]">{t('insights.forecast.swap')}
            <select value={swap} onChange={(e) => setSwap(e.target.value)} className="bg-card min-h-9 rounded-lg border px-2 text-sm">
              <option value="">{t('insights.forecast.noChange')}</option>
              {PURPOSES.map((p) => <option key={p} value={p}>{t(`insights.purpose.${p}`)}</option>)}
            </select>
          </label>
          <div className="flex flex-wrap gap-2">
            <button type="button" className="bg-primary text-primary-foreground min-h-9 rounded-lg px-3 text-[13px] hover:opacity-90" onClick={() => downloadFile('/brief.pdf', 'procurement-brief.pdf')}>{t('insights.forecast.briefPdf')}</button>
            <button type="button" className="hover:bg-muted min-h-9 rounded-lg border px-3 text-[13px]" onClick={() => downloadFile('/brief.csv', 'procurement-brief.csv')}>{t('insights.forecast.briefCsv')}</button>
          </div>
          <p className="text-muted-foreground m-0 text-xs">{t('insights.forecast.byoNote')}</p>
        </Card>
      </div>
      <Card>
        <Title sub={t('insights.forecast.tableSub')}>{t('insights.forecast.tableTitle')}</Title>
        <div className="overflow-x-auto">
          <table className="w-full min-w-[600px] text-[13px]">
            <thead><tr className="text-muted-foreground border-b text-xs">
              {['purpose', 'grows', 'tokens', 'range', 'cost', 'byo', 'share'].map((c, i) => <th key={c} className={`p-2 font-medium ${i < 2 ? 'text-left' : 'text-right'}`}>{t(`insights.forecast.col.${c}`)}</th>)}
            </tr></thead>
            <tbody>
              {fc.rows.map((r: any) => (
                <tr key={r.id} className={`border-b last:border-0 ${scope.purpose === r.id ? 'bg-muted' : ''}`}>
                  <td className="p-2"><PurposeName id={r.id} /></td>
                  <td className="p-2">{t(`insights.forecast.grows.${r.grows_with}`)}</td>
                  <td className="p-2 text-right">{fmtTokens(r.tokens)}</td>
                  <td className="p-2 text-right whitespace-nowrap">{fmtTokens(r.low)} – {fmtTokens(r.high)}</td>
                  <td className="p-2 text-right">{fmtMoney(r.cost)}</td>
                  <td className="p-2 text-right">{fmtMoney(r.cost * r.byo_share)} · {fmtPct(r.byo_share)}</td>
                  <td className="p-2 text-right">{fmtPct(r.share_cost)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>
      <div className="grid items-start gap-4" style={{ gridTemplateColumns: 'repeat(auto-fit, minmax(min(360px, 100%), 1fr))' }}>
        <Card>
          <Title sub={t('insights.forecast.unitsSub')}>{t('insights.forecast.units')}</Title>
          {(['document', 'image', 'question'] as const).map((u) => (
            <div key={u} className="flex justify-between gap-3 border-b py-1.5 text-[13px] last:border-0">
              <span>{t(`insights.forecast.unit.${u}`)} <span className="text-muted-foreground text-xs">{t(`insights.forecast.unitNote.${u}`)}</span></span>
              <strong>{fc.units[u] ? `${fmtTokens(fc.units[u].tokens)} · ${fmtMoney(fc.units[u].cost)}` : '–'}</strong>
            </div>
          ))}
        </Card>
        <Card>
          <Title sub={t('insights.forecast.billSub')}>{t('insights.forecast.bill')}</Title>
          {fc.bill.length ? (
            <table className="w-full text-[13px]">
              <thead><tr className="text-muted-foreground border-b text-xs"><th className="p-1.5 text-left font-medium">{t('insights.forecast.billCol.month')}</th><th className="p-1.5 text-right font-medium">{t('insights.forecast.billCol.system')}</th><th className="p-1.5 text-right font-medium">{t('insights.forecast.billCol.invoice')}</th><th className="p-1.5 text-right font-medium">{t('insights.forecast.billCol.difference')}</th></tr></thead>
              <tbody>
                {fc.bill.map((b: any) => (
                  <tr key={b.month} className="border-b last:border-0"><td className="p-1.5">{b.month}</td><td className="p-1.5 text-right">{fmtMoney(b.system_cost)}</td><td className="p-1.5 text-right">{fmtMoney(b.invoice)}</td><td className="p-1.5 text-right">{fmtChange(b.difference)}</td></tr>
                ))}
              </tbody>
            </table>
          ) : (
            <div className="flex flex-wrap items-center justify-between gap-2.5"><span className="text-muted-foreground text-[13px]">{t('insights.forecast.noInvoices')}</span><button type="button" className="hover:bg-muted min-h-9 rounded-lg border px-3 text-[13px]" onClick={onOpenSettings}>{t('insights.settings.open')}</button></div>
          )}
        </Card>
      </div>
    </div>
  )
}
