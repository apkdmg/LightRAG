import { useEffect, useState } from 'react'
import { useTranslation } from 'react-i18next'
import { toast } from 'sonner'
import { Dialog, DialogContent, DialogDescription, DialogHeader, DialogTitle } from '@/components/ui/Dialog'
import Input from '@/components/ui/Input'
import { insightsApi } from './api'

type PriceRow = { model: string; prompt: string; completion: string }
type InvoiceRow = { month: string; amount: string }

/** The dashboard's own numbers. System models and keys stay on the Providers page. */
export default function SettingsDialog({ open, onClose, onSaved, knownModels, backfill }: { open: boolean; onClose: () => void; onSaved: () => void; knownModels: string[]; backfill: any }) {
  const { t } = useTranslation()
  const [prices, setPrices] = useState<PriceRow[]>([])
  const [invoices, setInvoices] = useState<InvoiceRow[]>([])
  const [amount, setAmount] = useState('')
  const [start, setStart] = useState('')
  const [limit, setLimit] = useState('')
  const [rate, setRate] = useState('')
  const [saving, setSaving] = useState(false)

  useEffect(() => {
    if (!open) return
    insightsApi.settings().then((s) => {
      const names = Array.from(new Set([...Object.keys(s.prices || {}), ...knownModels.filter(Boolean)]))
      setPrices(names.map((model) => ({ model, prompt: s.prices?.[model]?.prompt?.toString() ?? '', completion: s.prices?.[model]?.completion?.toString() ?? '' })))
      setInvoices(Object.entries<number>(s.invoices || {}).map(([month, value]) => ({ month, amount: String(value) })))
      setAmount(s.budget?.amount?.toString() ?? '')
      setStart(s.budget?.start ?? '')
      setLimit(s.rate_limit_tpm?.toString() ?? '')
      setRate(s.myr_rate?.toString() ?? '')
    }).catch((e) => toast.error(e?.message))
  }, [open, knownModels])

  const save = async () => {
    setSaving(true)
    try {
      const priceMap = Object.fromEntries(prices.filter((p) => p.model.trim() && (p.prompt !== '' || p.completion !== '')).map((p) => [p.model.trim(), { prompt: Number(p.prompt || 0), completion: Number(p.completion || 0) }]))
      await insightsApi.saveSettings({
        prices: priceMap,
        invoices: Object.fromEntries(invoices.filter((i) => i.month && i.amount !== '').map((i) => [i.month, Number(i.amount)])),
        budget: { amount: amount === '' ? null : Number(amount), start: start || null },
        rate_limit_tpm: limit === '' ? null : Number(limit),
        myr_rate: rate === '' ? null : Number(rate)
      })
      toast.success(t('insights.settings.saved'))
      onSaved()
      onClose()
    } catch (e: any) {
      toast.error(e?.response?.data?.detail?.toString() || e?.message)
    } finally {
      setSaving(false)
    }
  }
  const runBackfill = async () => {
    try {
      await insightsApi.backfill()
      toast.success(t('insights.settings.backfillStarted'))
      onSaved()
    } catch (e: any) {
      toast.error(e?.message)
    }
  }
  const field = 'text-muted-foreground flex flex-col gap-0.5 text-xs'
  return (
    <Dialog open={open} onOpenChange={(v) => !v && onClose()}>
      <DialogContent className="max-h-[85vh] max-w-2xl overflow-y-auto">
        <DialogHeader>
          <DialogTitle>{t('insights.settings.title')}</DialogTitle>
          <DialogDescription>{t('insights.settings.sub')}</DialogDescription>
        </DialogHeader>
        <section className="flex flex-col gap-2">
          <h3 className="m-0 text-sm font-semibold">{t('insights.settings.prices')}</h3>
          <p className="text-muted-foreground m-0 text-xs">{t('insights.settings.pricesSub')}</p>
          {prices.length ? (
            <div className="text-muted-foreground grid gap-2 text-xs" style={{ gridTemplateColumns: 'minmax(0, 2fr) minmax(0, 1fr) minmax(0, 1fr)' }}>
              <span>{t('insights.settings.model')}</span><span>{t('insights.settings.prompt')}</span><span>{t('insights.settings.completion')}</span>
            </div>
          ) : null}
          {prices.map((p, i) => (
            <div key={i} className="grid items-center gap-2" style={{ gridTemplateColumns: 'minmax(0, 2fr) minmax(0, 1fr) minmax(0, 1fr)' }}>
              <Input value={p.model} aria-label={t('insights.settings.model')} placeholder={t('insights.settings.model')} onChange={(e) => setPrices(prices.map((x, j) => (j === i ? { ...x, model: e.target.value } : x)))} className="h-9 font-mono text-xs" />
              <Input type="number" min="0" step="0.01" value={p.prompt} aria-label={t('insights.settings.prompt')} placeholder={t('insights.settings.prompt')} onChange={(e) => setPrices(prices.map((x, j) => (j === i ? { ...x, prompt: e.target.value } : x)))} className="h-9" />
              <Input type="number" min="0" step="0.01" value={p.completion} aria-label={t('insights.settings.completion')} placeholder={t('insights.settings.completion')} onChange={(e) => setPrices(prices.map((x, j) => (j === i ? { ...x, completion: e.target.value } : x)))} className="h-9" />
            </div>
          ))}
          <div><button type="button" className="hover:bg-muted min-h-8 rounded-lg border px-3 text-[13px]" onClick={() => setPrices([...prices, { model: '', prompt: '', completion: '' }])}>{t('insights.settings.addModel')}</button></div>
        </section>
        <section className="flex flex-col gap-2">
          <h3 className="m-0 text-sm font-semibold">{t('insights.settings.budget')}</h3>
          <p className="text-muted-foreground m-0 text-xs">{t('insights.settings.budgetSub')}</p>
          <div className="grid gap-2" style={{ gridTemplateColumns: 'repeat(auto-fit, minmax(150px, 1fr))' }}>
            <label className={field}>{t('insights.settings.amount')}<Input type="number" min="0" value={amount} onChange={(e) => setAmount(e.target.value)} className="h-9" /></label>
            <label className={field}>{t('insights.settings.start')}<Input type="date" value={start} onChange={(e) => setStart(e.target.value)} className="h-9" /></label>
            <label className={field}>{t('insights.settings.rate')}<Input type="number" min="0" step="0.01" value={rate} onChange={(e) => setRate(e.target.value)} className="h-9" /></label>
            <label className={field}>{t('insights.settings.limit')}<Input type="number" min="0" value={limit} onChange={(e) => setLimit(e.target.value)} className="h-9" /></label>
          </div>
        </section>
        <section className="flex flex-col gap-2">
          <h3 className="m-0 text-sm font-semibold">{t('insights.settings.invoices')}</h3>
          <p className="text-muted-foreground m-0 text-xs">{t('insights.settings.invoicesSub')}</p>
          {invoices.map((inv, i) => (
            <div key={i} className="grid items-center gap-2" style={{ gridTemplateColumns: '1fr 1fr' }}>
              <Input type="month" value={inv.month} aria-label={t('insights.settings.month')} onChange={(e) => setInvoices(invoices.map((x, j) => (j === i ? { ...x, month: e.target.value } : x)))} className="h-9" />
              <Input type="number" min="0" step="0.01" value={inv.amount} aria-label={t('insights.settings.invoiceAmount')} onChange={(e) => setInvoices(invoices.map((x, j) => (j === i ? { ...x, amount: e.target.value } : x)))} className="h-9" />
            </div>
          ))}
          <div><button type="button" className="hover:bg-muted min-h-8 rounded-lg border px-3 text-[13px]" onClick={() => setInvoices([...invoices, { month: '', amount: '' }])}>{t('insights.settings.addInvoice')}</button></div>
        </section>
        <section className="flex flex-col gap-2">
          <h3 className="m-0 text-sm font-semibold">{t('insights.settings.backfill')}</h3>
          <p className="text-muted-foreground m-0 text-xs">{t('insights.settings.backfillSub')}</p>
          <div>
            <button type="button" disabled={backfill?.status === 'running' || backfill?.status === 'done'} className="hover:bg-muted min-h-8 rounded-lg border px-3 text-[13px] disabled:opacity-50" onClick={runBackfill}>
              {backfill?.status === 'done' ? t('insights.settings.backfillDone', { rows: backfill.rows }) : backfill?.status === 'running' ? t('insights.settings.backfillRunning') : t('insights.settings.backfillRun')}
            </button>
          </div>
        </section>
        <div className="flex justify-end gap-2">
          <button type="button" className="hover:bg-muted min-h-9 rounded-lg border px-3 text-[13px]" onClick={onClose}>{t('insights.settings.cancel')}</button>
          <button type="button" disabled={saving} className="bg-primary text-primary-foreground min-h-9 rounded-lg px-3 text-[13px] hover:opacity-90 disabled:opacity-50" onClick={save}>{t('insights.settings.save')}</button>
        </div>
      </DialogContent>
    </Dialog>
  )
}
