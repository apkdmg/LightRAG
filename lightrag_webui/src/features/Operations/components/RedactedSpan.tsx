import { useTranslation } from 'react-i18next'
import { cn } from '@/lib/utils'
import { splitRedacted } from '../logs'

/**
 * One value the server masked when the line was written: `⟨email⟩`,
 * `⟨ic⟩`, `⟨secret⟩`… or a masked file name (`Sen•••er.pdf`). It cannot be
 * revealed; the tooltip says so (§2.5).
 */
export function RedactedSpan({ token, text }: { token?: string; text?: string }) {
  const { t } = useTranslation()
  const name = token ? t(`operations.logs.token.${token}`, { defaultValue: token }) : null
  return (
    <span
      className={cn('bg-muted text-muted-foreground rounded px-1 font-mono text-[0.92em] whitespace-nowrap', !token && 'border border-dashed')}
      title={t('operations.logs.maskedHint')}
    >
      {name ? `⟨${name}⟩` : text}
    </span>
  )
}

/** Redacted text with each masked span marked; plain text otherwise (React escapes it). */
export function RedactedText({ text, className }: { text: string | null | undefined; className?: string }) {
  const parts = splitRedacted(text)
  if (!parts.length) return null
  return (
    <span className={className}>
      {parts.map((p, i) =>
        p.kind === 'text' ? <span key={i}>{p.text}</span> : p.kind === 'token' ? <RedactedSpan key={i} token={p.token} /> : <RedactedSpan key={i} text={p.text} />
      )}
    </span>
  )
}
