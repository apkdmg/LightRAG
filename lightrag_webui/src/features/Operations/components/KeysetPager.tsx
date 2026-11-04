import { useTranslation } from 'react-i18next'
import { Cursor, FIRST_PAGE, newerCursor, olderCursor } from '../keyset'

/**
 * "Older" pages back on the server's `(ts, id)` keyset; "Newer" and "Newest"
 * walk back up (the audit log, alert history).
 */
export default function KeysetPager({ next, cursor, onCursor }: { next: string | null; cursor: Cursor; onCursor: (c: Cursor) => void }) {
  const { t } = useTranslation()
  if (!next && !cursor.before) return null
  return (
    <div className="flex flex-wrap items-center justify-end gap-2 text-xs">
      {cursor.before ? (
        <>
          <button type="button" className="hover:bg-muted min-h-8 rounded-lg border px-3 text-[13px]" onClick={() => onCursor(FIRST_PAGE)}>
            {t('operations.audit.newest')}
          </button>
          <button type="button" className="hover:bg-muted min-h-8 rounded-lg border px-3 text-[13px]" onClick={() => onCursor(newerCursor(cursor))}>
            {t('operations.audit.newer')}
          </button>
        </>
      ) : null}
      <button
        type="button"
        className="hover:bg-muted min-h-8 rounded-lg border px-3 text-[13px] disabled:opacity-50"
        disabled={!next}
        onClick={() => next && onCursor(olderCursor(cursor, next))}
      >
        {t('operations.audit.older')}
      </button>
    </div>
  )
}
