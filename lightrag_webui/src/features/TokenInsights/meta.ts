export const PURPOSES = ['extract', 'vlm', 'keyword', 'query'] as const
export type Purpose = (typeof PURPOSES)[number]

export const PURPOSE_COLOR: Record<Purpose, string> = {
  extract: 'var(--ti-extract)',
  vlm: 'var(--ti-vlm)',
  keyword: 'var(--ti-keyword)',
  query: 'var(--ti-query)'
}

export const INGEST: Purpose[] = ['extract', 'vlm']
export const QUESTIONS: Purpose[] = ['keyword', 'query']

/** Settings that affect each purpose; the page names them and links to Providers. */
export const PURPOSE_SETTINGS: Record<Purpose, string[]> = {
  extract: ['EXTRACT_LLM_MODEL', 'MAX_GLEANING', 'CHUNK_SIZE', 'FORCE_LLM_SUMMARY_ON_MERGE', 'ENABLE_LLM_CACHE_FOR_EXTRACT'],
  vlm: ['VLM_LLM_MODEL', 'VLM_PROCESS_ENABLE', 'VLM_MIN_IMAGE_PIXEL', 'VLM_MAX_IMAGE_BYTES'],
  keyword: ['KEYWORD_LLM_MODEL', 'ENABLE_LLM_CACHE'],
  query: ['QUERY_LLM_MODEL', 'MAX_TOTAL_TOKENS', 'TOP_K', 'CHUNK_TOP_K', 'ENABLE_LLM_CACHE']
}


export const SEVERITY = {
  critical: { color: '#d03b3b', path: 'M8 2.5h8L21.5 8v8L16 21.5H8L2.5 16V8L8 2.5ZM12 8v5M12 16.5v.01' },
  serious: { color: '#ec835a', path: 'M12 3 2.5 20h19L12 3ZM12 10v4M12 17v.01' },
  warning: { color: '#b98100', path: 'M12 3a9 9 0 1 0 0 18 9 9 0 0 0 0-18ZM12 8v5M12 16v.01' }
} as const

export const ICON_CHECK = 'M12 3a9 9 0 1 0 0 18 9 9 0 0 0 0-18ZM8 12l3 3 5-6'
export const ICON_CLOCK = 'M12 3a9 9 0 1 0 0 18 9 9 0 0 0 0-18ZM12 7v5l3 2'
