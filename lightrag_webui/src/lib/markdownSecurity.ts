import rehypeSanitize, { defaultSchema } from 'rehype-sanitize'
import type { Options as SanitizeSchema } from 'rehype-sanitize'

/**
 * Sanitization schema for rendered chat content.
 *
 * LLM answers echo retrieved document text, so any HTML they contain is
 * attacker-influenced (indirect prompt injection). Raw HTML is still parsed
 * (rehype-raw) so harmless formatting keeps working, but everything is then
 * filtered through GitHub's allowlist: no scripts, iframes, forms, styles,
 * event handlers or javascript: URLs.
 *
 * Additions to the default schema, all inert:
 * - mark / u tags used for highlighting in answers
 * - math-inline / math-display classes emitted by remark-math, which
 *   rehype-katex (run after sanitization) uses to pick inline vs display mode
 * - footnote-ref class emitted by our remarkFootnotes plugin
 */
export const markdownSanitizeSchema: SanitizeSchema = {
  ...defaultSchema,
  tagNames: [...(defaultSchema.tagNames ?? []), 'mark', 'u'],
  attributes: {
    ...defaultSchema.attributes,
    // Only the first className rule for a tag applies, so extend it in place.
    a: [
      ...(defaultSchema.attributes?.a ?? []).filter(
        (rule) => !(Array.isArray(rule) && rule[0] === 'className')
      ),
      ['className', 'data-footnote-backref', 'footnote-ref']
    ],
    code: [['className', /^language-./, 'math-inline', 'math-display']]
  }
}

/** rehype plugin entry: sanitize with the chat schema. Must run right after rehype-raw. */
export const rehypeSanitizeChat = [rehypeSanitize, markdownSanitizeSchema] as const

/**
 * KaTeX options shared by every chat renderer. `trust: false` disables
 * \href, \url, \includegraphics and \htmlData, which would otherwise let
 * injected LaTeX create links or load remote images.
 */
export const katexSecurityOptions = {
  trust: false
} as const
