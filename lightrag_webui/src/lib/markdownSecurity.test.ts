import { describe, expect, test } from 'bun:test'
import { createElement } from 'react'
import { renderToStaticMarkup } from 'react-dom/server'
import ReactMarkdown from 'react-markdown'
import rehypeRaw from 'rehype-raw'
import remarkGfm from 'remark-gfm'
import remarkMath from 'remark-math'

import { rehypeSanitizeChat } from './markdownSecurity'
import { remarkFootnotes } from '@/utils/remarkFootnotes'

// Mirrors the plugin order in ChatMessage: raw HTML is parsed, then sanitized.
const render = (markdown: string) =>
  renderToStaticMarkup(
    createElement(
      ReactMarkdown,
      {
        remarkPlugins: [remarkGfm, remarkFootnotes, remarkMath],
        rehypePlugins: [rehypeRaw, rehypeSanitizeChat as any],
        skipHtml: false
      },
      markdown
    )
  )

describe('chat markdown sanitization', () => {
  test.each([
    ['iframe srcdoc', '<iframe srcdoc="<script>alert(1)</script>"></iframe>', /<iframe|srcdoc/i],
    ['script tag', 'hi <script>alert(1)</script>', /<script/i],
    ['event handler', '<img src="x" onerror="alert(1)">', /onerror/i],
    ['javascript link', '<a href="javascript:alert(1)">x</a>', /javascript:/i],
    ['style tag', '<style>body{background:url(https://evil.example/x)}</style>', /<style/i],
    ['inline style attribute', '<span style="background:url(https://evil.example/x)">x</span>', /style=/i],
    ['form', '<form action="https://evil.example"><input name="q"></form>', /<form/i],
    ['meta refresh', '<meta http-equiv="refresh" content="0;url=https://evil.example">', /<meta/i],
    ['object embed', '<object data="https://evil.example/x.swf"></object>', /<object/i],
    ['svg with script', '<svg><script>alert(1)</script></svg>', /<script|<svg/i],
    ['footnote id injection', 'see [^"><img src=x onerror=alert(1)>]', /onerror/i]
  ])('removes %s', (_name, input, forbidden) => {
    expect(render(input)).not.toMatch(forbidden)
  })

  test('keeps harmless formatting', () => {
    const html = render('**bold** <mark>hit</mark> <u>under</u> <sup>2</sup> <del>old</del>')
    expect(html).toContain('<strong>bold</strong>')
    expect(html).toContain('<mark>hit</mark>')
    expect(html).toContain('<u>under</u>')
    expect(html).toContain('<sup>2</sup>')
    expect(html).toContain('<del>old</del>')
  })

  test('keeps tables and code language classes', () => {
    const html = render('| a | b |\n|---|---|\n| 1 | 2 |\n\n```python\nprint(1)\n```')
    expect(html).toContain('<table>')
    expect(html).toContain('class="language-python"')
  })

  test('keeps math classes needed by KaTeX', () => {
    const html = render('inline $x^2$ and\n\n$$\ny=1\n$$')
    expect(html).toMatch(/class="language-math math-inline"/)
    expect(html).toMatch(/class="language-math math-display"/)
  })

  test('keeps footnote references from remarkFootnotes', () => {
    const html = render('claim [^1]')
    expect(html).toContain('class="footnote-ref"')
    expect(html).toContain('>1</a></sup>')
  })
})
