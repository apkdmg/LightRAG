import { describe, expect, test } from 'bun:test'
import {
  DEFAULT_BACKLOG,
  MAX_PAGE_SIZE,
  MAX_QUERY_LENGTH,
  backlogQuery,
  backlogQueryString,
  defaultDir,
  nextSort,
  pageRange,
  toggleFilter
} from './backlog'

describe('backlogQuery (GET /processing/workspaces)', () => {
  test('the default view asks for the first page, oldest in-flight first', () => {
    expect(backlogQuery(DEFAULT_BACKLOG)).toEqual({ sort: 'oldest', dir: 'asc', page: 1, page_size: 50 })
  })

  test('empty values are left out of the query string', () => {
    expect(backlogQueryString({ ...DEFAULT_BACKLOG, q: '   ' })).toBe('sort=oldest&dir=asc&page=1&page_size=50')
  })

  test('the search is trimmed, capped and encoded', () => {
    expect(backlogQuery({ ...DEFAULT_BACKLOG, q: '  a.bin@unimas.my ' }).q).toBe('a.bin@unimas.my')
    expect(String(backlogQuery({ ...DEFAULT_BACKLOG, q: 'x'.repeat(500) }).q).length).toBe(MAX_QUERY_LENGTH)
    // The route accepts q up to 255 characters (the longest workspace ID); a longer one would be a 422.
    expect(MAX_QUERY_LENGTH).toBe(255)
    expect(backlogQueryString({ ...DEFAULT_BACKLOG, q: 'a b&c' })).toBe('q=a+b%26c&sort=oldest&dir=asc&page=1&page_size=50')
  })

  test('filters are canonical: known ones only, in a fixed order', () => {
    expect(backlogQuery({ ...DEFAULT_BACKLOG, has: ['paused', 'failed', 'email'] }).has).toBe('failed,email,paused')
    expect(backlogQuery({ ...DEFAULT_BACKLOG, has: ['bogus' as any, 'stuck'] }).has).toBe('stuck')
    expect(backlogQuery({ ...DEFAULT_BACKLOG, has: [] }).has).toBeUndefined()
  })

  test('two equal views produce the same string (one server cache entry)', () => {
    const a = backlogQueryString({ ...DEFAULT_BACKLOG, has: ['email', 'failed'], q: 'fsktm' })
    const b = backlogQueryString({ ...DEFAULT_BACKLOG, has: ['failed', 'email'], q: ' fsktm ' })
    expect(a).toBe(b)
  })

  test('the page size is capped at 200 and the page is at least 1', () => {
    expect(backlogQuery({ ...DEFAULT_BACKLOG, pageSize: 5000 }).page_size).toBe(MAX_PAGE_SIZE)
    expect(backlogQuery({ ...DEFAULT_BACKLOG, pageSize: 0 }).page_size).toBe(1)
    expect(backlogQuery({ ...DEFAULT_BACKLOG, pageSize: Number.NaN }).page_size).toBe(50)
    expect(backlogQuery({ ...DEFAULT_BACKLOG, page: 0 }).page).toBe(1)
    expect(backlogQuery({ ...DEFAULT_BACKLOG, page: 3.7 }).page).toBe(3)
  })

  test('an unknown sort falls back to the default, and a missing direction follows the sort', () => {
    expect(backlogQuery({ sort: 'drop table' as any })).toMatchObject({ sort: 'oldest', dir: 'asc' })
    expect(backlogQuery({ sort: 'failed' })).toMatchObject({ sort: 'failed', dir: 'desc' })
    expect(backlogQuery({ sort: 'name', dir: 'sideways' as any })).toMatchObject({ sort: 'name', dir: 'asc' })
  })
})

describe('sorting and filters', () => {
  test('natural directions: oldest first, most in flight or failed first, names A→Z', () => {
    expect(defaultDir('oldest')).toBe('asc')
    expect(defaultDir('inflight')).toBe('desc')
    expect(defaultDir('failed')).toBe('desc')
    expect(defaultDir('name')).toBe('asc')
  })

  test('clicking the active sort flips it; a new sort starts in its natural direction', () => {
    expect(nextSort({ sort: 'oldest', dir: 'asc' }, 'oldest')).toEqual({ sort: 'oldest', dir: 'desc' })
    expect(nextSort({ sort: 'oldest', dir: 'desc' }, 'failed')).toEqual({ sort: 'failed', dir: 'desc' })
  })

  test('toggling a chip keeps the canonical order', () => {
    expect(toggleFilter([], 'paused')).toEqual(['paused'])
    expect(toggleFilter(['paused'], 'failed')).toEqual(['failed', 'paused'])
    expect(toggleFilter(['failed', 'paused'], 'failed')).toEqual(['paused'])
  })
})

describe('pageRange', () => {
  test('"1–50 of 1,312"', () => {
    expect(pageRange(1, 50, 1312)).toEqual({ first: 1, last: 50, pages: 27 })
    expect(pageRange(27, 50, 1312)).toEqual({ first: 1301, last: 1312, pages: 27 })
  })

  test('an empty result and an out-of-range page stay sensible', () => {
    expect(pageRange(1, 50, 0)).toEqual({ first: 0, last: 0, pages: 1 })
    expect(pageRange(99, 50, 120)).toEqual({ first: 101, last: 120, pages: 3 })
  })
})
