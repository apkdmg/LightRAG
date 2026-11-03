import { describe, expect, test } from 'bun:test'
import { STATES, isMultiWorker, isStale, lookOf, rankOf, runLookOf, toProcessMode, toState, workersState, worstOf } from './severity'

describe('severity', () => {
  test('every state has an icon, a tone and a label', () => {
    for (const s of STATES) {
      const look = lookOf(s)
      expect(look.icon).toBeTruthy()
      expect(look.tone).toBeTruthy()
      expect(look.label).toBe(`operations.state.${s}`)
    }
  })

  test('server spellings map onto the vocabulary; anything else is unknown', () => {
    expect(toState('healthy')).toBe('ok')
    expect(toState('DEGRADED')).toBe('warning')
    expect(toState('down')).toBe('critical')
    expect(toState('stale')).toBe('unknown')
    expect(toState('???')).toBe('unknown')
    expect(toState(null)).toBe('unknown')
  })

  test('rank orders OK < Warning < Serious < Critical; other states are not a worsening', () => {
    expect(rankOf('ok')).toBe(0)
    expect(rankOf('warning')).toBe(1)
    expect(rankOf('serious')).toBe(2)
    expect(rankOf('critical')).toBe(3)
    expect(rankOf('running')).toBe(0)
    expect(rankOf('maintenance')).toBe(0)
  })

  test('worstOf: the worst state wins, and unknown beats OK in a summary', () => {
    expect(worstOf([])).toBe('unknown')
    expect(worstOf(['ok', 'warning', 'ok'])).toBe('warning')
    expect(worstOf(['ok', 'critical', 'serious'])).toBe('critical')
    expect(worstOf(['ok', 'unknown'])).toBe('unknown')
    expect(worstOf(['unknown', 'warning'])).toBe('warning')
    expect(worstOf(['ok', 'running'])).toBe('ok')
  })

  test('stale after twice the source freshness', () => {
    const now = 1_000_000
    expect(isStale(now - 59_000, 30, now)).toBe(false)
    expect(isStale(now - 61_000, 30, now)).toBe(true)
    expect(isStale(now - 29 * 60_000, 900, now)).toBe(false)
    expect(isStale(now - 31 * 60_000, 900, now)).toBe(true)
    expect(isStale(null, 30, now)).toBe(true)
  })

  test('run states carry an explanation and map onto a severity', () => {
    expect(runLookOf('orphaned')).toMatchObject({ state: 'critical', label: 'operations.run.orphaned.label', explain: 'operations.run.orphaned.explain' })
    expect(runLookOf('paused')).toMatchObject({ state: 'paused', icon: 'pause' })
    expect(runLookOf('reservation').state).toBe('critical')
    expect(runLookOf('hung').state).toBe('serious')
    expect(runLookOf('stale_doc').state).toBe('warning')
    expect(runLookOf('mystery')).toMatchObject({ state: 'unknown', explain: 'operations.run.unknown.explain' })
  })

  test('process mode and worker counts', () => {
    expect(toProcessMode('uvicorn')).toBe('uvicorn')
    expect(toProcessMode('gunicorn')).toBe('gunicorn')
    expect(toProcessMode('hypercorn')).toBe('unknown')
    expect(isMultiWorker('gunicorn')).toBe(true)
    expect(isMultiWorker('uvicorn')).toBe(false)
    expect(workersState(3, 3)).toBe('ok')
    expect(workersState(3, 2)).toBe('warning')
    expect(workersState(3, 0)).toBe('critical')
    expect(workersState(3, null)).toBe('unknown')
    expect(workersState(null, 2)).toBe('unknown')
  })
})
