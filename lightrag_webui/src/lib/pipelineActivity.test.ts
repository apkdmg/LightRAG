import { describe, expect, test } from 'bun:test'
import { pipelineActiveFrom, resolvePipelineActive } from './pipelineActivity'

const idleOwn = { pipeline_active: false, pipeline_busy: false }
const busyOwn = { pipeline_active: true, pipeline_busy: true }

describe('pipeline activity source (OQ-13 acting mode)', () => {
  test('the four pipeline_status keys make it active, like /health', () => {
    expect(pipelineActiveFrom(null)).toBe(false)
    expect(pipelineActiveFrom({})).toBe(false)
    expect(pipelineActiveFrom({ busy: true })).toBe(true)
    expect(pipelineActiveFrom({ scanning: true })).toBe(true)
    expect(pipelineActiveFrom({ destructive_busy: true })).toBe(true)
    expect(pipelineActiveFrom({ pending_enqueues: 2 })).toBe(true)
    expect(pipelineActiveFrom({ busy: false, pending_enqueues: 0 })).toBe(false)
  })

  test('own workspace: /health decides and the target is never read', async () => {
    let reads = 0
    const read = async () => {
      reads += 1
      return { busy: true }
    }
    expect(await resolvePipelineActive(busyOwn, false, read)).toBe(true)
    expect(await resolvePipelineActive(idleOwn, false, read)).toBe(false)
    expect(await resolvePipelineActive({ pipeline_busy: true }, false, read)).toBe(true)
    expect(reads).toBe(0)
  })

  test('acting: the target decides, whatever own-workspace /health says', async () => {
    expect(await resolvePipelineActive(idleOwn, true, async () => ({ scanning: true }))).toBe(true)
    expect(await resolvePipelineActive(busyOwn, true, async () => ({ busy: false }))).toBe(false)
  })

  test('acting and the target unreadable: null, so the last value is kept', async () => {
    expect(
      await resolvePipelineActive(busyOwn, true, async () => {
        throw new Error('403')
      })
    ).toBeNull()
  })
})
