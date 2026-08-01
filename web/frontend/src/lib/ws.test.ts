import { describe, expect, it } from 'vitest'
import { applyStageMessage, initStages, BASE_STAGE_DEFS } from './ws'

describe('initStages', () => {
  it('starts with only the first stage running, rest pending', () => {
    const stages = initStages()
    expect(stages).toHaveLength(BASE_STAGE_DEFS.length)
    expect(stages[0].status).toBe('running')
    expect(stages.slice(1).every((s) => s.status === 'pending')).toBe(true)
  })
})

describe('applyStageMessage', () => {
  it('marks the matching stage done and advances the next pending one to running', () => {
    const stages = initStages()
    const next = applyStageMessage(stages, {
      type: 'stage',
      stage: 'resolving',
      label: 'Resolving PR',
      detail: 'main @ abc1234',
      status: 'done',
    })
    expect(next[0].status).toBe('done')
    expect(next[0].detail).toBe('main @ abc1234')
    expect(next[1].status).toBe('running') // "cloning" advances
  })

  it('expands a1_pass_done into the specific pass slot via detail, not stage name', () => {
    let stages = initStages()
    // simulate real observed out-of-order arrival: p4 before p1
    stages = applyStageMessage(stages, {
      type: 'stage', stage: 'a1_pass_done', label: 'AI review pass', detail: 'p4', status: 'done',
    })
    stages = applyStageMessage(stages, {
      type: 'stage', stage: 'a1_pass_done', label: 'AI review pass', detail: 'p1', status: 'done',
    })
    const p4 = stages.find((s) => s.key === 'a1_p4')!
    const p1 = stages.find((s) => s.key === 'a1_p1')!
    const p2 = stages.find((s) => s.key === 'a1_p2')!
    expect(p4.status).toBe('done')
    expect(p1.status).toBe('done')
    expect(p2.status).toBe('pending') // p2/p3 untouched by p1/p4 arriving
  })

  it('ignores an unknown stage key without throwing', () => {
    const stages = initStages()
    const next = applyStageMessage(stages, {
      type: 'stage', stage: 'totally_unknown', label: 'x', detail: '', status: 'done',
    })
    expect(next).toEqual(stages)
  })
})
