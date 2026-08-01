// WebSocket client for /ws/review, plus the message->UI-state reducer.
// The reducer is the one piece of real branching logic in this app - see
// ws.test.ts. Everything else here is thin connection plumbing.

export type StageStatus = 'pending' | 'running' | 'done'

export interface Stage {
  key: string
  label: string
  detail: string
  status: StageStatus
}

// Same stage order/labels the backend (web/backend/app.py) and the TUI
// (src/tui.py) already established. a1_pass_done fires once per A1 pass
// (4x, possibly out of order - already live-verified) with detail "p1".."p4",
// so it's expanded into 4 distinct slots here rather than one bucket.
export const BASE_STAGE_DEFS: { key: string; label: string }[] = [
  { key: 'resolving', label: 'Resolving PR' },
  { key: 'cloning', label: 'Cloning' },
  { key: 'size_gate', label: 'Size gate' },
  { key: 'diff_fetched', label: 'Diff fetched' },
  { key: 'semgrep_done', label: 'Semgrep scan' },
  { key: 'a1_p1', label: 'AI review · pass 1' },
  { key: 'a1_p2', label: 'AI review · pass 2' },
  { key: 'a1_p3', label: 'AI review · pass 3' },
  { key: 'a1_p4', label: 'AI review · pass 4' },
  { key: 'a2_done', label: 'Cluster + vote' },
  { key: 'a3_done', label: 'Adversarial validate' },
  { key: 'posted', label: 'Post' },
]

export function initStages(): Stage[] {
  return BASE_STAGE_DEFS.map((s, i) => ({
    ...s,
    detail: '',
    status: i === 0 ? 'running' : 'pending',
  }))
}

export type StageMessage = {
  type: 'stage'
  stage: string
  label: string
  detail: string
  status: 'running' | 'done'
}
export type ResultMessage = { type: 'result'; data: Record<string, unknown> }
export type ErrorMessage = { type: 'error'; stage: string; message: string }
export type ServerMessage = StageMessage | ResultMessage | ErrorMessage

function keyFor(msg: StageMessage): string {
  if (msg.stage === 'a1_pass_done') return `a1_${msg.detail}`
  return msg.stage
}

/** Pure reducer: apply one 'stage' message to the current stage list. */
export function applyStageMessage(stages: Stage[], msg: StageMessage): Stage[] {
  const key = keyFor(msg)
  const idx = stages.findIndex((s) => s.key === key)
  if (idx === -1) return stages

  const next = stages.slice()
  next[idx] = {
    ...next[idx],
    status: msg.status,
    detail: msg.stage === 'a1_pass_done' ? '' : msg.detail,
  }

  if (msg.status === 'done') {
    const nextPendingIdx = next.findIndex((s) => s.status === 'pending')
    if (nextPendingIdx !== -1) {
      next[nextPendingIdx] = { ...next[nextPendingIdx], status: 'running' }
    }
  }
  return next
}

export interface ProviderConfig {
  baseUrl: string
  apiKey: string
  model: string
}

export function emptyProviderConfig(): ProviderConfig {
  return { baseUrl: '', apiKey: '', model: '' }
}

// Any OpenAI-compatible provider, not just Gemini/Groq - a1/a2 are grouped
// as "Reviewer" in the UI, a3Primary/a3Fallback as "Validator" (ideally a
// different provider from Reviewer, for genuine adversarial independence -
// see README's architecture section), but the wire shape keeps all four
// fully independent, matching the backend's ProviderConfig.
export interface ReviewKeys {
  githubToken: string
  a1: ProviderConfig
  a2: ProviderConfig
  a3Primary: ProviderConfig
  a3Fallback: ProviderConfig
}

export function emptyReviewKeys(): ReviewKeys {
  return {
    githubToken: '',
    a1: emptyProviderConfig(),
    a2: emptyProviderConfig(),
    a3Primary: emptyProviderConfig(),
    a3Fallback: emptyProviderConfig(),
  }
}

export function backendWsUrl(): string {
  const base = (import.meta.env.VITE_BACKEND_URL as string | undefined) ?? 'http://localhost:8000'
  return base.replace(/^http/, 'ws') + '/ws/review'
}

export function startReview(
  url: string,
  keys: ReviewKeys,
  post: boolean,
  handlers: {
    onMessage: (msg: ServerMessage) => void
    onClose: () => void
  },
): WebSocket {
  const wireConfig = (c: ProviderConfig) => ({ base_url: c.baseUrl, api_key: c.apiKey, model: c.model })
  const socket = new WebSocket(backendWsUrl())
  socket.onopen = () => {
    socket.send(
      JSON.stringify({
        type: 'start',
        url,
        github_token: keys.githubToken,
        a1: wireConfig(keys.a1),
        a2: wireConfig(keys.a2),
        a3_primary: wireConfig(keys.a3Primary),
        a3_fallback: wireConfig(keys.a3Fallback),
        post,
      }),
    )
  }
  socket.onmessage = (event) => {
    handlers.onMessage(JSON.parse(event.data) as ServerMessage)
  }
  socket.onclose = () => handlers.onClose()
  return socket
}
