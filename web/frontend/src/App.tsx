import { useState } from 'react'
import { ApiKeysPanel, loadStoredKeys } from './components/ApiKeysPanel'
import { UrlBar } from './components/UrlBar'
import { StepList } from './components/StepList'
import { FindingsTable } from './components/FindingsTable'
import {
  initStages,
  applyStageMessage,
  startReview,
  type ReviewKeys,
  type Stage,
  type ServerMessage,
} from './lib/ws'

type Phase = 'idle' | 'running' | 'done' | 'error'

function App() {
  const [keys, setKeys] = useState<ReviewKeys>(loadStoredKeys)
  const [phase, setPhase] = useState<Phase>('idle')
  const [stages, setStages] = useState<Stage[]>([])
  const [result, setResult] = useState<Record<string, unknown> | null>(null)
  const [error, setError] = useState<string | null>(null)

  const configComplete = (c: { baseUrl: string; apiKey: string; model: string }) =>
    Boolean(c.baseUrl && c.apiKey && c.model)
  const allKeysSet =
    Boolean(keys.githubToken) && configComplete(keys.a1) && configComplete(keys.a2) && configComplete(keys.a3Primary)

  const handleSubmit = (url: string, post: boolean) => {
    if (!allKeysSet) {
      setError('Set your GitHub token and Reviewer/Validator provider keys above first.')
      return
    }
    setError(null)
    setResult(null)
    setStages(initStages())
    setPhase('running')

    startReview(url, keys, post, {
      onMessage: (msg: ServerMessage) => {
        if (msg.type === 'stage') {
          setStages((prev) => applyStageMessage(prev, msg))
        } else if (msg.type === 'result') {
          setResult(msg.data)
          setPhase('done')
        } else if (msg.type === 'error') {
          setError(msg.message)
          setPhase('error')
        }
      },
      onClose: () => {
        setPhase((p) => (p === 'running' ? 'error' : p))
      },
    })
  }

  return (
    <div className="min-h-screen bg-[#0a0a0f] text-zinc-100 flex flex-col items-center px-4 py-10">
      <header className="text-center mb-8">
        <h1 className="text-2xl font-semibold tracking-tight">oss-bugbot</h1>
        <p className="text-sm text-zinc-500 mt-1">
          AI code review · multi-pass reviewer · vote · adversarial validate · bring your own provider
        </p>
      </header>

      <ApiKeysPanel keys={keys} onChange={setKeys} />
      <UrlBar disabled={phase === 'running'} onSubmit={handleSubmit} />

      {error && (
        <div className="w-full max-w-2xl mx-auto mt-4 p-3 rounded-lg border border-red-900/50 bg-red-950/20 text-red-300 text-sm">
          {error}
        </div>
      )}

      {phase !== 'idle' && (
        <div className="w-full mt-8">
          <StepList stages={stages} />
        </div>
      )}

      {phase === 'done' && result && <FindingsTable result={result} />}
    </div>
  )
}

export default App
