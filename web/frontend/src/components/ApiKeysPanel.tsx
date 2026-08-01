import { useEffect, useState } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { emptyReviewKeys, type ProviderConfig, type ReviewKeys } from '../lib/ws'

const STORAGE_KEY = 'oss-bugbot:keys'

export function loadStoredKeys(): ReviewKeys {
  try {
    const raw = localStorage.getItem(STORAGE_KEY)
    if (raw) return { ...emptyReviewKeys(), ...(JSON.parse(raw) as Partial<ReviewKeys>) }
  } catch {
    // ignore malformed/blocked storage, fall through to empty
  }
  return emptyReviewKeys()
}

function configComplete(c: ProviderConfig): boolean {
  return Boolean(c.baseUrl && c.apiKey && c.model)
}

export function ApiKeysPanel({
  keys,
  onChange,
}: {
  keys: ReviewKeys
  onChange: (k: ReviewKeys) => void
}) {
  const allSet =
    Boolean(keys.githubToken) &&
    configComplete(keys.a1) &&
    configComplete(keys.a2) &&
    configComplete(keys.a3Primary)
  const [open, setOpen] = useState(() => !allSet)
  const [fallbackOpen, setFallbackOpen] = useState(() => configComplete(keys.a3Fallback))

  useEffect(() => {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(keys))
  }, [keys])

  const setGithubToken = (e: React.ChangeEvent<HTMLInputElement>) =>
    onChange({ ...keys, githubToken: e.target.value })

  // Reviewer's base_url + api_key are shared across A1/A2 by default (one
  // provider, two model choices) - the common case. Each model name stays
  // independent since A1 (per-hunk reviewer) and A2 (aggregator) often
  // want different model sizes even from the same provider.
  const setReviewerShared = (field: 'baseUrl' | 'apiKey') => (e: React.ChangeEvent<HTMLInputElement>) => {
    const value = e.target.value
    onChange({ ...keys, a1: { ...keys.a1, [field]: value }, a2: { ...keys.a2, [field]: value } })
  }
  const setReviewerModel = (role: 'a1' | 'a2') => (e: React.ChangeEvent<HTMLInputElement>) =>
    onChange({ ...keys, [role]: { ...keys[role], model: e.target.value } })

  const setValidator = (role: 'a3Primary' | 'a3Fallback', field: keyof ProviderConfig) =>
    (e: React.ChangeEvent<HTMLInputElement>) =>
      onChange({ ...keys, [role]: { ...keys[role], [field]: e.target.value } })

  return (
    <div className="w-full max-w-2xl mx-auto mb-4">
      <button
        onClick={() => setOpen((o) => !o)}
        className="flex items-center gap-2 text-sm text-zinc-400 hover:text-zinc-200 transition-colors"
      >
        <span className={`inline-block w-2 h-2 rounded-full ${allSet ? 'bg-emerald-500' : 'bg-amber-500'}`} />
        Provider keys {allSet ? 'set' : 'required'} {open ? '▲' : '▼'}
      </button>
      <AnimatePresence>
        {open && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: 'auto', opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.2 }}
            className="overflow-hidden"
          >
            <div className="mt-3 flex flex-col gap-4 p-4 rounded-xl border border-zinc-800 bg-zinc-900/60">
              <KeyInput
                label="GitHub token"
                value={keys.githubToken}
                onChange={setGithubToken}
                placeholder="ghp_... (classic PAT, public_repo scope)"
              />

              <Section title="Reviewer (A1 + A2)" hint="any OpenAI-compatible provider - OpenAI, Groq, Gemini, Together, OpenRouter, local Ollama, ...">
                <KeyInput
                  label="Base URL"
                  value={keys.a1.baseUrl}
                  onChange={setReviewerShared('baseUrl')}
                  placeholder="https://api.openai.com/v1"
                />
                <KeyInput
                  label="API key"
                  value={keys.a1.apiKey}
                  onChange={setReviewerShared('apiKey')}
                  placeholder="sk-..."
                />
                <div className="grid grid-cols-2 gap-2">
                  <KeyInput label="A1 model" value={keys.a1.model} onChange={setReviewerModel('a1')} placeholder="gpt-4o-mini" text />
                  <KeyInput label="A2 model" value={keys.a2.model} onChange={setReviewerModel('a2')} placeholder="gpt-4o-mini" text />
                </div>
              </Section>

              <Section title="Validator (A3)" hint="ideally a different provider from Reviewer, for genuine adversarial independence">
                <KeyInput label="Base URL" value={keys.a3Primary.baseUrl} onChange={setValidator('a3Primary', 'baseUrl')} placeholder="https://api.groq.com/openai/v1" />
                <KeyInput label="API key" value={keys.a3Primary.apiKey} onChange={setValidator('a3Primary', 'apiKey')} placeholder="gsk_..." />
                <KeyInput label="Model" value={keys.a3Primary.model} onChange={setValidator('a3Primary', 'model')} placeholder="openai/gpt-oss-120b" text />

                <button
                  onClick={() => setFallbackOpen((o) => !o)}
                  className="text-xs text-zinc-500 hover:text-zinc-300 text-left mt-1 transition-colors"
                >
                  {fallbackOpen ? '▲' : '▼'} Fallback provider (optional - used if the primary refuses)
                </button>
                <AnimatePresence>
                  {fallbackOpen && (
                    <motion.div
                      initial={{ height: 0, opacity: 0 }}
                      animate={{ height: 'auto', opacity: 1 }}
                      exit={{ height: 0, opacity: 0 }}
                      transition={{ duration: 0.2 }}
                      className="overflow-hidden flex flex-col gap-2"
                    >
                      <KeyInput label="Base URL" value={keys.a3Fallback.baseUrl} onChange={setValidator('a3Fallback', 'baseUrl')} placeholder="https://generativelanguage.googleapis.com/v1beta/openai/" />
                      <KeyInput label="API key" value={keys.a3Fallback.apiKey} onChange={setValidator('a3Fallback', 'apiKey')} placeholder="AIza..." />
                      <KeyInput label="Model" value={keys.a3Fallback.model} onChange={setValidator('a3Fallback', 'model')} placeholder="gemini-3.5-flash" text />
                    </motion.div>
                  )}
                </AnimatePresence>
              </Section>

              <p className="text-xs text-zinc-500">
                Stored only in your browser (localStorage), sent directly to the backend once when
                you start a review. Never logged, never stored server-side.
              </p>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  )
}

function Section({ title, hint, children }: { title: string; hint: string; children: React.ReactNode }) {
  return (
    <div className="flex flex-col gap-2 pt-1 border-t border-zinc-800/80">
      <div className="pt-3">
        <div className="text-sm text-zinc-300 font-medium">{title}</div>
        <div className="text-xs text-zinc-600">{hint}</div>
      </div>
      {children}
    </div>
  )
}

function KeyInput({
  label,
  value,
  onChange,
  placeholder,
  text = false,
}: {
  label: string
  value: string
  onChange: (e: React.ChangeEvent<HTMLInputElement>) => void
  placeholder: string
  text?: boolean
}) {
  return (
    <label className="flex flex-col gap-1">
      <span className="text-xs text-zinc-400">{label}</span>
      <input
        type={text ? 'text' : 'password'}
        value={value}
        onChange={onChange}
        placeholder={placeholder}
        autoComplete="off"
        className="bg-zinc-950 border border-zinc-800 rounded-lg px-3 py-2 text-sm text-zinc-100 placeholder:text-zinc-600 focus:outline-none focus:border-cyan-600 transition-colors"
      />
    </label>
  )
}
