import { motion } from 'framer-motion'

interface Finding {
  file: string
  line: number
  category: string
  severity: string
  title: string
  verdict: string
  score: number
  vote_count: number
  passes_surviving: number
}

const SEVERITY_COLOR: Record<string, string> = {
  high: 'text-red-400',
  medium: 'text-amber-400',
  low: 'text-cyan-400',
}
const VERDICT_COLOR: Record<string, string> = {
  confirmed: 'text-emerald-400',
  uncertain: 'text-amber-400',
  false_positive: 'text-zinc-500',
}

export function FindingsTable({ result }: { result: Record<string, any> }) {
  if (result.skipped) {
    return (
      <div className="w-full max-w-2xl mx-auto mt-4 p-4 rounded-xl border border-amber-900/50 bg-amber-950/20 text-amber-300 text-sm">
        Skipped: {result.skip_reason}
      </div>
    )
  }

  const findings: Finding[] = result.findings ?? []

  return (
    <div className="w-full max-w-2xl mx-auto mt-4 flex flex-col gap-3">
      {findings.length === 0 ? (
        <div className="p-4 rounded-xl border border-emerald-900/50 bg-emerald-950/20 text-emerald-300 text-sm">
          No confirmed findings.
        </div>
      ) : (
        findings.map((f, i) => (
          <motion.div
            key={i}
            initial={{ opacity: 0, y: 6 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: i * 0.05 }}
            className="p-4 rounded-xl border border-zinc-800 bg-zinc-900/40"
          >
            <div className="flex items-center justify-between text-xs mb-1">
              <span className="font-mono text-zinc-400">
                {f.file}:{f.line}
              </span>
              <span className={VERDICT_COLOR[f.verdict] ?? 'text-zinc-400'}>{f.verdict}</span>
            </div>
            <div className="text-sm text-zinc-100 mb-1">{f.title}</div>
            <div className="flex items-center gap-3 text-xs text-zinc-500">
              <span className={SEVERITY_COLOR[f.severity] ?? ''}>
                {f.category}/{f.severity}
              </span>
              <span>score {f.score.toFixed(2)}</span>
              <span>
                {f.vote_count}/{f.passes_surviving} votes
              </span>
            </div>
          </motion.div>
        ))
      )}

      {result.post_result && (
        <div className="text-xs text-zinc-500">
          {result.post_result.reason === 'dry_run'
            ? `Dry run - would post ${result.post_result.count} comment(s).`
            : result.post_result.posted
              ? `Posted ${result.post_result.count} comment(s) to the PR.`
              : `Nothing posted (${result.post_result.reason ?? 'unknown'}).`}
        </div>
      )}
      {result.token_usage && (
        <div className="text-xs text-zinc-600">
          {result.token_usage.total_tokens} tokens used · cost depends on your own API plan
        </div>
      )}

      <button
        onClick={() => downloadFindings(result)}
        className="self-start text-xs px-3 py-1.5 rounded-lg border border-zinc-800 text-zinc-400 hover:text-zinc-200 hover:border-zinc-700 transition-colors"
      >
        Download findings.json
      </button>
    </div>
  )
}

function downloadFindings(result: Record<string, unknown>) {
  const blob = new Blob([JSON.stringify(result, null, 2)], { type: 'application/json' })
  const url = URL.createObjectURL(blob)
  const a = document.createElement('a')
  a.href = url
  a.download = 'findings.json'
  a.click()
  URL.revokeObjectURL(url)
}
