import { motion, AnimatePresence } from 'framer-motion'
import type { Stage } from '../lib/ws'

export function StepList({ stages }: { stages: Stage[] }) {
  return (
    <div className="w-full max-w-2xl mx-auto flex flex-col gap-1.5">
      <AnimatePresence initial={false}>
        {stages.map((stage, i) => (
          <StepCard key={stage.key} stage={stage} index={i} />
        ))}
      </AnimatePresence>
    </div>
  )
}

function StepCard({ stage, index }: { stage: Stage; index: number }) {
  const color =
    stage.status === 'done'
      ? 'text-emerald-400 border-emerald-900/50 bg-emerald-950/20'
      : stage.status === 'running'
        ? 'text-cyan-300 border-cyan-800/50 bg-cyan-950/20'
        : 'text-zinc-600 border-zinc-800/50 bg-zinc-900/20'

  return (
    <motion.div
      initial={{ opacity: 0, x: -8 }}
      animate={{ opacity: 1, x: 0 }}
      transition={{ duration: 0.25, delay: index * 0.02 }}
      className={`flex items-center gap-3 px-3 py-2 rounded-lg border text-sm transition-colors duration-300 ${color}`}
    >
      <StepIcon status={stage.status} />
      <span className="flex-1">{stage.label}</span>
      {stage.detail && <span className="text-xs opacity-70 font-mono">{stage.detail}</span>}
    </motion.div>
  )
}

function StepIcon({ status }: { status: Stage['status'] }) {
  if (status === 'done') {
    return (
      <motion.span
        initial={{ scale: 0.6, opacity: 0 }}
        animate={{ scale: 1, opacity: 1 }}
        className="text-emerald-400"
      >
        ✓
      </motion.span>
    )
  }
  if (status === 'running') {
    return (
      <motion.span
        animate={{ rotate: 360 }}
        transition={{ repeat: Infinity, duration: 1, ease: 'linear' }}
        className="inline-block w-3 h-3 border-2 border-cyan-400 border-t-transparent rounded-full"
      />
    )
  }
  return <span className="w-3 h-3 rounded-full border border-zinc-700 inline-block" />
}
