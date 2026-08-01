import { useState } from 'react'

export function UrlBar({
  disabled,
  onSubmit,
}: {
  disabled: boolean
  onSubmit: (url: string, post: boolean) => void
}) {
  const [url, setUrl] = useState('')
  const [post, setPost] = useState(false)

  const submit = () => {
    if (!url.trim() || disabled) return
    onSubmit(url.trim(), post)
  }

  return (
    <div className="w-full max-w-2xl mx-auto">
      <div className="flex items-center gap-2 bg-zinc-900/80 border border-zinc-800 rounded-xl px-3 py-2 focus-within:border-cyan-600 transition-colors">
        <span className="text-zinc-500 text-sm select-none">›</span>
        <input
          value={url}
          onChange={(e) => setUrl(e.target.value)}
          onKeyDown={(e) => e.key === 'Enter' && submit()}
          disabled={disabled}
          placeholder="github.com/owner/repo or .../pull/N"
          className="flex-1 bg-transparent outline-none text-sm text-zinc-100 placeholder:text-zinc-600 disabled:opacity-50"
        />
        <button
          onClick={submit}
          disabled={disabled || !url.trim()}
          className="text-sm px-3 py-1.5 rounded-lg bg-cyan-600 hover:bg-cyan-500 disabled:bg-zinc-800 disabled:text-zinc-600 text-white transition-colors"
        >
          Review
        </button>
      </div>
      <label className="flex items-center gap-2 mt-2 text-xs text-zinc-500 select-none">
        <input
          type="checkbox"
          checked={post}
          onChange={(e) => setPost(e.target.checked)}
          disabled={disabled}
          className="accent-cyan-600"
        />
        Actually post the review to GitHub (default: dry run only)
      </label>
    </div>
  )
}
