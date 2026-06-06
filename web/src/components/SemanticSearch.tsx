'use client'

import { useState } from 'react'
import Link from 'next/link'

type SearchResult = {
  recording_id: string
  score: number
  task_title: string
  objects: string[]
  detected_objects: string[]
  gemini_score: number
  passed: boolean
  embedding_snippet: string
  supabase_data: { status?: string; created_at?: string } | null
}

function scoreBadgeClasses(score: number): string {
  const pct = score * 100
  if (pct >= 80) return 'bg-[rgba(72,180,97,0.18)] text-[#8ad09a]'
  if (pct >= 60) return 'bg-[rgba(212,177,58,0.18)] text-[#e0c45a]'
  return 'bg-[rgba(115,120,131,0.18)] text-[#d3d7de]'
}

export default function SemanticSearch() {
  const [query, setQuery] = useState('')
  const [loading, setLoading] = useState(false)
  const [results, setResults] = useState<SearchResult[] | null>(null)
  const [error, setError] = useState(false)

  async function runSearch() {
    const q = query.trim()
    if (!q || loading) return
    setLoading(true)
    setError(false)
    try {
      const res = await fetch('/api/search-recordings', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query: q }),
      })
      if (!res.ok) throw new Error('search failed')
      const data = await res.json()
      setResults(data.results ?? [])
    } catch {
      setError(true)
      setResults(null)
    } finally {
      setLoading(false)
    }
  }

  return (
    <section className="surface-panel mb-8 p-5">
      <h2 className="mb-1 text-sm font-semibold uppercase tracking-[0.16em] text-white">
        Semantic search
      </h2>
      <p className="mb-4 text-sm text-[var(--foreground-secondary)]">
        Search across all your processed recordings by what actually happened.
      </p>

      <div className="flex gap-2">
        <input
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === 'Enter') runSearch()
          }}
          placeholder="Search recordings… e.g. 'grasped cylindrical object off table at arm height'"
          className="input-dark text-sm"
        />
        <button
          type="button"
          onClick={runSearch}
          disabled={loading || !query.trim()}
          className="btn-lab shrink-0 rounded-lg px-4 py-2 text-sm font-medium transition-colors disabled:opacity-50"
        >
          {loading ? (
            <span className="flex items-center gap-2">
              <span className="h-3.5 w-3.5 animate-spin rounded-full border-2 border-white/30 border-t-white" />
              Searching
            </span>
          ) : (
            'Search'
          )}
        </button>
      </div>

      {error && (
        <p className="mt-4 text-sm text-[#e08a8a]">Search unavailable. Please try again.</p>
      )}

      {results !== null && !error && results.length === 0 && (
        <p className="mt-4 text-sm text-[var(--foreground-secondary)]">
          No matching recordings found. Try different search terms.
        </p>
      )}

      {results && results.length > 0 && (
        <div className="mt-5 space-y-3">
          {results.map((r) => {
            const pct = Math.round(r.score * 100)
            return (
              <div
                key={r.recording_id}
                className={`surface-panel p-4 ${r.score < 0.5 ? 'opacity-60' : ''}`}
              >
                <div className="flex items-start justify-between gap-3">
                  <div className="min-w-0">
                    <h3 className="truncate font-semibold text-white">
                      {r.task_title || 'Untitled recording'}
                    </h3>
                    <div className="mt-1 flex flex-wrap items-center gap-2 text-xs text-[var(--foreground-secondary)]">
                      <span>Quality {r.gemini_score}/10</span>
                      <span>·</span>
                      <span className={r.passed ? 'text-[#8ad09a]' : 'text-[#e08a8a]'}>
                        {r.passed ? '✓ passed' : '✕ failed'}
                      </span>
                    </div>
                    {r.detected_objects.length > 0 && (
                      <div className="mt-2 flex flex-wrap gap-1.5">
                        {r.detected_objects.slice(0, 8).map((obj) => (
                          <span
                            key={obj}
                            className="rounded-full bg-[rgba(255,255,255,0.06)] px-2 py-0.5 text-[11px] text-[var(--foreground-secondary)]"
                          >
                            {obj}
                          </span>
                        ))}
                      </div>
                    )}
                  </div>
                  <div className="flex shrink-0 flex-col items-end gap-2">
                    <span className={`rounded-full px-2 py-1 text-xs font-semibold ${scoreBadgeClasses(r.score)}`}>
                      {pct}% match
                    </span>
                    <Link
                      href={`/studio/${r.recording_id}`}
                      className="btn-neutral rounded-lg px-3 py-1.5 text-xs font-medium"
                    >
                      Open in Studio
                    </Link>
                  </div>
                </div>
              </div>
            )
          })}
        </div>
      )}
    </section>
  )
}
