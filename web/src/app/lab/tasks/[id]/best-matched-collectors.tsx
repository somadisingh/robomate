'use client'

import { useEffect, useState } from 'react'

type CollectorMatch = {
  collector_id: string
  display_name: string | null
  match_score: number
  total_approved: number
  avg_score: number
  object_specialties: string[]
}

export default function BestMatchedCollectors({
  taskTitle,
  taskDescription,
  objects,
}: {
  taskTitle: string
  taskDescription: string | null
  objects: string[]
}) {
  const [matches, setMatches] = useState<CollectorMatch[] | null>(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    let cancelled = false
    ;(async () => {
      try {
        const res = await fetch('/api/match-collectors', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ taskTitle, taskDescription, objects }),
        })
        const data = await res.json()
        if (!cancelled) setMatches(data.matches ?? [])
      } catch {
        if (!cancelled) setMatches([])
      } finally {
        if (!cancelled) setLoading(false)
      }
    })()
    return () => {
      cancelled = true
    }
  }, [taskTitle, taskDescription, objects])

  // Nothing to show until collectors have profiles — keep the page clean.
  if (!loading && (!matches || matches.length === 0)) return null

  return (
    <div className="surface-panel mb-6 p-6">
      <h2 className="text-sm font-semibold uppercase tracking-[0.16em] text-white">
        Best matched collectors
      </h2>
      <p className="mt-1 text-sm text-[var(--foreground-secondary)]">
        Collectors likely to succeed at this task, ranked by past performance.
      </p>

      {loading ? (
        <div className="mt-4 flex items-center gap-2 text-sm text-[var(--foreground-secondary)]">
          <span className="h-3.5 w-3.5 animate-spin rounded-full border-2 border-white/30 border-t-white" />
          Finding matches…
        </div>
      ) : (
        <ol className="mt-4 space-y-2">
          {matches!.slice(0, 5).map((m, idx) => (
            <li
              key={m.collector_id}
              className="flex items-start justify-between gap-3 rounded-md bg-[rgba(255,255,255,0.04)] px-3 py-2.5"
            >
              <div className="min-w-0">
                <div className="flex items-center gap-2">
                  <span className="text-sm font-semibold text-[#aebeff]">#{idx + 1}</span>
                  <span className="truncate text-sm text-white">
                    {m.display_name || `${m.collector_id.slice(0, 8)}…`}
                  </span>
                </div>
                <p className="mt-0.5 text-xs text-[var(--foreground-secondary)]">
                  Match {Math.round(m.match_score * 100)}% · Avg score{' '}
                  {Number(m.avg_score).toFixed(1)}/10 · {m.total_approved} approved demos
                </p>
                {m.object_specialties.length > 0 && (
                  <p className="mt-1 truncate text-xs text-[var(--foreground-secondary)]">
                    Specialties: {m.object_specialties.slice(0, 6).join(', ')}
                  </p>
                )}
              </div>
            </li>
          ))}
        </ol>
      )}
    </div>
  )
}
