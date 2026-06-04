'use client'

import { useState, useRef, useEffect } from 'react'
import Link from 'next/link'

interface SearchClip {
  video_id: string
  start: number
  end: number
  rank: number
  score?: number
  confidence?: string
  thumbnail_url?: string
  submissionId: string | null
  taskId: string | null
  signedUrl: string | null
}

interface IndexedVideo {
  submissionId: string
  taskId: string | null
  taskTitle: string | null
  twelvelabsVideoId: string | null
  signedUrl: string | null
  createdAt: string
}

type IndexAllStatus = 'idle' | 'running' | 'done' | 'error'

export default function LabSearch() {
  const [query, setQuery] = useState('')
  const [clips, setClips] = useState<SearchClip[]>([])
  const [allVideos, setAllVideos] = useState<IndexedVideo[]>([])
  const [loading, setLoading] = useState(false)
  const [loadingAll, setLoadingAll] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [lastQuery, setLastQuery] = useState('')
  const [indexAllStatus, setIndexAllStatus] = useState<IndexAllStatus>('idle')
  const [indexAllResult, setIndexAllResult] = useState<{ indexed: number; failed: number; total: number } | null>(null)
  const [indexAllError, setIndexAllError] = useState<string | null>(null)
  const inputRef = useRef<HTMLInputElement>(null)

  async function refreshAllVideos() {
    const res = await fetch('/api/videos')
    if (res.ok) {
      const data = await res.json()
      setAllVideos(data.videos ?? [])
    }
  }

  // Load all indexed videos on mount
  useEffect(() => {
    async function fetchAllVideos() {
      try {
        await refreshAllVideos()
      } finally {
        setLoadingAll(false)
      }
    }
    fetchAllVideos()
  }, [])

  async function handleIndexAll() {
    setIndexAllStatus('running')
    setIndexAllResult(null)
    setIndexAllError(null)
    try {
      const res = await fetch('/api/index-all', { method: 'POST' })
      const data = await res.json()
      if (!res.ok) {
        setIndexAllError(data.error ?? `Server error (${res.status})`)
        setIndexAllStatus('error')
        return
      }
      setIndexAllResult({ indexed: data.indexed, failed: data.failed, total: data.total })
      setIndexAllStatus('done')
      // Refresh the video list to show newly indexed videos
      await refreshAllVideos()
    } catch (err) {
      setIndexAllError(err instanceof Error ? err.message : 'Network error')
      setIndexAllStatus('error')
    }
  }

  async function handleSearch(e: React.FormEvent) {
    e.preventDefault()
    const q = query.trim()
    if (!q) {
      setClips([])
      setLastQuery('')
      return
    }

    setLoading(true)
    setError(null)

    try {
      const res = await fetch(`/api/search?q=${encodeURIComponent(q)}`)
      const data = await res.json()

      if (!res.ok) {
        setError(data.error ?? 'Search failed. Check your TwelveLabs configuration.')
        setClips([])
        return
      }

      setClips(data.clips ?? [])
      setLastQuery(q)
    } catch {
      setError('Network error. Please try again.')
      setClips([])
    } finally {
      setLoading(false)
    }
  }

  function handleClear() {
    setQuery('')
    setClips([])
    setLastQuery('')
    setError(null)
    inputRef.current?.focus()
  }

  const isSearching = lastQuery !== ''

  return (
    <div>
      <div className="mb-8">
        <h1 className="text-2xl font-bold text-white mb-1">Search Videos</h1>
        <p className="text-sm text-[var(--foreground-secondary)]">
          Find specific moments across all indexed submissions using natural language.
        </p>
      </div>

      {/* Search bar */}
      <form onSubmit={handleSearch} className="flex gap-3 mb-8">
        <div className="relative flex-1">
          <input
            ref={inputRef}
            type="text"
            className="input-dark w-full text-base pr-8"
            placeholder='e.g. "bottle on a table" or "person walking outside"'
            value={query}
            onChange={e => setQuery(e.target.value)}
            disabled={loading}
            autoFocus
          />
          {query && (
            <button
              type="button"
              onClick={handleClear}
              className="absolute right-2 top-1/2 -translate-y-1/2 text-[var(--foreground-secondary)] hover:text-white transition-colors text-lg leading-none"
              aria-label="Clear search"
            >
              ×
            </button>
          )}
        </div>
        <button
          type="submit"
          disabled={loading || !query.trim()}
          className="btn-lab rounded-lg px-5 py-2 text-sm font-medium transition-colors disabled:opacity-40 shrink-0"
        >
          {loading ? (
            <span className="flex items-center gap-2">
              <span className="h-3 w-3 rounded-full border-2 border-white/30 border-t-white animate-spin" />
              Searching
            </span>
          ) : 'Search'}
        </button>
      </form>

      {/* Error */}
      {error && (
        <div className="mb-6 rounded-lg border border-red-500/30 bg-red-500/10 p-4 text-sm text-red-400">
          {error}
        </div>
      )}

      {/* Search results */}
      {isSearching && !loading && (
        <>
          <div className="flex items-center justify-between mb-4">
            <p className="text-sm text-[var(--foreground-secondary)]">
              <span className="text-white font-medium">{clips.length}</span> result{clips.length !== 1 ? 's' : ''}{' '}
              for &ldquo;{lastQuery}&rdquo;
            </p>
            <button
              onClick={handleClear}
              className="text-xs text-[var(--foreground-secondary)] hover:text-white transition-colors"
            >
              ← Show all videos
            </button>
          </div>

          {clips.length === 0 ? (
            <div className="surface-panel py-16 text-center">
              <div className="text-3xl mb-3">🔍</div>
              <p className="font-medium text-white">No results found</p>
              <p className="mt-1 text-sm text-[var(--foreground-secondary)]">
                Try different search terms, or make sure your videos are indexed for search.
              </p>
            </div>
          ) : (
            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
              {clips.map((clip, i) => (
                <ClipCard key={`${clip.video_id}-${i}`} clip={clip} />
              ))}
            </div>
          )}
        </>
      )}

      {/* All videos (default state) */}
      {!isSearching && !loading && (
        <>
          {loadingAll ? (
            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
              {[...Array(6)].map((_, i) => (
                <div key={i} className="surface-panel overflow-hidden animate-pulse">
                  <div className="w-full aspect-video bg-[var(--surface-muted)]" />
                  <div className="p-3 flex flex-col gap-2">
                    <div className="h-3 bg-[var(--surface-muted)] rounded w-3/4" />
                    <div className="h-3 bg-[var(--surface-muted)] rounded w-1/2" />
                  </div>
                </div>
              ))}
            </div>
          ) : allVideos.length === 0 ? (
            <div className="surface-panel py-20 text-center">
              <div className="text-4xl mb-3">🎬</div>
              <p className="font-medium text-white">No indexed videos yet</p>
              <p className="mt-2 text-sm text-[var(--foreground-secondary)] max-w-sm mx-auto">
                Index all existing submissions at once, or go to a task and click &quot;Index for Search&quot; on individual videos.
              </p>
              <button
                onClick={handleIndexAll}
                disabled={indexAllStatus === 'running'}
                className="mt-5 btn-lab rounded-lg px-4 py-2 text-sm font-medium transition-colors disabled:opacity-40 flex items-center gap-2 mx-auto"
              >
                {indexAllStatus === 'running' ? (
                  <>
                    <span className="h-3 w-3 rounded-full border-2 border-white/30 border-t-white animate-spin" />
                    Indexing all videos…
                  </>
                ) : 'Index all videos'}
              </button>
              {indexAllStatus === 'done' && indexAllResult && (
                <p className="mt-3 text-xs text-[#99ddaa]">
                  Done — indexed {indexAllResult.indexed}/{indexAllResult.total} videos
                  {indexAllResult.failed > 0 ? `, ${indexAllResult.failed} failed` : ''}
                </p>
              )}
            </div>
          ) : (
            <>
              <div className="flex items-center justify-between mb-4">
                <p className="text-sm text-[var(--foreground-secondary)]">
                  <span className="text-white font-medium">{allVideos.length}</span> indexed video{allVideos.length !== 1 ? 's' : ''}
                </p>
                <div className="flex items-center gap-3">
                  {indexAllStatus === 'done' && indexAllResult && (
                    <span className="text-xs text-[#99ddaa]">
                      Indexed {indexAllResult.indexed}/{indexAllResult.total}
                      {indexAllResult.failed > 0 ? `, ${indexAllResult.failed} failed` : ''}
                    </span>
                  )}
                  {indexAllStatus === 'error' && (
                    <span className="text-xs text-[#f3a8a8]" title={indexAllError ?? undefined}>
                      {indexAllError ?? 'Index all failed'}
                    </span>
                  )}
                  <button
                    onClick={handleIndexAll}
                    disabled={indexAllStatus === 'running'}
                    className="btn-neutral rounded-lg px-3 py-1.5 text-xs transition-colors disabled:opacity-40 flex items-center gap-1.5"
                  >
                    {indexAllStatus === 'running' ? (
                      <>
                        <span className="h-2.5 w-2.5 rounded-full border-2 border-[var(--foreground-secondary)]/30 border-t-[var(--foreground-secondary)] animate-spin" />
                        Indexing…
                      </>
                    ) : 'Index all unindexed'}
                  </button>
                </div>
              </div>
              <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
                {allVideos.map((video) => (
                  <VideoCard key={video.submissionId} video={video} />
                ))}
              </div>
            </>
          )}
        </>
      )}
    </div>
  )
}

function formatTime(s: number): string {
  const m = Math.floor(s / 60)
  const sec = Math.floor(s % 60)
  return m > 0 ? `${m}:${sec.toString().padStart(2, '0')}` : `${sec}s`
}

function VideoCard({ video }: { video: IndexedVideo }) {
  return (
    <div className="surface-panel overflow-hidden flex flex-col">
      {video.signedUrl ? (
        <video
          src={video.signedUrl}
          controls
          className="w-full aspect-video bg-black object-contain"
          preload="metadata"
        />
      ) : (
        <div className="w-full aspect-video bg-[var(--surface-muted)] flex flex-col items-center justify-center gap-1">
          <span className="text-2xl">🎥</span>
          <span className="text-xs text-[var(--foreground-secondary)]">Video unavailable</span>
        </div>
      )}

      <div className="p-3 flex flex-col gap-1 flex-1">
        {video.taskTitle && (
          <p className="text-xs font-medium text-white truncate">{video.taskTitle}</p>
        )}
        {video.taskId ? (
          <Link
            href={`/lab/tasks/${video.taskId}`}
            className="mt-auto text-xs text-[var(--foreground-secondary)] hover:text-white transition-colors"
          >
            View submission →
          </Link>
        ) : (
          <p className="mt-auto text-xs text-[var(--foreground-secondary)]">
            ID: {video.submissionId.slice(0, 12)}…
          </p>
        )}
      </div>
    </div>
  )
}

function ClipCard({ clip }: { clip: SearchClip }) {
  const confidenceStyle =
    clip.confidence === 'high'
      ? 'bg-[rgba(47,158,68,0.16)] text-[#99ddaa]'
      : clip.confidence === 'medium'
      ? 'bg-[rgba(216,163,71,0.16)] text-[#f0cb7c]'
      : 'bg-[rgba(115,120,131,0.18)] text-[#d3d7de]'

  return (
    <div className="surface-panel overflow-hidden flex flex-col">
      {clip.signedUrl ? (
        <video
          src={`${clip.signedUrl}#t=${Math.floor(clip.start)}`}
          controls
          className="w-full aspect-video bg-black object-contain"
          preload="metadata"
        />
      ) : clip.thumbnail_url ? (
        // eslint-disable-next-line @next/next/no-img-element
        <img
          src={clip.thumbnail_url}
          alt="Video clip thumbnail"
          className="w-full aspect-video object-cover"
        />
      ) : (
        <div className="w-full aspect-video bg-[var(--surface-muted)] flex flex-col items-center justify-center gap-1">
          <span className="text-2xl">🎥</span>
          <span className="text-xs text-[var(--foreground-secondary)]">Not yet indexed in your library</span>
        </div>
      )}

      <div className="p-3 flex flex-col gap-2 flex-1">
        <div className="flex items-center justify-between">
          <span className="text-xs font-semibold text-[#aebeff]">Rank #{clip.rank}</span>
          <span className="text-xs font-mono text-[var(--foreground-secondary)]">
            {formatTime(clip.start)} – {formatTime(clip.end)}
          </span>
        </div>

        <div className="flex items-center gap-2 flex-wrap">
          {clip.confidence && (
            <span className={`text-[10px] px-1.5 py-0.5 rounded-full font-medium ${confidenceStyle}`}>
              {clip.confidence}
            </span>
          )}
          {clip.score != null && (
            <span className="text-[10px] text-[var(--foreground-secondary)]">
              score {clip.score.toFixed(1)}
            </span>
          )}
        </div>

        {clip.taskId ? (
          <Link
            href={`/lab/tasks/${clip.taskId}`}
            className="mt-auto text-xs text-[var(--foreground-secondary)] hover:text-white transition-colors"
          >
            View submission →
          </Link>
        ) : (
          <p className="mt-auto text-xs text-[var(--foreground-secondary)]">
            Video ID: {clip.video_id.slice(0, 12)}…
          </p>
        )}
      </div>
    </div>
  )
}
