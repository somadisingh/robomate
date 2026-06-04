'use client'

import { useEffect, useRef, useState } from 'react'

const MAX_PREVIEW_BYTES = 200_000
const MAX_PREVIEW_LINES = 2000

type FetchState =
  | { status: 'idle' }
  | { status: 'loading' }
  | { status: 'error'; message: string }
  | { status: 'ready'; body: string; truncated: boolean }

export default function ArtifactPreview({
  signedUrl,
  kind,
}: {
  signedUrl: string
  kind: string
}) {
  const [open, setOpen] = useState(false)
  const [state, setState] = useState<FetchState>({ status: 'idle' })
  const abortRef = useRef<AbortController | null>(null)

  useEffect(() => {
    return () => {
      abortRef.current?.abort()
    }
  }, [])

  async function runFetch() {
    abortRef.current?.abort()
    const controller = new AbortController()
    abortRef.current = controller

    setState({ status: 'loading' })
    try {
      const response = await fetch(signedUrl, { signal: controller.signal })
      if (controller.signal.aborted) return
      if (!response.ok) {
        setState({ status: 'error', message: `HTTP ${response.status}` })
        return
      }
      const json = await response.json()
      if (controller.signal.aborted) return
      const full = JSON.stringify(json, null, 2)
      let truncated = false
      let body = full
      if (body.length > MAX_PREVIEW_BYTES) {
        body = body.slice(0, MAX_PREVIEW_BYTES)
        truncated = true
      }
      const lines = body.split('\n')
      if (lines.length > MAX_PREVIEW_LINES) {
        body = lines.slice(0, MAX_PREVIEW_LINES).join('\n')
        truncated = true
      }
      setState({ status: 'ready', body, truncated })
    } catch (error) {
      if (controller.signal.aborted) return
      if (error instanceof DOMException && error.name === 'AbortError') return
      const message = error instanceof Error ? error.message : 'Failed to fetch artifact'
      setState({ status: 'error', message })
    }
  }

  async function handleToggle() {
    const next = !open
    setOpen(next)
    if (next) {
      if (state.status === 'idle') {
        await runFetch()
      }
    } else {
      abortRef.current?.abort()
    }
  }

  async function handleRetry() {
    await runFetch()
  }

  return (
    <div className="mt-2">
      <button
        type="button"
        onClick={handleToggle}
        className="text-xs font-medium text-[#aebeff] hover:text-white transition-colors"
      >
        {open ? '▼' : '▶'} Preview JSON
      </button>
      {open && (
        <div className="mt-2 rounded-lg border border-[var(--border)] bg-[var(--surface-muted)] p-3 text-xs">
          {state.status === 'loading' && (
            <p className="text-[var(--foreground-secondary)]">Loading {kind} artifact…</p>
          )}
          {state.status === 'error' && (
            <div className="flex flex-wrap items-center gap-3">
              <p className="text-[#f3a8a8]">Failed to load: {state.message}</p>
              <button
                type="button"
                onClick={handleRetry}
                className="text-xs font-medium text-[#aebeff] hover:text-white transition-colors"
              >
                Retry
              </button>
            </div>
          )}
          {state.status === 'ready' && (
            <>
              {state.truncated && (
                <p className="mb-2 text-[11px] text-[var(--foreground-secondary)]">
                  Truncated — open Download JSON for full file
                </p>
              )}
              <pre className="max-h-[60vh] overflow-auto whitespace-pre-wrap break-words text-[11px] leading-5 text-[#d3d7de]">
                {state.body}
              </pre>
            </>
          )}
        </div>
      )}
    </div>
  )
}
