'use client'

import { useState, useTransition } from 'react'
import { updateTaskMeta } from './actions'

interface Props {
  taskId: string
  initialTitle: string
  initialDescription: string | null
  status: string
}

export default function EditableTaskHeader({ taskId, initialTitle, initialDescription, status }: Props) {
  const [editing, setEditing] = useState(false)
  const [title, setTitle] = useState(initialTitle)
  const [description, setDescription] = useState(initialDescription ?? '')
  const [draftTitle, setDraftTitle] = useState(initialTitle)
  const [draftDescription, setDraftDescription] = useState(initialDescription ?? '')
  const [error, setError] = useState<string | null>(null)
  const [isPending, startTransition] = useTransition()

  function startEdit() {
    setDraftTitle(title)
    setDraftDescription(description)
    setError(null)
    setEditing(true)
  }

  function cancel() {
    setEditing(false)
    setError(null)
  }

  function save() {
    startTransition(async () => {
      const result = await updateTaskMeta(taskId, draftTitle, draftDescription)
      if (result.error) {
        setError(result.error)
        return
      }
      setTitle(draftTitle.trim())
      setDescription(draftDescription.trim())
      setEditing(false)
    })
  }

  if (editing) {
    return (
      <div className="flex-1">
        <input
          type="text"
          value={draftTitle}
          onChange={(e) => setDraftTitle(e.target.value)}
          className="w-full rounded-md border border-[var(--border)] bg-[var(--surface-muted)] px-3 py-1.5 text-xl font-bold text-white outline-none focus:border-[#3b5bdb] mb-2"
          placeholder="Task title"
          disabled={isPending}
          autoFocus
        />
        <textarea
          value={draftDescription}
          onChange={(e) => setDraftDescription(e.target.value)}
          rows={3}
          className="w-full rounded-md border border-[var(--border)] bg-[var(--surface-muted)] px-3 py-1.5 text-sm text-[var(--foreground-secondary)] outline-none focus:border-[#3b5bdb] resize-none"
          placeholder="Task description (optional)"
          disabled={isPending}
        />
        {error && <p className="mt-1 text-xs text-red-400">{error}</p>}
        <div className="flex gap-2 mt-2">
          <button
            onClick={save}
            disabled={isPending || !draftTitle.trim()}
            className="rounded-md bg-[#3b5bdb] px-3 py-1 text-xs font-medium text-white hover:bg-[#4c6ef5] disabled:opacity-50 transition-colors"
          >
            {isPending ? 'Saving…' : 'Save'}
          </button>
          <button
            onClick={cancel}
            disabled={isPending}
            className="rounded-md border border-[var(--border)] px-3 py-1 text-xs font-medium text-[var(--foreground-secondary)] hover:text-white transition-colors"
          >
            Cancel
          </button>
        </div>
      </div>
    )
  }

  const statusBadge = (
    <span className={`rounded-full px-2 py-0.5 text-xs font-medium ${
      status === 'open' ? 'bg-[rgba(59,91,219,0.16)] text-[#aebeff]' : 'bg-[rgba(115,120,131,0.18)] text-[#d3d7de]'
    }`}>
      {status}
    </span>
  )

  return (
    <div className="flex-1 group/header">
      <div className="flex items-center gap-2 mb-1">
        <h1 className="text-xl font-bold text-white">{title}</h1>
        {statusBadge}
        <button
          onClick={startEdit}
          title="Edit title and description"
          className="rounded p-0.5 text-[var(--foreground-secondary)] opacity-0 group-hover/header:opacity-100 hover:text-white transition-opacity"
        >
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <path d="M11 4H4a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2v-7"/>
            <path d="M18.5 2.5a2.121 2.121 0 0 1 3 3L12 15l-4 1 1-4 9.5-9.5z"/>
          </svg>
        </button>
      </div>
      {description && (
        <p className="mt-1 text-sm text-[var(--foreground-secondary)]">{description}</p>
      )}
    </div>
  )
}
