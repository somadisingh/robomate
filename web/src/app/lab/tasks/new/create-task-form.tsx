'use client'

import { useEffect, useId, useState } from 'react'

const REQUIREMENTS = [
  { value: 'outdoor', label: 'Outdoor', className: 'tag-outdoor' },
  { value: 'indoor', label: 'Indoor', className: 'tag-indoor' },
  { value: 'motion', label: 'Motion', className: 'tag-motion' },
  { value: 'monochrome', label: 'Monochrome', className: 'tag-monochrome' },
] as const

type Requirement = (typeof REQUIREMENTS)[number]

type PreviewAsset = {
  name: string
  type: string
  url: string
}

export default function CreateTaskForm({
  action,
}: {
  action: (formData: FormData) => void
}) {
  const uploadId = useId()
  const [title, setTitle] = useState('')
  const [description, setDescription] = useState('')
  const [bounty, setBounty] = useState('5')
  const [quantity, setQuantity] = useState(24)
  const [deadline, setDeadline] = useState('')
  const [requirements, setRequirements] = useState<string[]>([])
  const [objects, setObjects] = useState('')
  const [assets, setAssets] = useState<PreviewAsset[]>([])

  useEffect(() => {
    return () => {
      assets.forEach((asset) => URL.revokeObjectURL(asset.url))
    }
  }, [assets])

  function toggleRequirement(value: Requirement['value']) {
    setRequirements((current) =>
      current.includes(value)
        ? current.filter((item) => item !== value)
        : [...current, value]
    )
  }

  function handleFiles(files: FileList | null) {
    if (!files) return

    setAssets((current) => {
      current.forEach((asset) => URL.revokeObjectURL(asset.url))
      return Array.from(files).map((file) => ({
        name: file.name,
        type: file.type,
        url: URL.createObjectURL(file),
      }))
    })
  }

  const bountyValue = Number.parseFloat(bounty || '0')
  const estimatedSpend = Number.isFinite(bountyValue) ? bountyValue * quantity : 0
  const filledChecks = [
    title.trim().length > 0,
    description.trim().length > 0,
    assets.length > 0,
    requirements.length > 0,
    bountyValue > 0,
    deadline.length > 0,
  ].filter(Boolean).length

  return (
    <div className="space-y-8">
      <div className="flex flex-wrap items-center justify-between gap-4">
        <div>
          <div className="mb-3 flex items-center gap-3">
            <span className="role-pill-lab rounded-full px-2.5 py-1 text-[11px] font-semibold uppercase tracking-[0.14em]">
              Lab
            </span>
            <span className="text-xs uppercase tracking-[0.18em] text-[var(--foreground-secondary)]">
              Create Task
            </span>
          </div>
          <h1 className="text-3xl font-black tracking-[-0.03em] text-white sm:text-4xl">
            Build a collection brief collectors can execute fast
          </h1>
          <p className="mt-3 max-w-2xl text-sm leading-6 text-[var(--foreground-secondary)] sm:text-base">
            Define the task, attach reference context, and price the work so the collector sees exactly what to capture.
          </p>
        </div>
      </div>

      <div className="grid gap-8 xl:grid-cols-[minmax(0,1.55fr)_360px]">
        <form id="create-task-form" action={action} className="space-y-6">
          <section className="surface-panel p-5 sm:p-6">
            <div className="mb-4">
              <h2 className="text-sm font-semibold uppercase tracking-[0.16em] text-white">Task brief</h2>
              <p className="mt-1 text-sm text-[var(--foreground-secondary)]">
                Anchor the assignment with a clear title and capture instructions.
              </p>
            </div>
            <div className="space-y-4">
              <div>
                <label className="mb-1.5 block text-sm font-medium text-white">Task title</label>
                <input
                  name="title"
                  required
                  value={title}
                  onChange={(event) => setTitle(event.target.value)}
                  placeholder="Record 30-second walking clips through busy intersections"
                  className="input-dark text-sm"
                />
              </div>
              <div>
                <label className="mb-1.5 block text-sm font-medium text-white">Description</label>
                <textarea
                  name="description"
                  required
                  rows={5}
                  value={description}
                  onChange={(event) => setDescription(event.target.value)}
                  placeholder="Describe the framing, movement, duration, and any environment constraints collectors must follow."
                  className="input-dark resize-none py-3 text-sm"
                />
              </div>
            </div>
          </section>

          <section className="surface-panel p-5 sm:p-6">
            <div className="mb-4">
              <h2 className="text-sm font-semibold uppercase tracking-[0.16em] text-white">Reference assets</h2>
              <p className="mt-1 text-sm text-[var(--foreground-secondary)]">
                Upload example stills or clips that show the collector the expected look.
              </p>
            </div>

            <label
              htmlFor={uploadId}
              className="flex min-h-52 cursor-pointer flex-col items-center justify-center rounded-lg border border-dashed border-[var(--border)] bg-[var(--surface-muted)] px-6 py-8 text-center transition-colors hover:border-[rgba(255,255,255,0.22)]"
            >
              <span className="text-sm font-semibold text-white">Drop reference files here</span>
              <span className="mt-2 text-sm text-[var(--foreground-secondary)]">
                Images and short video clips work best
              </span>
              <span className="mt-5 rounded-lg border border-[var(--border)] px-3 py-2 text-xs font-medium uppercase tracking-[0.16em] text-[var(--foreground-secondary)]">
                Browse files
              </span>
            </label>
            <input
              id={uploadId}
              name="reference_assets"
              type="file"
              accept="image/*,video/*"
              multiple
              onChange={(event) => handleFiles(event.target.files)}
              className="sr-only"
            />

            {assets.length > 0 && (
              <div className="mt-4 grid grid-cols-2 gap-3 sm:grid-cols-4">
                {assets.map((asset) => (
                  <div key={asset.url} className="overflow-hidden rounded-lg border border-[var(--border)] bg-[var(--surface-elevated)]">
                    <div className="aspect-[4/3] bg-black/30">
                      {asset.type.startsWith('video/') ? (
                        <video src={asset.url} className="h-full w-full object-cover" muted playsInline />
                      ) : (
                        // eslint-disable-next-line @next/next/no-img-element
                        <img src={asset.url} alt={asset.name} className="h-full w-full object-cover" />
                      )}
                    </div>
                    <div className="truncate border-t border-[var(--border)] px-2 py-2 text-xs text-[var(--foreground-secondary)]">
                      {asset.name}
                    </div>
                  </div>
                ))}
              </div>
            )}
          </section>

          <section className="surface-panel p-5 sm:p-6">
            <div className="mb-4">
              <h2 className="text-sm font-semibold uppercase tracking-[0.16em] text-white">Requirements</h2>
              <p className="mt-1 text-sm text-[var(--foreground-secondary)]">
                Signal the environments or qualities the collector must satisfy.
              </p>
            </div>

            <div className="flex flex-wrap gap-3">
              {REQUIREMENTS.map((requirement) => {
                const selected = requirements.includes(requirement.value)

                return (
                  <button
                    key={requirement.value}
                    type="button"
                    onClick={() => toggleRequirement(requirement.value)}
                    className={`rounded-full border px-4 py-2 text-sm font-medium transition-colors ${
                      selected
                        ? requirement.className
                        : 'border-[var(--border)] bg-transparent text-[var(--foreground-secondary)] hover:border-[rgba(255,255,255,0.22)] hover:text-white'
                    }`}
                  >
                    {requirement.label}
                  </button>
                )
              })}
            </div>

            {requirements.map((requirement) => (
              <input key={requirement} type="hidden" name="requirements" value={requirement} />
            ))}
          </section>

          <section className="surface-panel p-5 sm:p-6">
            <div className="mb-4">
              <h2 className="text-sm font-semibold uppercase tracking-[0.16em] text-white">Objects to segment</h2>
              <p className="mt-1 text-sm text-[var(--foreground-secondary)]">
                Comma-separated object tags for SAM. Human hand is always included automatically.
              </p>
            </div>
            <input
              name="objects"
              value={objects}
              onChange={(event) => setObjects(event.target.value)}
              placeholder="cup, can, spoon"
              className="input-dark text-sm"
            />
          </section>

          <section className="surface-panel p-5 sm:p-6">
            <div className="mb-4">
              <h2 className="text-sm font-semibold uppercase tracking-[0.16em] text-white">Payout</h2>
              <p className="mt-1 text-sm text-[var(--foreground-secondary)]">
                Price each accepted submission and define how many are needed.
              </p>
            </div>

            <div className="grid gap-4 md:grid-cols-[minmax(0,1fr)_220px]">
              <div>
                <label className="mb-1.5 block text-sm font-medium text-white">Bounty per submission</label>
                <div className="flex overflow-hidden rounded-lg border border-[var(--border)] bg-[var(--surface-elevated)]">
                  <span className="flex items-center border-r border-[var(--border)] px-4 text-sm font-medium text-[var(--foreground-secondary)]">
                    $
                  </span>
                  <input
                    name="bounty_amount"
                    type="number"
                    min="0.01"
                    step="0.01"
                    required
                    value={bounty}
                    onChange={(event) => setBounty(event.target.value)}
                    className="w-full bg-transparent px-4 py-3 text-sm text-white outline-none"
                  />
                </div>
              </div>

              <div>
                <label className="mb-1.5 block text-sm font-medium text-white">Submissions required</label>
                <div className="flex overflow-hidden rounded-lg border border-[var(--border)] bg-[var(--surface-elevated)]">
                  <button
                    type="button"
                    onClick={() => setQuantity((current) => Math.max(1, current - 1))}
                    className="w-11 border-r border-[var(--border)] text-lg text-[var(--foreground-secondary)] transition-colors hover:text-white"
                  >
                    −
                  </button>
                  <input
                    name="quantity_needed"
                    type="number"
                    min="1"
                    required
                    value={quantity}
                    onChange={(event) => setQuantity(Math.max(1, Number.parseInt(event.target.value || '1', 10)))}
                    className="w-full bg-transparent px-4 py-3 text-center text-sm text-white outline-none"
                  />
                  <button
                    type="button"
                    onClick={() => setQuantity((current) => current + 1)}
                    className="w-11 border-l border-[var(--border)] text-lg text-[var(--foreground-secondary)] transition-colors hover:text-white"
                  >
                    +
                  </button>
                </div>
              </div>
            </div>
          </section>

          <section className="surface-panel p-5 sm:p-6">
            <div className="mb-4">
              <h2 className="text-sm font-semibold uppercase tracking-[0.16em] text-white">Deadline</h2>
              <p className="mt-1 text-sm text-[var(--foreground-secondary)]">
                Set the final collection cutoff for this task.
              </p>
            </div>
            <input
              name="deadline"
              type="datetime-local"
              value={deadline}
              onChange={(event) => setDeadline(event.target.value)}
              className="input-dark text-sm"
            />
          </section>
        </form>

        <aside className="xl:sticky xl:top-8 xl:self-start">
          <div className="surface-panel p-5 sm:p-6">
            <div className="border-b border-[var(--border)] pb-5">
              <p className="text-xs font-semibold uppercase tracking-[0.16em] text-[var(--foreground-secondary)]">
                Estimated total spend
              </p>
              <div className="mt-3 text-4xl font-black tracking-[-0.04em] text-[#aebeff]">
                ${estimatedSpend.toFixed(2)}
              </div>
              <p className="mt-2 text-sm text-[var(--foreground-secondary)]">
                {quantity} submissions at ${bountyValue.toFixed(2) || '0.00'} each
              </p>
            </div>

            <div className="space-y-5 py-5">
              <div>
                <p className="mb-3 text-xs font-semibold uppercase tracking-[0.16em] text-[var(--foreground-secondary)]">
                  Selected requirements
                </p>
                <div className="flex flex-wrap gap-2">
                  {requirements.length > 0 ? (
                    REQUIREMENTS.filter((item) => requirements.includes(item.value)).map((item) => (
                      <span key={item.value} className={`rounded-full border px-2.5 py-1 text-xs font-medium ${item.className}`}>
                        {item.label}
                      </span>
                    ))
                  ) : (
                    <span className="text-sm text-[var(--foreground-secondary)]">No tags selected yet</span>
                  )}
                </div>
              </div>

              <div>
                <p className="mb-3 text-xs font-semibold uppercase tracking-[0.16em] text-[var(--foreground-secondary)]">
                  Task completeness
                </p>
                <ul className="space-y-2 text-sm">
                  {[
                    { label: 'Task title', done: title.trim().length > 0 },
                    { label: 'Description', done: description.trim().length > 0 },
                    { label: 'Reference assets', done: assets.length > 0 },
                    { label: 'Requirement tags', done: requirements.length > 0 },
                    { label: 'Pricing', done: bountyValue > 0 },
                    { label: 'Deadline', done: deadline.length > 0 },
                  ].map((item) => (
                    <li key={item.label} className="flex items-center justify-between gap-3">
                      <span className={item.done ? 'text-white' : 'text-[var(--foreground-secondary)]'}>
                        {item.label}
                      </span>
                      <span className={`text-xs font-semibold uppercase tracking-[0.12em] ${item.done ? 'text-[#99ddaa]' : 'text-[var(--foreground-secondary)]'}`}>
                        {item.done ? 'Ready' : 'Missing'}
                      </span>
                    </li>
                  ))}
                </ul>
                <p className="mt-3 text-xs text-[var(--foreground-secondary)]">
                  {filledChecks} of 6 pieces are in place.
                </p>
              </div>

              <div className="border-t border-[var(--border)] pt-5">
                <p className="mb-3 text-xs font-semibold uppercase tracking-[0.16em] text-[var(--foreground-secondary)]">
                  Collector preview
                </p>
                <div className="space-y-3">
                  <div className="flex items-start justify-between gap-3">
                    <div>
                      <h3 className="text-base font-semibold text-white">
                        {title.trim() || 'Untitled task'}
                      </h3>
                      <p className="mt-1 text-sm text-[var(--foreground-secondary)]">
                        {description.trim() || 'Collectors will see your capture instructions here.'}
                      </p>
                    </div>
                    <div className="text-right">
                      <div className="text-2xl font-black tracking-[-0.04em] text-[#8ad09a]">
                        ${bountyValue.toFixed(2)}
                      </div>
                      <div className="text-xs text-[var(--foreground-secondary)]">per submission</div>
                    </div>
                  </div>
                  <div className="flex flex-wrap gap-2">
                    {requirements.length > 0 ? (
                      REQUIREMENTS.filter((item) => requirements.includes(item.value)).map((item) => (
                        <span key={item.value} className={`rounded-full border px-2.5 py-1 text-xs font-medium ${item.className}`}>
                          {item.label}
                        </span>
                      ))
                    ) : (
                      <span className="text-xs text-[var(--foreground-secondary)]">Tags appear here once selected.</span>
                    )}
                  </div>
                  <div className="flex items-center justify-between text-xs text-[var(--foreground-secondary)]">
                    <span>{quantity} needed</span>
                    <span>{deadline ? new Date(deadline).toLocaleString() : 'No deadline yet'}</span>
                  </div>
                </div>
              </div>
            </div>

            <button
              form="create-task-form"
              type="submit"
              className="btn-lab mt-2 w-full rounded-lg py-3 text-sm font-semibold transition-colors"
            >
              Post task
            </button>
          </div>
        </aside>
      </div>
    </div>
  )
}
