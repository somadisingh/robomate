import { createClient } from '@/lib/supabase/server'
import { notFound } from 'next/navigation'
import Link from 'next/link'
import CopyButton from './copy-button'

type ReferenceAsset = {
  url: string
  type?: string
  label?: string
}

const REQUIREMENT_STYLES: Record<string, string> = {
  outdoor: 'tag-outdoor',
  indoor: 'tag-indoor',
  motion: 'tag-motion',
  monochrome: 'tag-monochrome',
}

const STATUS_STYLES: Record<string, string> = {
  open: 'bg-[rgba(59,91,219,0.16)] text-[#aebeff]',
  submitted: 'bg-[rgba(47,158,68,0.16)] text-[#99ddaa]',
  'under review': 'bg-[rgba(216,163,71,0.16)] text-[#f0cb7c]',
  approved: 'bg-[rgba(47,158,68,0.2)] text-[#b4f0c1]',
  rejected: 'bg-[rgba(210,100,100,0.16)] text-[#f3a8a8]',
  full: 'bg-[rgba(115,120,131,0.18)] text-[#d3d7de]',
}

function formatRequirement(value: string) {
  return value.charAt(0).toUpperCase() + value.slice(1)
}

export default async function CollectorTaskPage({ params }: { params: Promise<{ id: string }> }) {
  const { id } = await params
  const supabase = await createClient()
  const { data: { user } } = await supabase.auth.getUser()

  const { data: task, error } = await supabase
    .from('tasks')
    .select('*')
    .eq('id', id)
    .single()

  if (error || !task) notFound()

  const { data: existingSubmission } = await supabase
    .from('submissions')
    .select('id, status, created_at')
    .eq('task_id', id)
    .eq('collector_id', user!.id)
    .single()

  const spotsLeft = task.quantity_needed - task.quantity_filled
  const deepLink = `aperture://task/${id}`
  const requirements = ((task.required_capabilities as string[] | null) ?? []).filter(Boolean)
  const taskWithMedia = task as typeof task & {
    metadata?: { reference_assets?: ReferenceAsset[] }
    reference_assets?: ReferenceAsset[]
  }
  const referenceAssets: ReferenceAsset[] =
    taskWithMedia.reference_assets ??
    taskWithMedia.metadata?.reference_assets ??
    []

  let statusLabel = 'open'
  if (existingSubmission?.status === 'pending') statusLabel = 'under review'
  if (existingSubmission?.status === 'approved') statusLabel = 'approved'
  if (existingSubmission?.status === 'rejected') statusLabel = 'rejected'
  if (!existingSubmission && spotsLeft <= 0) statusLabel = 'full'

  return (
    <div className="space-y-6">
      <Link
        href="/collector/tasks"
        className="inline-block text-sm text-[var(--foreground-secondary)] transition-colors hover:text-white"
      >
        ← Browse tasks
      </Link>

      <section className="surface-panel p-6 sm:p-7">
        <div className="flex flex-col gap-6 lg:flex-row lg:items-start lg:justify-between">
          <div className="max-w-3xl">
            <div className="mb-4 flex flex-wrap items-center gap-3">
              <span className={`rounded-full px-2.5 py-1 text-xs font-semibold uppercase tracking-[0.14em] ${STATUS_STYLES[statusLabel]}`}>
                {statusLabel}
              </span>
              <span className="text-xs uppercase tracking-[0.16em] text-[var(--foreground-secondary)]">
                {spotsLeft > 0 ? `${spotsLeft} spots left` : 'No spots left'}
              </span>
              {task.deadline && (
                <span className="text-xs uppercase tracking-[0.16em] text-[var(--foreground-secondary)]">
                  Due {new Date(task.deadline).toLocaleDateString()}
                </span>
              )}
            </div>

            <h1 className="text-3xl font-black tracking-[-0.04em] text-white sm:text-4xl">
              {task.title}
            </h1>
            <p className="mt-4 max-w-2xl text-base leading-7 text-[var(--foreground-secondary)] sm:text-lg">
              {task.description || 'Open the task in the iPhone app to follow the capture brief and upload your submission.'}
            </p>
          </div>

          <div className="shrink-0">
            <div className="text-5xl font-black tracking-[-0.05em] text-[#8ad09a]">
              ${task.bounty_amount}
            </div>
            <div className="mt-2 text-sm uppercase tracking-[0.16em] text-[var(--foreground-secondary)]">
              Per submission
            </div>
          </div>
        </div>
      </section>

      <div className="grid gap-6 xl:grid-cols-[minmax(0,1.45fr)_320px]">
        <div className="space-y-6">
          <section className="surface-panel p-6">
            <h2 className="text-sm font-semibold uppercase tracking-[0.16em] text-white">Task description</h2>
            <p className="mt-4 text-base leading-7 text-[var(--foreground-secondary)]">
              {task.description || 'No additional description was provided for this task.'}
            </p>
          </section>

          <section className="surface-panel p-6">
            <div className="flex items-center justify-between gap-4">
              <h2 className="text-sm font-semibold uppercase tracking-[0.16em] text-white">Reference media</h2>
              <span className="text-xs uppercase tracking-[0.16em] text-[var(--foreground-secondary)]">
                {referenceAssets.length} asset{referenceAssets.length === 1 ? '' : 's'}
              </span>
            </div>

            {referenceAssets.length > 0 ? (
              <div className="mt-4 space-y-3">
                <div className="overflow-hidden rounded-lg border border-[var(--border)] bg-[var(--surface-muted)]">
                  <div className="aspect-[16/9] bg-black/20">
                    {referenceAssets[0]?.type?.startsWith('video') ? (
                      <video src={referenceAssets[0].url} controls className="h-full w-full object-cover" />
                    ) : (
                      // eslint-disable-next-line @next/next/no-img-element
                      <img src={referenceAssets[0].url} alt={referenceAssets[0].label ?? task.title} className="h-full w-full object-cover" />
                    )}
                  </div>
                </div>
                {referenceAssets.length > 1 && (
                  <div className="grid grid-cols-3 gap-3 sm:grid-cols-4">
                    {referenceAssets.slice(1).map((asset: ReferenceAsset) => (
                      <div key={asset.url} className="overflow-hidden rounded-lg border border-[var(--border)] bg-[var(--surface-muted)]">
                        <div className="aspect-[4/3]">
                          {asset.type?.startsWith('video') ? (
                            <video src={asset.url} className="h-full w-full object-cover" muted playsInline />
                          ) : (
                            // eslint-disable-next-line @next/next/no-img-element
                            <img src={asset.url} alt={asset.label ?? task.title} className="h-full w-full object-cover" />
                          )}
                        </div>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            ) : (
              <div className="mt-4 rounded-lg border border-dashed border-[var(--border)] bg-[var(--surface-muted)] px-4 py-10 text-center text-sm text-[var(--foreground-secondary)]">
                No reference image or video has been attached to this task yet.
              </div>
            )}
          </section>

          <section className="surface-panel p-6">
            <h2 className="text-sm font-semibold uppercase tracking-[0.16em] text-white">Requirements</h2>
            <div className="mt-4 flex flex-wrap gap-2">
              {requirements.length > 0 ? (
                requirements.map((requirement) => (
                  <span
                    key={requirement}
                    className={`rounded-full border px-3 py-1.5 text-sm font-medium ${REQUIREMENT_STYLES[requirement] ?? 'border-[var(--border)] text-[var(--foreground-secondary)]'}`}
                  >
                    {formatRequirement(requirement)}
                  </span>
                ))
              ) : (
                <span className="text-sm text-[var(--foreground-secondary)]">No additional requirement tags.</span>
              )}
            </div>
          </section>

          <section className="surface-panel p-6">
            <h2 className="text-sm font-semibold uppercase tracking-[0.16em] text-white">Secondary metadata</h2>
            <dl className="mt-4 grid gap-4 sm:grid-cols-3">
              <div>
                <dt className="text-xs uppercase tracking-[0.16em] text-[var(--foreground-secondary)]">Task type</dt>
                <dd className="mt-2 text-sm font-medium text-white">{task.data_type}</dd>
              </div>
              <div>
                <dt className="text-xs uppercase tracking-[0.16em] text-[var(--foreground-secondary)]">Submission target</dt>
                <dd className="mt-2 text-sm font-medium text-white">{task.quantity_needed}</dd>
              </div>
              <div>
                <dt className="text-xs uppercase tracking-[0.16em] text-[var(--foreground-secondary)]">Task ID</dt>
                <dd className="mt-2 font-mono text-sm text-white">{id}</dd>
              </div>
            </dl>
          </section>
        </div>

        <aside className="xl:sticky xl:top-8 xl:self-start">
          <div className="surface-panel p-6">
            {existingSubmission ? (
              <div className="space-y-5">
                <div>
                  <div className={`inline-flex rounded-full px-2.5 py-1 text-xs font-semibold uppercase tracking-[0.14em] ${STATUS_STYLES[statusLabel]}`}>
                    {existingSubmission.status === 'pending' ? 'Under review' : existingSubmission.status}
                  </div>
                  <h2 className="mt-4 text-xl font-bold text-white">
                    {existingSubmission.status === 'approved'
                      ? 'Submission approved'
                      : existingSubmission.status === 'rejected'
                        ? 'Submission rejected'
                        : 'Submission received'}
                  </h2>
                  <p className="mt-2 text-sm leading-6 text-[var(--foreground-secondary)]">
                    Submitted {new Date(existingSubmission.created_at).toLocaleDateString()} from the iPhone app.
                  </p>
                </div>
                <div className="border-t border-[var(--border)] pt-4">
                  <p className="text-xs uppercase tracking-[0.16em] text-[var(--foreground-secondary)]">
                    Payout outcome
                  </p>
                  <p className="mt-3 text-3xl font-black tracking-[-0.04em] text-[#8ad09a]">
                    {existingSubmission.status === 'approved' ? `$${task.bounty_amount}` : '$0.00'}
                  </p>
                  <p className="mt-2 text-sm text-[var(--foreground-secondary)]">
                    {existingSubmission.status === 'approved'
                      ? 'This reward has been added to your earnings.'
                      : existingSubmission.status === 'rejected'
                        ? 'Rejected submissions do not generate a payout.'
                        : 'The payout is held until review is complete.'}
                  </p>
                </div>
              </div>
            ) : spotsLeft <= 0 ? (
              <div className="space-y-4">
                <div className={`inline-flex rounded-full px-2.5 py-1 text-xs font-semibold uppercase tracking-[0.14em] ${STATUS_STYLES.full}`}>
                  Full
                </div>
                <h2 className="text-xl font-bold text-white">This task is closed</h2>
                <p className="text-sm leading-6 text-[var(--foreground-secondary)]">
                  All available spots have been filled. Browse other tasks to find the next opportunity.
                </p>
                <Link
                  href="/collector/tasks"
                  className="btn-neutral inline-flex rounded-lg px-4 py-2 text-sm font-medium transition-colors"
                >
                  Back to tasks
                </Link>
              </div>
            ) : (
              <div className="space-y-5">
                <div>
                  <div className={`inline-flex rounded-full px-2.5 py-1 text-xs font-semibold uppercase tracking-[0.14em] ${STATUS_STYLES.open}`}>
                    Open
                  </div>
                  <h2 className="mt-4 text-xl font-bold text-white">Open in iPhone app</h2>
                  <p className="mt-2 text-sm leading-6 text-[var(--foreground-secondary)]">
                    Submission happens on iOS only. Launch the Robomate app, capture the task, and your confirmation state will appear here automatically.
                  </p>
                </div>

                <a
                  href={deepLink}
                  className="btn-collector block rounded-lg py-3 text-center text-sm font-semibold transition-colors"
                >
                  Open in iPhone app
                </a>

                <div className="border-t border-[var(--border)] pt-5">
                  <p className="text-xs font-semibold uppercase tracking-[0.16em] text-[var(--foreground-secondary)]">
                    Task ID
                  </p>
                  <div className="mt-3 flex items-center gap-2">
                    <code className="min-w-0 flex-1 truncate rounded-lg border border-[var(--border)] bg-[var(--surface-elevated)] px-3 py-2 text-xs text-white">
                      {id}
                    </code>
                    <CopyButton text={id} />
                  </div>
                </div>

                <p className="text-xs leading-6 text-[var(--foreground-secondary)]">
                  If the app does not open automatically, copy the task ID into the iPhone app manually and submit from there.
                </p>
              </div>
            )}
          </div>
        </aside>
      </div>
    </div>
  )
}
