import { createClient } from '@/lib/supabase/server'
import { redirect } from 'next/navigation'
import Link from 'next/link'

export default async function CollectorTasksPage() {
  const supabase = await createClient()
  const { data: { user } } = await supabase.auth.getUser()

  // Redirect to onboarding if not complete
  const { data: profile } = await supabase
    .from('collector_profiles')
    .select('capabilities')
    .eq('user_id', user!.id)
    .single()

  if (!profile) redirect('/collector/onboard')

  // RLS filters tasks to only those matching this collector's capabilities
  const { data: tasks } = await supabase
    .from('tasks')
    .select('id, title, description, data_type, bounty_amount, quantity_needed, quantity_filled, deadline')
    .eq('status', 'open')
    .order('bounty_amount', { ascending: false })

  return (
    <div>
      <div className="mb-8">
        <h1 className="text-2xl font-bold text-white">Available tasks</h1>
        <p className="mt-1 text-sm text-[var(--foreground-secondary)]">
          Tasks matched to your capabilities. Earn money for each approved submission.
        </p>
      </div>

      {!tasks?.length ? (
        <div className="surface-panel py-20 text-center">
          <div className="text-4xl mb-3">🔍</div>
          <p className="text-[var(--foreground-secondary)]">No matching tasks right now. Check back soon.</p>
        </div>
      ) : (
        <div className="grid gap-4">
          {tasks.map(task => {
            const progress = Math.min(100, (task.quantity_filled / task.quantity_needed) * 100)
            const spotsLeft = task.quantity_needed - task.quantity_filled

            return (
              <Link
                key={task.id}
                href={`/collector/tasks/${task.id}`}
                className="surface-panel block p-5 transition-colors hover:border-[rgba(255,255,255,0.2)]"
              >
                <div className="flex items-start justify-between gap-4">
                  <div className="flex-1 min-w-0">
                    <h3 className="truncate font-semibold text-white">{task.title}</h3>
                    {task.description && (
                      <p className="mt-1 line-clamp-2 text-sm text-[var(--foreground-secondary)]">{task.description}</p>
                    )}
                    <div className="flex items-center gap-3 mt-3">
                      <span className="rounded-full border border-[var(--border)] bg-[var(--surface-muted)] px-2 py-0.5 text-xs text-[var(--foreground-secondary)]">
                        {task.data_type}
                      </span>
                      <span className="text-xs text-[var(--foreground-secondary)]">{spotsLeft} spots left</span>
                      {task.deadline && (
                        <span className="text-xs text-[var(--foreground-secondary)]">
                          Due {new Date(task.deadline).toLocaleDateString()}
                        </span>
                      )}
                    </div>
                  </div>
                  <div className="text-right shrink-0">
                    <div className="text-xl font-bold text-[#8ad09a]">${task.bounty_amount}</div>
                    <div className="text-xs text-[var(--foreground-secondary)]">per submission</div>
                  </div>
                </div>
                <div className="mt-3 h-1.5 overflow-hidden rounded-full bg-[rgba(255,255,255,0.08)]">
                  <div
                    className="h-full rounded-full bg-[#2f9e44]"
                    style={{ width: `${progress}%` }}
                  />
                </div>
              </Link>
            )
          })}
        </div>
      )}
    </div>
  )
}
