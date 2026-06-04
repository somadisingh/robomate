import { createClient } from '@/lib/supabase/server'
import Link from 'next/link'

export default async function LabDashboard() {
  const supabase = await createClient()
  const { data: { user } } = await supabase.auth.getUser()

  const { data: tasks } = await supabase
    .from('tasks')
    .select('id, title, bounty_amount, quantity_needed, quantity_filled, status, created_at')
    .eq('lab_id', user!.id)
    .order('created_at', { ascending: false })

  // Fetch submission counts per task (all received, regardless of approval status)
  const taskIds = tasks?.map(t => t.id) ?? []
  const { data: submissionCounts } = taskIds.length
    ? await supabase
        .from('submissions')
        .select('task_id, status')
        .in('task_id', taskIds)
    : { data: [] }

  const receivedByTask: Record<string, number> = {}
  for (const s of submissionCounts ?? []) {
    receivedByTask[s.task_id] = (receivedByTask[s.task_id] ?? 0) + 1
  }

  const totalSpend = tasks?.reduce((sum, t) => sum + (t.bounty_amount * t.quantity_filled), 0) ?? 0
  const totalReceived = Object.values(receivedByTask).reduce((sum, n) => sum + n, 0)

  return (
    <div>
      <div className="flex items-center justify-between mb-8">
        <h1 className="text-2xl font-bold text-white">Your Tasks</h1>
        <Link
          href="/lab/tasks/new"
          className="btn-lab rounded-lg px-4 py-2 text-sm font-medium transition-colors"
        >
          + New Task
        </Link>
      </div>

      {/* Stats */}
      <div className="grid grid-cols-1 sm:grid-cols-3 gap-4 mb-8">
        <div className="surface-panel p-4">
          <div className="text-2xl font-bold text-white">{tasks?.length ?? 0}</div>
          <div className="mt-1 text-sm text-[var(--foreground-secondary)]">Total tasks</div>
        </div>
        <div className="surface-panel p-4">
          <div className="text-2xl font-bold text-white">{totalReceived}</div>
          <div className="mt-1 text-sm text-[var(--foreground-secondary)]">Submissions received</div>
        </div>
        <div className="surface-panel p-4">
          <div className="text-2xl font-bold text-[#aebeff]">${totalSpend.toFixed(2)}</div>
          <div className="mt-1 text-sm text-[var(--foreground-secondary)]">Total spent</div>
        </div>
      </div>

      {/* Task list */}
      {!tasks?.length ? (
        <div className="surface-panel py-20 text-center">
          <div className="text-4xl mb-3">📋</div>
          <p className="text-[var(--foreground-secondary)]">No tasks yet. Create your first one.</p>
          <Link
            href="/lab/tasks/new"
            className="btn-lab mt-4 inline-block rounded-lg px-4 py-2 text-sm font-medium"
          >
            Create task
          </Link>
        </div>
      ) : (
        <div className="space-y-3">
          {tasks.map(task => (
            <Link
              key={task.id}
              href={`/lab/tasks/${task.id}`}
              className="surface-panel block p-5 transition-colors hover:border-[rgba(255,255,255,0.2)]"
            >
              <div className="flex items-start justify-between">
                <div>
                  <h3 className="font-semibold text-white">{task.title}</h3>
                  <div className="flex items-center gap-3 mt-2 flex-wrap">
                    <span className="text-sm text-[var(--foreground-secondary)]">
                      ${task.bounty_amount} / submission
                    </span>
                    <span className="text-sm text-[var(--foreground-secondary)]">·</span>
                    <span className="text-sm text-[var(--foreground-secondary)]">
                      {receivedByTask[task.id] ?? 0} / {task.quantity_needed} received
                    </span>
                    <span className="text-sm text-[var(--foreground-secondary)]">·</span>
                    <span className="text-sm text-[#aebeff]">
                      {task.quantity_filled} / {task.quantity_needed} approved
                    </span>
                  </div>
                </div>
                <span className={`rounded-full px-2 py-1 text-xs font-medium ${
                  task.status === 'open'
                    ? 'bg-[rgba(59,91,219,0.16)] text-[#aebeff]'
                    : 'bg-[rgba(115,120,131,0.18)] text-[#d3d7de]'
                }`}>
                  {task.status}
                </span>
              </div>
              {/* Progress bars: received (dim) behind approved (solid) */}
              <div className="mt-3 relative h-1.5 overflow-hidden rounded-full bg-[rgba(255,255,255,0.08)]">
                <div
                  className="absolute inset-y-0 left-0 rounded-full bg-[rgba(59,91,219,0.35)]"
                  style={{ width: `${Math.min(100, ((receivedByTask[task.id] ?? 0) / task.quantity_needed) * 100)}%` }}
                />
                <div
                  className="absolute inset-y-0 left-0 rounded-full bg-[#3b5bdb]"
                  style={{ width: `${Math.min(100, (task.quantity_filled / task.quantity_needed) * 100)}%` }}
                />
              </div>
            </Link>
          ))}
        </div>
      )}
    </div>
  )
}
