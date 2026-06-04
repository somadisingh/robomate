import { createClient } from '@/lib/supabase/server'
import Link from 'next/link'

type EarningsRow = {
  id: string
  amount: number
  status: 'pending' | 'approved'
  created_at: string
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  submissions: any
}

const BADGE_STYLES = {
  pending: 'bg-[rgba(216,163,71,0.16)] text-[#f0cb7c]',
  approved: 'bg-[rgba(47,158,68,0.16)] text-[#99ddaa]',
}

function buildChartPoints(rows: EarningsRow[]) {
  const byDay = new Map<string, number>()

  for (const earning of rows) {
    const day = new Date(earning.created_at).toLocaleDateString('en-CA')
    byDay.set(day, (byDay.get(day) ?? 0) + earning.amount)
  }

  const points = [...byDay.entries()]
    .sort(([left], [right]) => left.localeCompare(right))
    .slice(-7)
    .map(([date, amount]) => ({ date, amount }))

  if (!points.length) {
    return []
  }

  let running = 0
  return points.map((point) => {
    running += point.amount
    return {
      date: point.date,
      amount: point.amount,
      cumulative: running,
    }
  })
}

function buildPolyline(values: { cumulative: number }[]) {
  if (!values.length) return ''

  if (values.length === 1) {
    return '24,104 296,104'
  }

  const max = Math.max(...values.map((value) => value.cumulative), 1)

  return values
    .map((value, index) => {
      const x = 24 + (272 * index) / (values.length - 1)
      const y = 104 - (72 * value.cumulative) / max
      return `${x},${y}`
    })
    .join(' ')
}

export default async function EarningsPage() {
  const supabase = await createClient()
  const { data: { user } } = await supabase.auth.getUser()

  const { data: earnings } = await supabase
    .from('earnings')
    .select('id, amount, status, created_at, submissions(task_id, tasks(title))')
    .eq('collector_id', user!.id)
    .order('created_at', { ascending: false })

  const rows = (earnings ?? []) as EarningsRow[]
  const total = rows.reduce((sum, entry) => sum + entry.amount, 0)
  const pending = rows
    .filter((entry) => entry.status === 'pending')
    .reduce((sum, entry) => sum + entry.amount, 0)
  const paidOut = total - pending
  const chartPoints = buildChartPoints([...rows].reverse())
  const polyline = buildPolyline(chartPoints)

  return (
    <div className="space-y-8">
      <section className="space-y-6">
        <div>
          <h1 className="text-3xl font-black tracking-[-0.04em] text-white sm:text-4xl">Earnings</h1>
          <p className="mt-2 text-sm text-[var(--foreground-secondary)]">
            Track what you have earned, what is still pending review, and how your payout history is building over time.
          </p>
        </div>

        <div className="grid gap-4 lg:grid-cols-[minmax(0,1.3fr)_minmax(180px,0.35fr)_minmax(180px,0.35fr)]">
          <div className="surface-panel p-6">
            <p className="text-xs font-semibold uppercase tracking-[0.16em] text-[var(--foreground-secondary)]">
              Total earned
            </p>
            <div className="mt-4 text-[clamp(3rem,7vw,5rem)] font-black leading-none tracking-[-0.05em] text-[#8ad09a]">
              ${total.toFixed(2)}
            </div>
          </div>

          <div className="surface-panel p-6">
            <p className="text-xs font-semibold uppercase tracking-[0.16em] text-[var(--foreground-secondary)]">
              Pending
            </p>
            <div className="mt-4 text-3xl font-black tracking-[-0.04em] text-[#f0cb7c]">
              ${pending.toFixed(2)}
            </div>
          </div>

          <div className="surface-panel p-6">
            <p className="text-xs font-semibold uppercase tracking-[0.16em] text-[var(--foreground-secondary)]">
              Paid out
            </p>
            <div className="mt-4 text-3xl font-black tracking-[-0.04em] text-white">
              ${paidOut.toFixed(2)}
            </div>
          </div>
        </div>

        <div className="surface-panel p-6">
          <div className="flex items-center justify-between gap-4">
            <div>
              <h2 className="text-sm font-semibold uppercase tracking-[0.16em] text-white">
                Earnings over time
              </h2>
              <p className="mt-1 text-sm text-[var(--foreground-secondary)]">
                Cumulative earnings across your most recent payout dates.
              </p>
            </div>
            <span className="text-xs uppercase tracking-[0.16em] text-[var(--foreground-secondary)]">
              Last {chartPoints.length || 0} checkpoints
            </span>
          </div>

          <div className="mt-6">
            {chartPoints.length > 0 ? (
              <div className="space-y-4">
                <svg viewBox="0 0 320 128" className="h-40 w-full">
                  <line x1="24" y1="104" x2="296" y2="104" stroke="rgba(255,255,255,0.12)" strokeWidth="1" />
                  <line x1="24" y1="68" x2="296" y2="68" stroke="rgba(255,255,255,0.08)" strokeWidth="1" />
                  <line x1="24" y1="32" x2="296" y2="32" stroke="rgba(255,255,255,0.08)" strokeWidth="1" />
                  <polyline
                    fill="none"
                    stroke="#2f9e44"
                    strokeWidth="4"
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    points={polyline}
                  />
                  {polyline.split(' ').map((point) => {
                    const [x, y] = point.split(',')
                    return <circle key={point} cx={x} cy={y} r="4" fill="#2f9e44" />
                  })}
                </svg>
                <div className="grid grid-cols-2 gap-3 text-xs text-[var(--foreground-secondary)] sm:grid-cols-4 lg:grid-cols-7">
                  {chartPoints.map((point) => (
                    <div key={point.date}>
                      <div>{new Date(point.date).toLocaleDateString(undefined, { month: 'short', day: 'numeric' })}</div>
                      <div className="mt-1 font-medium text-white">${point.cumulative.toFixed(2)}</div>
                    </div>
                  ))}
                </div>
              </div>
            ) : (
              <div className="rounded-lg border border-dashed border-[var(--border)] bg-[var(--surface-muted)] px-4 py-16 text-center text-sm text-[var(--foreground-secondary)]">
                Your earnings graph will appear once payouts start landing.
              </div>
            )}
          </div>
        </div>
      </section>

      <section className="space-y-4">
        <div className="flex items-center justify-between gap-4">
          <h2 className="text-sm font-semibold uppercase tracking-[0.16em] text-white">Earnings log</h2>
          <span className="text-xs uppercase tracking-[0.16em] text-[var(--foreground-secondary)]">
            Newest first
          </span>
        </div>

        {rows.length > 0 ? (
          <div className="space-y-2">
            {rows.map((earning) => {
              const submission = earning.submissions

              return (
                <div
                  key={earning.id}
                  className="surface-panel grid gap-4 px-4 py-4 sm:grid-cols-[minmax(0,1.3fr)_140px_120px_110px] sm:items-center"
                >
                  <div>
                    <div className="text-sm font-medium text-white">
                      {submission?.tasks?.title ?? 'Task'}
                    </div>
                    <div className="mt-1 text-xs text-[var(--foreground-secondary)]">
                      {new Date(earning.created_at).toLocaleDateString()}
                    </div>
                  </div>
                  <div className="text-sm text-[var(--foreground-secondary)]">
                    {new Date(earning.created_at).toLocaleString(undefined, {
                      month: 'short',
                      day: 'numeric',
                      hour: '2-digit',
                      minute: '2-digit',
                    })}
                  </div>
                  <div className="text-lg font-bold text-white">
                    ${earning.amount.toFixed(2)}
                  </div>
                  <div>
                    <span className={`inline-flex rounded-full px-2.5 py-1 text-xs font-semibold uppercase tracking-[0.14em] ${BADGE_STYLES[earning.status]}`}>
                      {earning.status}
                    </span>
                  </div>
                </div>
              )
            })}
          </div>
        ) : (
          <div className="surface-panel space-y-3 px-6 py-10">
            <p className="text-base font-medium text-white">No earnings yet</p>
            <p className="text-sm leading-6 text-[var(--foreground-secondary)]">
              Once your first approved submission lands, this page will populate with totals, a payout graph, and your earnings ledger.
            </p>
            <Link
              href="/collector/tasks"
              className="btn-collector inline-flex rounded-lg px-4 py-2 text-sm font-medium transition-colors"
            >
              Find tasks
            </Link>
          </div>
        )}
      </section>
    </div>
  )
}
