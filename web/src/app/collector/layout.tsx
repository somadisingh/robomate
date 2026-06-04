import { createClient } from '@/lib/supabase/server'
import { redirect } from 'next/navigation'
import { signOut } from '@/app/actions/auth'
import Link from 'next/link'

export default async function CollectorLayout({ children }: { children: React.ReactNode }) {
  const supabase = await createClient()
  const { data: { user } } = await supabase.auth.getUser()

  if (!user) redirect('/login')

  const { data: profile } = await supabase
    .from('profiles')
    .select('role, display_name')
    .eq('id', user.id)
    .single()

  if (profile?.role !== 'collector') redirect('/lab/dashboard')

  return (
    <div className="min-h-screen bg-[var(--background)] text-[var(--foreground)]">
      <nav className="border-b border-[var(--border)] bg-[rgba(15,15,15,0.92)] px-4 py-4 backdrop-blur sm:px-6">
        <div className="mx-auto flex max-w-6xl items-center justify-between">
          <div className="flex items-center gap-4 sm:gap-6">
            <Link href="/" className="flex items-center gap-2.5 transition-opacity hover:opacity-80">
              <svg viewBox="0 0 48 48" fill="none" xmlns="http://www.w3.org/2000/svg" className="w-6 h-6 text-white">
                <line x1="2" y1="24" x2="46" y2="24" stroke="currentColor" strokeWidth="0.35" strokeDasharray="1 1.6" opacity="0.38" />
                <ellipse cx="9" cy="24" rx="1.55" ry="10" stroke="currentColor" strokeWidth="0.75" opacity="0.55" />
                <ellipse cx="13" cy="24" rx="1.05" ry="6.2" stroke="currentColor" strokeWidth="0.65" opacity="0.42" />
                <ellipse cx="18" cy="24" rx="0.72" ry="4.1" stroke="currentColor" strokeWidth="0.6" opacity="0.32" />
                <ellipse cx="25" cy="24" rx="0.5" ry="2.8" stroke="currentColor" strokeWidth="0.55" opacity="0.24" />
                <ellipse cx="34" cy="24" rx="0.36" ry="2.05" stroke="currentColor" strokeWidth="0.5" opacity="0.18" />
                <path d="M 5.6 5.4 C 7.5 9, 11 22.2, 44.5 22.85" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" />
                <path d="M 5.6 42.6 C 7.5 39, 11 25.8, 44.5 25.15" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" />
                <ellipse cx="5.6" cy="24" rx="2.6" ry="18.6" stroke="currentColor" strokeWidth="1.6" />
              </svg>
              <span className="font-semibold tracking-[0.08em] text-white">Robomate</span>
            </Link>
            <span className="role-pill-collector rounded-full px-2.5 py-1 text-[11px] font-semibold uppercase tracking-[0.14em]">Collector</span>
            <Link href="/collector/tasks" className="text-sm text-[var(--foreground-secondary)] transition-colors hover:text-white">Tasks</Link>
            <Link href="/collector/earnings" className="text-sm text-[var(--foreground-secondary)] transition-colors hover:text-white">Earnings</Link>
            <Link href="/collector/leaderboard" className="text-sm text-[var(--foreground-secondary)] transition-colors hover:text-white">Leaderboard</Link>
          </div>
          <div className="flex items-center gap-3 sm:gap-4">
            <span className="hidden text-sm text-[var(--foreground-secondary)] sm:inline">{profile?.display_name}</span>
            <form action={signOut}>
              <button className="text-sm text-[var(--foreground-secondary)] transition-colors hover:text-white">Sign out</button>
            </form>
          </div>
        </div>
      </nav>
      <main className="mx-auto w-full max-w-6xl px-4 py-8 sm:px-6">{children}</main>
    </div>
  )
}
