import { createClient } from '@/lib/supabase/server'
import { redirect } from 'next/navigation'
import { signOut } from '@/app/actions/auth'
import ToastContainer from '@/components/toast'
import { Brand } from '@/components/logo'
import Link from 'next/link'

export default async function LabLayout({ children }: { children: React.ReactNode }) {
  const supabase = await createClient()
  const { data: { user } } = await supabase.auth.getUser()

  if (!user) redirect('/login')

  const { data: profile } = await supabase
    .from('profiles')
    .select('role, display_name')
    .eq('id', user.id)
    .single()

  if (profile?.role !== 'lab') redirect('/collector/tasks')

  return (
    <div className="min-h-screen bg-[var(--background)] text-[var(--foreground)]">
      <div className="sticky top-0 z-30 px-3 pt-3 sm:px-4 sm:pt-4">
        <nav className="glass-nav mx-auto max-w-6xl px-4 py-3 sm:px-5 sm:py-3.5">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-4 sm:gap-6">
              <Link href="/" className="flex items-center transition-opacity hover:opacity-80">
                <Brand logoClassName="h-6 w-6" nameClassName="text-sm" />
              </Link>
              <span className="role-pill-lab rounded-full px-2.5 py-1 text-[11px] font-semibold uppercase tracking-[0.14em]">Lab</span>
              <Link href="/lab/dashboard" className="hidden text-sm text-[var(--foreground-secondary)] transition-colors hover:text-white sm:inline">Dashboard</Link>
              <Link href="/lab/tasks/new" className="hidden text-sm text-[var(--foreground-secondary)] transition-colors hover:text-white sm:inline">New Task</Link>
              <Link href="/lab/search" className="hidden text-sm text-[var(--foreground-secondary)] transition-colors hover:text-white sm:inline">Search</Link>
            </div>
            <div className="flex items-center gap-3 sm:gap-4">
              <span className="hidden text-sm text-[var(--foreground-secondary)] sm:inline">{profile?.display_name}</span>
              <form action={signOut}>
                <button className="text-sm text-[var(--foreground-secondary)] transition-colors hover:text-white">Sign out</button>
              </form>
            </div>
          </div>
          {/* Mobile-only nav row */}
          <div className="flex items-center gap-5 pt-2 sm:hidden">
            <Link href="/lab/dashboard" className="text-sm text-[var(--foreground-secondary)] transition-colors hover:text-white">Dashboard</Link>
            <Link href="/lab/tasks/new" className="text-sm text-[var(--foreground-secondary)] transition-colors hover:text-white">New Task</Link>
            <Link href="/lab/search" className="text-sm text-[var(--foreground-secondary)] transition-colors hover:text-white">Search</Link>
          </div>
        </nav>
      </div>
      <main className="mx-auto w-full max-w-6xl px-4 py-8 sm:px-6">{children}</main>
      <ToastContainer />
    </div>
  )
}
