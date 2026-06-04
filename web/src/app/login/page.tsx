'use client'

import { useActionState } from 'react'
import { signIn } from '@/app/actions/auth'
import Link from 'next/link'

const isSupabaseConfigured = Boolean(
  process.env.NEXT_PUBLIC_SUPABASE_URL &&
    process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY
)

export default function LoginPage() {
  const [state, formAction, pending] = useActionState(signIn, {
    error: null,
    message: null,
  })

  return (
    <div className="flex min-h-screen items-center justify-center bg-[var(--background)] px-4">
      <div className="surface-panel w-full max-w-md p-8">
        <div className="mb-8">
          <h1 className="text-2xl font-bold text-white">Welcome back</h1>
          <p className="mt-1 text-sm text-[var(--foreground-secondary)]">Sign in to your account</p>
        </div>

        <form action={formAction} className="space-y-4">
          <div>
            <label className="mb-1 block text-sm font-medium text-white">Email</label>
            <input
              name="email"
              type="email"
              required
              className="input-dark text-sm"
            />
          </div>
          <div>
            <label className="mb-1 block text-sm font-medium text-white">Password</label>
            <input
              name="password"
              type="password"
              required
              className="input-dark text-sm"
            />
          </div>

          {state.error && (
            <p className="text-red-500 text-sm">{state.error}</p>
          )}

          {!isSupabaseConfigured && (
            <p className="text-red-500 text-sm">
              Supabase is not configured for this local server.
            </p>
          )}

          <button
            type="submit"
            disabled={pending || !isSupabaseConfigured}
            className="btn-lab w-full rounded-lg py-2 text-sm font-medium transition-colors disabled:opacity-50"
          >
            {pending ? 'Signing in...' : 'Sign in'}
          </button>
        </form>

        <p className="mt-6 text-center text-sm text-[var(--foreground-secondary)]">
          No account?{' '}
          <Link href="/signup" className="font-medium text-white hover:underline">
            Sign up
          </Link>
        </p>
      </div>
    </div>
  )
}
