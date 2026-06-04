'use client'

import { useActionState, useState } from 'react'
import { signUp } from '@/app/actions/auth'
import Link from 'next/link'

const isSupabaseConfigured = Boolean(
  process.env.NEXT_PUBLIC_SUPABASE_URL &&
    process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY
)

export default function SignupPage() {
  const [state, formAction, pending] = useActionState(signUp, {
    error: null,
    message: null,
  })
  const [role, setRole] = useState<'lab' | 'collector' | null>(null)

  return (
    <div className="flex min-h-screen items-center justify-center bg-[var(--background)] px-4 py-8">
      <div className="surface-panel w-full max-w-md p-8">
        <div className="mb-8">
          <h1 className="text-2xl font-bold text-white">Create account</h1>
          <p className="mt-1 text-sm text-[var(--foreground-secondary)]">Join the data marketplace</p>
        </div>

        <form action={formAction} className="space-y-4">
          {/* Role selection */}
          <div>
            <label className="mb-2 block text-sm font-medium text-white">I am a...</label>
            <div className="grid grid-cols-2 gap-3">
              <button
                type="button"
                onClick={() => setRole('lab')}
                className={`rounded-lg border p-4 text-left transition-all ${
                  role === 'lab'
                    ? 'border-[rgba(59,91,219,0.65)] bg-[rgba(59,91,219,0.16)] text-white'
                    : 'border-[var(--border)] bg-[var(--surface-muted)] text-[var(--foreground-secondary)] hover:border-[rgba(255,255,255,0.22)]'
                }`}
              >
                <div className="text-lg mb-1">🔬</div>
                <div className="font-medium text-sm">Lab / Researcher</div>
                <div className={`mt-1 text-xs ${role === 'lab' ? 'text-[#d7deff]' : 'text-[var(--foreground-secondary)]'}`}>
                  Post data collection tasks
                </div>
              </button>
              <button
                type="button"
                onClick={() => setRole('collector')}
                className={`rounded-lg border p-4 text-left transition-all ${
                  role === 'collector'
                    ? 'border-[rgba(47,158,68,0.7)] bg-[rgba(47,158,68,0.16)] text-white'
                    : 'border-[var(--border)] bg-[var(--surface-muted)] text-[var(--foreground-secondary)] hover:border-[rgba(255,255,255,0.22)]'
                }`}
              >
                <div className="text-lg mb-1">📱</div>
                <div className="font-medium text-sm">Data Collector</div>
                <div className={`mt-1 text-xs ${role === 'collector' ? 'text-[#c6efd0]' : 'text-[var(--foreground-secondary)]'}`}>
                  Earn money collecting data
                </div>
              </button>
            </div>
            {role && <input type="hidden" name="role" value={role} />}
          </div>

          <div>
            <label className="mb-1 block text-sm font-medium text-white">Display name</label>
            <input
              name="display_name"
              type="text"
              required
              placeholder={role === 'lab' ? 'e.g. Stanford AI Lab' : 'e.g. Alex Chen'}
              className="input-dark text-sm"
            />
          </div>

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
              minLength={6}
              className="input-dark text-sm"
            />
          </div>

          {state.error && (
            <p className="text-red-500 text-sm">{state.error}</p>
          )}

          {state.message && (
            <p className="text-green-600 text-sm">{state.message}</p>
          )}

          {!isSupabaseConfigured && (
            <p className="text-red-500 text-sm">
              Supabase is not configured for this local server.
            </p>
          )}

          <button
            type="submit"
            disabled={pending || !role || !isSupabaseConfigured}
            className={`w-full rounded-lg py-2 text-sm font-medium transition-colors disabled:opacity-40 ${
              role === 'collector' ? 'btn-collector' : 'btn-lab'
            }`}
          >
            {pending ? 'Creating account...' : 'Create account'}
          </button>
        </form>

        <p className="mt-6 text-center text-sm text-[var(--foreground-secondary)]">
          Already have an account?{' '}
          <Link href="/login" className="font-medium text-white hover:underline">
            Sign in
          </Link>
        </p>
      </div>
    </div>
  )
}
