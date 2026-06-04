'use server'

import { createClient } from '@/lib/supabase/server'
import { redirect } from 'next/navigation'

export type AuthActionState = {
  error: string | null
  message?: string | null
}

function authErrorMessage(error: unknown, fallback: string) {
  if (error instanceof Error) return error.message
  return fallback
}

async function redirectForCurrentUser(
  supabase: Awaited<ReturnType<typeof createClient>>,
  userId: string,
  fallbackRole?: string
) {
  const { data: profile } = await supabase
    .from('profiles')
    .select('role')
    .eq('id', userId)
    .maybeSingle()

  const role = profile?.role ?? fallbackRole

  if (role === 'lab') return '/lab/dashboard'
  if (role === 'collector') return '/collector/tasks'
  return '/login'
}

export async function signUp(
  _prevState: AuthActionState,
  formData: FormData
): Promise<AuthActionState> {
  let redirectTo = '/login'

  const email = formData.get('email') as string
  const password = formData.get('password') as string
  const role = formData.get('role') as 'lab' | 'collector'
  const displayName = formData.get('display_name') as string

  try {
    const supabase = await createClient()

    // Pass role + display_name as metadata. The DB trigger creates the profiles row.
    const { data, error } = await supabase.auth.signUp({
      email,
      password,
      options: { data: { role, display_name: displayName } },
    })

    if (error || !data.user) {
      return { error: error?.message ?? 'Signup failed' }
    }

    if (!data.session) {
      return {
        error: null,
        message: 'Account created. Check your email to confirm it before signing in.',
      }
    }

    redirectTo = role === 'lab' ? '/lab/dashboard' : '/collector/onboard'
  } catch (error) {
    return { error: authErrorMessage(error, 'Signup failed') }
  }

  redirect(redirectTo)
}

export async function signIn(
  _prevState: AuthActionState,
  formData: FormData
): Promise<AuthActionState> {
  let redirectTo = '/login'

  const email = formData.get('email') as string
  const password = formData.get('password') as string

  try {
    const supabase = await createClient()
    const { data, error } = await supabase.auth.signInWithPassword({ email, password })

    if (error) {
      return { error: error.message }
    }

    if (!data.user) {
      return { error: 'Sign in failed' }
    }

    redirectTo = await redirectForCurrentUser(
      supabase,
      data.user.id,
      data.user.user_metadata?.role
    )
  } catch (error) {
    return { error: authErrorMessage(error, 'Sign in failed') }
  }

  redirect(redirectTo)
}

export async function signOut() {
  const supabase = await createClient()
  await supabase.auth.signOut()
  redirect('/login')
}
