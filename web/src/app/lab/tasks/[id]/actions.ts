'use server'

import { createClient } from '@/lib/supabase/server'
import { revalidatePath } from 'next/cache'

export async function updateTaskMeta(
  taskId: string,
  title: string,
  description: string
): Promise<{ error?: string }> {
  const supabase = await createClient()

  const { data: { user } } = await supabase.auth.getUser()
  if (!user) return { error: 'Unauthorized' }

  const trimmedTitle = title.trim()
  if (!trimmedTitle) return { error: 'Title cannot be empty' }

  const { error } = await supabase
    .from('tasks')
    .update({ title: trimmedTitle, description: description.trim() })
    .eq('id', taskId)
    .eq('lab_id', user.id)

  if (error) return { error: error.message }

  revalidatePath(`/lab/tasks/${taskId}`)
  revalidatePath('/lab/dashboard')
  return {}
}
