'use server'

import { createClient } from '@/lib/supabase/server'
import { revalidatePath } from 'next/cache'

export async function approveSubmission(submissionId: string, taskId: string) {
  const supabase = await createClient()

  // 1. Update submission status
  const { error: updateError } = await supabase
    .from('submissions')
    .update({ status: 'approved' })
    .eq('id', submissionId)

  if (updateError) return { error: updateError.message }

  // 2. Fetch submission to get collector_id and bounty amount
  const { data: submission, error: fetchError } = await supabase
    .from('submissions')
    .select('collector_id, tasks(bounty_amount)')
    .eq('id', submissionId)
    .single()

  if (fetchError || !submission) return { error: 'Could not fetch submission details' }

  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const bounty = (submission.tasks as any)?.bounty_amount ?? 0

  // 3. Insert earnings record
  await supabase.from('earnings').insert({
    collector_id: submission.collector_id,
    submission_id: submissionId,
    amount: bounty,
    status: 'pending',
  })

  // 4. Increment quantity_filled on the task
  await supabase.rpc('increment_quantity_filled', { task_id: taskId })

  revalidatePath(`/lab/tasks/${taskId}`)
  return { success: true }
}

export async function rejectSubmission(submissionId: string, taskId: string) {
  const supabase = await createClient()

  const { error } = await supabase
    .from('submissions')
    .update({ status: 'rejected' })
    .eq('id', submissionId)

  if (error) return { error: error.message }

  revalidatePath(`/lab/tasks/${taskId}`)
  return { success: true }
}
