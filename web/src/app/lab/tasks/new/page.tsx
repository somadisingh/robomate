import { createClient } from '@/lib/supabase/server'
import { redirect } from 'next/navigation'
import CreateTaskForm from './create-task-form'

async function createTask(formData: FormData) {
  'use server'

  const supabase = await createClient()
  const { data: { user } } = await supabase.auth.getUser()

  if (!user) redirect('/login')

  const requirements = formData.getAll('requirements') as string[]
  const objects = String(formData.get('objects') ?? '')
    .split(',')
    .map((item) => item.trim().toLowerCase())
    .filter(Boolean)
  const referenceAssets = formData.getAll('reference_assets') as File[]
  const firstAsset = referenceAssets.find((asset) => asset.size > 0)
  const dataType = firstAsset?.type.startsWith('video/')
    ? 'video'
    : firstAsset?.type.startsWith('image/')
      ? 'image'
      : 'image'

  const { data, error } = await supabase
    .from('tasks')
    .insert({
      lab_id: user.id,
      title: formData.get('title') as string,
      description: formData.get('description') as string,
      data_type: dataType,
      required_capabilities: requirements,
      bounty_amount: parseFloat(formData.get('bounty_amount') as string),
      quantity_needed: parseInt(formData.get('quantity_needed') as string, 10),
      deadline: formData.get('deadline') || null,
      objects,
    })
    .select('id')
    .single()

  if (error || !data) {
    console.error(error)
    return
  }

  redirect(`/lab/tasks/${data.id}`)
}

export default function NewTaskPage() {
  return <CreateTaskForm action={createTask} />
}
