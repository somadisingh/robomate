import { createClient } from '@/lib/supabase/server'
import { redirect } from 'next/navigation'
import CreateTaskForm from './create-task-form'
import {
  embedDocument,
  pineconeUpsert,
  pineconeConfigured,
  NAMESPACE_TASKS,
} from '@/lib/pinecone'

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
  const title = formData.get('title') as string
  const description = (formData.get('description') as string) ?? ''
  const bountyAmount = parseFloat(formData.get('bounty_amount') as string)
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
      title,
      description,
      data_type: dataType,
      required_capabilities: requirements,
      bounty_amount: bountyAmount,
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

  // Feature 3: index the task into Pinecone "tasks" namespace so collector
  // profiles can be matched against it. Fire-and-forget — never block creation.
  try {
    if (pineconeConfigured()) {
      const taskText = `Task: ${title}. ${description}. Objects: ${objects.join(', ')}.`
      const vector = await embedDocument(taskText)
      await pineconeUpsert({
        vectors: [
          {
            id: data.id,
            values: vector,
            metadata: {
              task_id: data.id,
              title,
              lab_id: user.id,
              bounty_amount: bountyAmount,
              objects,
            },
          },
        ],
        namespace: NAMESPACE_TASKS,
      })
    }
  } catch (e) {
    console.error('[task-index]', e)
  }

  redirect(`/lab/tasks/${data.id}`)
}

export default function NewTaskPage() {
  return <CreateTaskForm action={createTask} />
}
