import { NextResponse } from 'next/server'
import { createClient } from '@/lib/supabase/server'

const TL_API = 'https://api.twelvelabs.io/v1.3'

async function requireLabUser() {
  const supabase = await createClient()
  const { data: { user } } = await supabase.auth.getUser()
  if (!user) return { error: NextResponse.json({ error: 'Unauthorized' }, { status: 401 }) }

  const { data: profile } = await supabase
    .from('profiles')
    .select('role')
    .eq('id', user.id)
    .single()

  if (profile?.role !== 'lab') {
    return { error: NextResponse.json({ error: 'Forbidden' }, { status: 403 }) }
  }

  return { supabase, user }
}

export async function POST() {
  const auth = await requireLabUser()
  if ('error' in auth && auth.error) return auth.error
  const { supabase, user } = auth

  const apiKey = process.env.TWELVELABS_API_KEY
  const indexId = process.env.TWELVELABS_INDEX_ID
  if (!apiKey || !indexId) {
    return NextResponse.json({ error: 'TwelveLabs not configured' }, { status: 500 })
  }

  // Get all tasks owned by this lab user
  const { data: tasks } = await supabase
    .from('tasks')
    .select('id')
    .eq('lab_id', user.id)

  if (!tasks || tasks.length === 0) {
    return NextResponse.json({ indexed: 0, failed: 0, total: 0 })
  }

  const taskIds = tasks.map(t => t.id)

  // Get all submissions without a twelvelabs_video_id
  const { data: submissions } = await supabase
    .from('submissions')
    .select('id, storage_path, metadata')
    .in('task_id', taskIds)
    .filter('metadata->>twelvelabs_video_id', 'is', null)

  if (!submissions || submissions.length === 0) {
    return NextResponse.json({ indexed: 0, failed: 0, total: 0 })
  }

  const results = await Promise.allSettled(
    submissions.map(async (sub) => {
      const { data: urlData, error: urlErr } = await supabase.storage
        .from('recordings')
        .createSignedUrl(sub.storage_path.replace(/\/$/, '') + '/video.mp4', 7200)

      if (urlErr || !urlData?.signedUrl) throw new Error('could not generate URL')

      const assetForm = new FormData()
      assetForm.append('method', 'url')
      assetForm.append('url', urlData.signedUrl)

      const assetRes = await fetch(`${TL_API}/assets`, {
        method: 'POST',
        headers: { 'x-api-key': apiKey },
        body: assetForm,
      })
      if (!assetRes.ok) throw new Error(`asset creation failed (${assetRes.status})`)

      const assetId: string = (await assetRes.json())._id
      if (!assetId) throw new Error('no asset ID returned')

      const indexRes = await fetch(`${TL_API}/indexes/${indexId}/indexed-assets`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', 'x-api-key': apiKey },
        body: JSON.stringify({ asset_id: assetId }),
      })
      if (!indexRes.ok) throw new Error(`indexing failed (${indexRes.status})`)

      const videoId: string = (await indexRes.json())._id
      if (!videoId) throw new Error('no video ID returned')

      const meta = (sub.metadata ?? {}) as Record<string, unknown>
      await supabase
        .from('submissions')
        .update({ metadata: { ...meta, twelvelabs_video_id: videoId, twelvelabs_asset_id: assetId } })
        .eq('id', sub.id)
    })
  )

  const indexed = results.filter(r => r.status === 'fulfilled').length
  const errors = results
    .map((r, i) => r.status === 'rejected' ? `${submissions[i].id}: ${r.reason?.message}` : null)
    .filter(Boolean) as string[]

  return NextResponse.json({ indexed, failed: errors.length, total: submissions.length, errors })
}
