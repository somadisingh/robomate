import { NextRequest, NextResponse } from 'next/server'
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

export async function POST(request: NextRequest) {
  const auth = await requireLabUser()
  if ('error' in auth && auth.error) return auth.error
  const { supabase, user } = auth

  let body: { submissionId?: string; force?: boolean }
  try {
    body = await request.json()
  } catch {
    return NextResponse.json({ error: 'Invalid JSON' }, { status: 400 })
  }

  const { submissionId, force } = body
  if (!submissionId) return NextResponse.json({ error: 'Missing submissionId' }, { status: 400 })

  const apiKey = process.env.TWELVELABS_API_KEY
  const indexId = process.env.TWELVELABS_INDEX_ID
  if (!apiKey || !indexId) {
    return NextResponse.json({ error: 'TwelveLabs not configured' }, { status: 500 })
  }

  const { data: sub, error: fetchErr } = await supabase
    .from('submissions')
    .select('id, storage_path, metadata, task_id')
    .eq('id', submissionId)
    .single()

  if (fetchErr || !sub) return NextResponse.json({ error: 'Submission not found' }, { status: 404 })

  const { data: task } = await supabase
    .from('tasks')
    .select('lab_id')
    .eq('id', sub.task_id)
    .single()

  if (!task || task.lab_id !== user.id) {
    return NextResponse.json({ error: 'Forbidden' }, { status: 403 })
  }

  const meta = (sub.metadata ?? {}) as Record<string, unknown>

  // If already indexed, verify the video actually exists in TwelveLabs before short-circuiting
  if (meta.twelvelabs_video_id && !force) {
    const verifyRes = await fetch(
      `${TL_API}/indexes/${indexId}/indexed-assets/${meta.twelvelabs_video_id}`,
      { headers: { 'x-api-key': apiKey } }
    )
    if (verifyRes.ok) {
      return NextResponse.json({ success: true, videoId: meta.twelvelabs_video_id, alreadyIndexed: true })
    }
    // Stale metadata — fall through and re-index
    console.log(`Stale twelvelabs_video_id for submission ${submissionId}, re-indexing`)
  }

  // Generate a 2-hour signed URL so TwelveLabs has time to download it
  const { data: urlData, error: urlErr } = await supabase.storage
    .from('recordings')
    .createSignedUrl(sub.storage_path.replace(/\/$/, '') + '/video.mp4', 7200)

  if (urlErr || !urlData?.signedUrl) {
    return NextResponse.json({ error: 'Could not generate video URL' }, { status: 500 })
  }

  // Step 1: Create a TwelveLabs asset from the video URL
  const assetForm = new FormData()
  assetForm.append('method', 'url')
  assetForm.append('url', urlData.signedUrl)

  const assetRes = await fetch(`${TL_API}/assets`, {
    method: 'POST',
    headers: { 'x-api-key': apiKey },
    body: assetForm,
  })

  if (!assetRes.ok) {
    const err = await assetRes.text()
    console.error('TwelveLabs asset creation failed:', assetRes.status, err)
    return NextResponse.json({ error: `Failed to create asset: ${err}` }, { status: 502 })
  }

  // TwelveLabs returns _id (not id) in REST responses
  const assetData = await assetRes.json()
  const assetId: string = assetData._id

  if (!assetId) {
    console.error('TwelveLabs asset response missing _id:', assetData)
    return NextResponse.json({ error: 'Asset creation returned no ID' }, { status: 502 })
  }

  // Step 2: Index the asset (kicks off async processing in TwelveLabs)
  const indexRes = await fetch(`${TL_API}/indexes/${indexId}/indexed-assets`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', 'x-api-key': apiKey },
    body: JSON.stringify({ asset_id: assetId }),
  })

  if (!indexRes.ok) {
    const err = await indexRes.text()
    console.error('TwelveLabs indexing failed:', indexRes.status, err)
    return NextResponse.json({ error: `Failed to index video: ${err}` }, { status: 502 })
  }

  // TwelveLabs returns _id (not id) in REST responses
  const indexData = await indexRes.json()
  const videoId: string = indexData._id

  if (!videoId) {
    console.error('TwelveLabs indexed-asset response missing _id:', indexData)
    return NextResponse.json({ error: 'Indexing returned no video ID' }, { status: 502 })
  }

  // Step 3: Store TwelveLabs IDs in submission metadata for search mapping
  const { error: updateErr } = await supabase
    .from('submissions')
    .update({ metadata: { ...meta, twelvelabs_video_id: videoId, twelvelabs_asset_id: assetId } })
    .eq('id', submissionId)

  if (updateErr) {
    console.error('Failed to update submission metadata:', updateErr)
    // Return success anyway — video is indexed in TwelveLabs, just metadata link failed
    return NextResponse.json({ success: true, videoId, metadataUpdateFailed: true })
  }

  return NextResponse.json({ success: true, videoId })
}
