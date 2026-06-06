/**
 * Shared Pinecone + Gemini embedding helpers for the web app.
 *
 * ONE index, THREE namespaces ("recordings", "tasks", "collectors").
 *
 * Embeddings use Gemini `gemini-embedding-001` at 768 dimensions (NOT
 * text-embedding-004, which 404s for this project's key). This matches exactly
 * how the Python backend populates the index, so query and document vectors are
 * comparable. The index metric is cosine, so output magnitude does not affect
 * ranking.
 *
 * All calls use plain fetch — no Pinecone SDK dependency.
 */

const EMBED_MODEL = 'gemini-embedding-001'
const EMBED_DIM = 768

export const NAMESPACE_RECORDINGS = 'recordings'
export const NAMESPACE_TASKS = 'tasks'
export const NAMESPACE_COLLECTORS = 'collectors'

export type EmbedTaskType = 'RETRIEVAL_QUERY' | 'RETRIEVAL_DOCUMENT'

export type PineconeMatch = {
  id: string
  score: number
  metadata: Record<string, unknown>
}

export type PineconeVector = {
  id: string
  values: number[]
  metadata?: Record<string, unknown>
}

export function pineconeConfigured(): boolean {
  return Boolean(
    process.env.PINECONE_API_KEY &&
      process.env.PINECONE_INDEX_HOST &&
      process.env.GEMINI_API_KEY
  )
}

/** Embed text with Gemini. Throws on failure. */
export async function embedText(
  text: string,
  taskType: EmbedTaskType
): Promise<number[]> {
  const key = process.env.GEMINI_API_KEY
  if (!key) throw new Error('GEMINI_API_KEY not configured')

  const res = await fetch(
    `https://generativelanguage.googleapis.com/v1beta/models/${EMBED_MODEL}:embedContent?key=${key}`,
    {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        model: `models/${EMBED_MODEL}`,
        content: { parts: [{ text }] },
        taskType,
        outputDimensionality: EMBED_DIM,
      }),
    }
  )
  if (!res.ok) {
    throw new Error(`Gemini embed failed (${res.status}): ${await res.text()}`)
  }
  const data = await res.json()
  const values = data?.embedding?.values
  if (!Array.isArray(values)) {
    throw new Error('Gemini embed returned no values')
  }
  return values as number[]
}

export async function embedQuery(text: string): Promise<number[]> {
  return embedText(text, 'RETRIEVAL_QUERY')
}

export async function embedDocument(text: string): Promise<number[]> {
  return embedText(text, 'RETRIEVAL_DOCUMENT')
}

function pineconeHostKey(): { host: string; key: string } {
  const host = process.env.PINECONE_INDEX_HOST
  const key = process.env.PINECONE_API_KEY
  if (!host || !key) throw new Error('Pinecone not configured')
  return { host, key }
}

export async function pineconeQuery(params: {
  vector: number[]
  topK: number
  namespace: string
  filter?: Record<string, unknown>
}): Promise<PineconeMatch[]> {
  const { host, key } = pineconeHostKey()
  const res = await fetch(`${host}/query`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', 'Api-Key': key },
    body: JSON.stringify({
      vector: params.vector,
      topK: params.topK,
      includeMetadata: true,
      namespace: params.namespace,
      ...(params.filter ? { filter: params.filter } : {}),
    }),
  })
  if (!res.ok) {
    throw new Error(`Pinecone query failed (${res.status}): ${await res.text()}`)
  }
  const data = await res.json()
  return (data.matches ?? []) as PineconeMatch[]
}

export async function pineconeUpsert(params: {
  vectors: PineconeVector[]
  namespace: string
}): Promise<void> {
  const { host, key } = pineconeHostKey()
  const res = await fetch(`${host}/vectors/upsert`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', 'Api-Key': key },
    body: JSON.stringify({ vectors: params.vectors, namespace: params.namespace }),
  })
  if (!res.ok) {
    throw new Error(`Pinecone upsert failed (${res.status}): ${await res.text()}`)
  }
}
