'use client'

import { useState } from 'react'

export default function CopyButton({ text }: { text: string }) {
  const [copied, setCopied] = useState(false)

  async function handleCopy() {
    await navigator.clipboard.writeText(text)
    setCopied(true)
    setTimeout(() => setCopied(false), 2000)
  }

  return (
    <button
      onClick={handleCopy}
      className="btn-neutral shrink-0 rounded-lg px-3 py-2 text-xs font-medium transition-colors"
    >
      {copied ? 'Copied!' : 'Copy'}
    </button>
  )
}
