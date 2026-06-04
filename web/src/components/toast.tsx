'use client'

import { useEffect, useState } from 'react'

type Toast = { id: number; message: string }

let addToast: ((message: string) => void) | null = null

export function triggerToast(message: string) {
  addToast?.(message)
}

export default function ToastContainer() {
  const [toasts, setToasts] = useState<Toast[]>([])

  useEffect(() => {
    addToast = (message: string) => {
      const id = Date.now()
      setToasts(prev => [...prev, { id, message }])
      setTimeout(() => {
        setToasts(prev => prev.filter(t => t.id !== id))
      }, 4000)
    }
    return () => { addToast = null }
  }, [])

  if (!toasts.length) return null

  return (
    <div className="fixed bottom-4 right-4 flex flex-col gap-2 z-50">
      {toasts.map(toast => (
        <div
          key={toast.id}
          className="surface-panel flex items-center gap-2 px-4 py-3 text-sm text-white"
          style={{ animation: 'toast-in 220ms cubic-bezier(0.2, 0.7, 0.2, 1)' }}
        >
          <span className="h-2 w-2 shrink-0 rounded-full bg-[#2f9e44]" />
          {toast.message}
        </div>
      ))}
    </div>
  )
}
