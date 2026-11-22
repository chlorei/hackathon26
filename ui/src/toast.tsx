import { useSyncExternalStore } from 'react'
import clsx from 'clsx'

// One small toast stack for the whole panel: an action that failed says so here, in plain words, with its details
// one click away. Only actions (mutations) toast; a page that cannot load shows it in place.

interface Toast { id: number; tone: 'error' | 'ok'; text: string; details?: string }
let toasts: Toast[] = []
let next = 1
const listeners = new Set<() => void>()
const emit = () => listeners.forEach((l) => l())

export function toast(text: string, opts: { tone?: Toast['tone']; details?: string } = {}) {
  const t: Toast = { id: next++, tone: opts.tone ?? 'error', text, details: opts.details }
  toasts = [...toasts, t].slice(-3)
  emit()
  setTimeout(() => dismiss(t.id), t.tone === 'error' ? 8000 : 3500)
}

function dismiss(id: number) {
  toasts = toasts.filter((t) => t.id !== id)
  emit()
}

export function Toaster() {
  const list = useSyncExternalStore((l) => (listeners.add(l), () => listeners.delete(l)), () => toasts)
  if (!list.length) return null
  return (
    <div className="fixed right-5 bottom-5 z-50 flex w-[380px] flex-col gap-2" role="status" aria-live="polite">
      {list.map((t) => (
        <div key={t.id} className={clsx('rounded-xl border bg-panel px-4 py-3 shadow-[0_4px_16px_rgba(23,24,26,.08)]',
          t.tone === 'error' ? 'border-bad/40' : 'border-ok/40')}>
          <div className="flex items-start gap-3">
            <p className={clsx('flex-1 text-[14px]', t.tone === 'error' ? 'text-bad' : 'text-ok')}>{t.text}</p>
            <button onClick={() => dismiss(t.id)} className="text-faint hover:text-ink" aria-label="Dismiss">×</button>
          </div>
          {t.details && (
            <details className="mt-1.5 text-[12px] text-dim">
              <summary className="cursor-pointer select-none">Details</summary>
              <pre className="mt-1 max-h-40 overflow-auto font-mono whitespace-pre-wrap">{t.details}</pre>
            </details>
          )}
        </div>
      ))}
    </div>
  )
}
