// Small shadcn-style building blocks.
import clsx from 'clsx'
import type { ButtonHTMLAttributes, InputHTMLAttributes, ReactNode } from 'react'
import {
  Activity, Camera, CircleCheck, Film, FileText, Gauge, Plug, RefreshCcw, ShieldCheck, Sparkles, TriangleAlert,
} from 'lucide-react'
import type { Level } from './api'

export function Card({ className, children }: { className?: string; children: ReactNode }) {
  return <div className={clsx('rounded-xl border border-zinc-800 bg-zinc-900/60', className)}>{children}</div>
}

export function CardHeader({ title, icon, right }: { title: ReactNode; icon?: ReactNode; right?: ReactNode }) {
  return (
    <div className="flex items-center justify-between border-b border-zinc-800 px-4 py-2.5">
      <div className="flex items-center gap-2 text-sm font-medium text-zinc-300">
        {icon}
        {title}
      </div>
      {right}
    </div>
  )
}

type Variant = 'primary' | 'danger' | 'ghost' | 'outline'
export function Button({ variant = 'primary', className, ...props }: ButtonHTMLAttributes<HTMLButtonElement> & { variant?: Variant }) {
  return (
    <button
      {...props}
      className={clsx(
        'inline-flex items-center justify-center gap-2 rounded-lg px-3.5 py-2 text-sm font-medium transition disabled:cursor-not-allowed disabled:opacity-40',
        variant === 'primary' && 'bg-zinc-100 text-zinc-900 hover:bg-white',
        variant === 'danger' && 'bg-red-500/90 text-white hover:bg-red-500',
        variant === 'ghost' && 'text-zinc-400 hover:bg-zinc-800 hover:text-zinc-100',
        variant === 'outline' && 'border border-zinc-700 text-zinc-200 hover:bg-zinc-800',
        className,
      )}
    />
  )
}

export function Input(props: InputHTMLAttributes<HTMLInputElement>) {
  return (
    <input
      {...props}
      className={clsx(
        'w-full rounded-lg border border-zinc-800 bg-zinc-950 px-3 py-2 text-sm text-zinc-100 outline-none placeholder:text-zinc-600 focus:border-zinc-500',
        props.className,
      )}
    />
  )
}

export function Label({ children }: { children: ReactNode }) {
  return <label className="mb-1 block text-xs font-medium uppercase tracking-wide text-zinc-500">{children}</label>
}

export function Switch({ checked, onChange, disabled }: { checked: boolean; onChange: (v: boolean) => void; disabled?: boolean }) {
  return (
    <button
      type="button"
      role="switch"
      aria-checked={checked}
      disabled={disabled}
      onClick={() => onChange(!checked)}
      className={clsx(
        'relative h-5 w-9 shrink-0 rounded-full transition disabled:opacity-40',
        checked ? 'bg-emerald-500' : 'bg-zinc-700',
      )}
    >
      <span className={clsx('absolute top-0.5 h-4 w-4 rounded-full bg-white transition-all', checked ? 'left-[18px]' : 'left-0.5')} />
    </button>
  )
}

export function Badge({ tone = 'zinc', children }: { tone?: 'zinc' | 'green' | 'yellow' | 'red' | 'blue' | 'violet'; children: ReactNode }) {
  const tones = {
    zinc: 'bg-zinc-800 text-zinc-300',
    green: 'bg-emerald-500/15 text-emerald-400',
    yellow: 'bg-amber-500/15 text-amber-400',
    red: 'bg-red-500/15 text-red-400',
    blue: 'bg-sky-500/15 text-sky-400',
    violet: 'bg-violet-500/15 text-violet-300',
  }
  return <span className={clsx('inline-flex items-center gap-1 rounded-md px-1.5 py-0.5 text-xs font-medium', tones[tone])}>{children}</span>
}

export const LEVEL_STYLE: Record<Level, { text: string; dot: string; border: string }> = {
  info: { text: 'text-zinc-300', dot: 'bg-zinc-500', border: 'border-zinc-800' },
  warn: { text: 'text-amber-300', dot: 'bg-amber-400', border: 'border-amber-500/30' },
  error: { text: 'text-red-300', dot: 'bg-red-500', border: 'border-red-500/40' },
  success: { text: 'text-emerald-300', dot: 'bg-emerald-400', border: 'border-emerald-500/40' },
}

const PLUGIN_ICONS: Record<string, typeof Plug> = {
  'success-check': CircleCheck,
  'failure-monitor': Sparkles,
  'camera-check': Camera,
  retry: RefreshCcw,
  'latency-monitor': Gauge,
  recorder: Film,
  safety: ShieldCheck,
  report: FileText,
  dreamkit: Activity,
}

export function PluginIcon({ name, className }: { name: string; className?: string }) {
  const Icon = PLUGIN_ICONS[name] ?? (name.includes('check') ? TriangleAlert : Plug)
  return <Icon className={clsx('h-4 w-4', className)} />
}

export function ResultMark({ success }: { success: boolean | null | undefined }) {
  if (success === true) return <span className="font-semibold text-emerald-400">✓</span>
  if (success === false) return <span className="font-semibold text-red-400">✗</span>
  return <span className="text-zinc-500">?</span>
}

export function formatTime(ts?: number) {
  if (!ts) return ''
  return new Date(ts * 1000).toLocaleString(undefined, { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' })
}
