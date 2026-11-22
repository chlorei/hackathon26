import clsx from 'clsx'
import { useEffect, useRef, useState, type ButtonHTMLAttributes, type InputHTMLAttributes, type ReactNode } from 'react'
import { icons, Plug } from 'lucide-react'
import type { Level } from './api'
import type { LiveStore } from './useRunStream'

export function Panel({ className, children }: { className?: string; children: ReactNode }) {
  return <section className={clsx('rounded-xl border border-line bg-panel', className)}>{children}</section>
}

export function PanelTitle({ children, right, className }: { children: ReactNode; right?: ReactNode; className?: string }) {
  return (
    <div className={clsx('flex items-center justify-between gap-3 px-5 pt-4 pb-2.5', className)}>
      <h2 className="text-[15px] font-semibold text-ink">{children}</h2>
      {right}
    </div>
  )
}

type Variant = 'go' | 'stop' | 'quiet' | 'line'
export function Button({ variant = 'line', className, ...props }: ButtonHTMLAttributes<HTMLButtonElement> & { variant?: Variant }) {
  return (
    <button
      {...props}
      className={clsx(
        'inline-flex items-center justify-center gap-2 rounded-lg px-3.5 py-1.5 text-[13px] font-medium transition-colors disabled:cursor-not-allowed disabled:opacity-40',
        variant === 'go' && 'bg-ink text-white hover:bg-black',
        variant === 'stop' && 'border border-line bg-panel text-bad hover:bg-bad-soft',
        variant === 'quiet' && 'text-dim hover:bg-raised hover:text-ink',
        variant === 'line' && 'border border-line bg-panel text-ink hover:bg-raised',
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
        'w-full rounded-lg border border-line bg-panel px-2.5 py-1.5 text-[13px] text-ink outline-none placeholder:text-faint focus:border-dim disabled:opacity-60',
        props.className,
      )}
    />
  )
}

export function Toggle({ checked, onChange, disabled, label }: { checked: boolean; onChange: (v: boolean) => void; disabled?: boolean; label?: ReactNode }) {
  return (
    <button type="button" role="switch" aria-checked={checked} disabled={disabled} onClick={() => onChange(!checked)}
      className="group inline-flex items-center gap-2 text-[13px] text-dim disabled:opacity-40">
      <span className={clsx('relative h-[18px] w-8 shrink-0 rounded-full transition-colors', checked ? 'bg-ok' : 'bg-line')}>
        <span className={clsx('absolute top-[2px] h-[14px] w-[14px] rounded-full bg-white shadow-sm transition-all', checked ? 'left-[16px]' : 'left-[2px]')} />
      </span>
      {label}
    </button>
  )
}

export function Tag({ tone = 'plain', children, title }: { tone?: 'plain' | 'ok' | 'bad' | 'signal' | 'steel' | 'warn'; children: ReactNode; title?: string }) {
  return (
    <span title={title} className={clsx('inline-flex items-center gap-1 rounded-[20px] px-2 py-px text-[12px] font-semibold',
      tone === 'plain' && 'bg-raised text-dim',
      tone === 'ok' && 'bg-ok-soft text-ok',
      tone === 'bad' && 'bg-bad-soft text-bad',
      tone === 'signal' && 'bg-signal-soft text-signal',
      tone === 'warn' && 'bg-[#f6ecdc] text-warn',
      tone === 'steel' && 'bg-[#e3f0f5] text-steel')}>{children}</span>
  )
}

export function StatusDot({ tone, pulse }: { tone: 'ok' | 'bad' | 'run' | 'off' | 'warn'; pulse?: boolean }) {
  return <span className={clsx('inline-block h-[9px] w-[9px] shrink-0 rounded-full',
    tone === 'ok' && 'bg-ok', tone === 'bad' && 'bg-bad', tone === 'run' && 'bg-signal', tone === 'warn' && 'bg-warn',
    tone === 'off' && 'bg-[#cfccc5]', pulse && 'dk-pulse')} />
}

export const LEVEL_TONE: Record<Level, string> = { info: 'text-dim', warn: 'text-warn', error: 'text-bad', success: 'text-ok' }

/** Plugin icons come from the plugin's manifest (a lucide icon name); the feed learns them from /api/plugins. */
export const ICON_BY_PLUGIN: Record<string, string> = { dreamkit: 'Activity' }
export function PluginIcon({ name, icon, className }: { name: string; icon?: string; className?: string }) {
  const Icon = (icons as Record<string, typeof Plug>)[icon ?? ICON_BY_PLUGIN[name] ?? 'Plug'] ?? Plug
  return <Icon className={clsx('h-[15px] w-[15px] shrink-0', className)} strokeWidth={1.75} />
}

export function Outcome({ success }: { success: boolean | null | undefined }) {
  if (success === true) return <span className="text-ok" aria-label="success">✓</span>
  if (success === false) return <span className="text-bad" aria-label="failed">✗</span>
  return <span className="text-faint" aria-label="unknown">–</span>
}

export function when(ts?: number) {
  if (!ts) return ''
  return new Date(ts * 1000).toLocaleString('en-US', { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' })  // 'Nov 22, 11:45 AM', whatever the system language
}

// ---------------------------------------------------------------------------------------------------------------
// The attempt strip: one attempt as a line of time. Marks are what the plugins saw; the red notch is the verdict.

export interface Mark { t: number; label: string; tone: 'info' | 'warn' | 'bad' | 'ok' }

export function AttemptStrip({ duration, marks, verdict, playhead, onSeek, live }: {
  duration: number
  marks: Mark[]
  verdict?: { t: number; label: string } | null
  playhead?: number
  onSeek?: (t: number) => void
  live?: boolean
}) {
  const [hover, setHover] = useState<number | null>(null)
  const span = Math.max(duration, 1)
  const pct = (t: number) => `${Math.min(100, Math.max(0, (100 * t) / span))}%`
  const ticks = Array.from({ length: Math.floor(span / 5) + 1 }, (_, i) => i * 5)
  return (
    <div className="select-none">
      <div
        className={clsx('relative h-11 rounded-lg border border-line bg-raised', onSeek && 'cursor-pointer')}
        onMouseMove={(e) => {
          const r = e.currentTarget.getBoundingClientRect()
          setHover(((e.clientX - r.left) / r.width) * span)
        }}
        onMouseLeave={() => setHover(null)}
        onClick={(e) => {
          if (!onSeek) return
          const r = e.currentTarget.getBoundingClientRect()
          onSeek(((e.clientX - r.left) / r.width) * span)
        }}
      >
        {playhead != null && (
          <div className={clsx('absolute inset-y-0 left-0 rounded-l-lg', live ? 'bg-signal-soft' : 'bg-panel/70')} style={{ width: pct(playhead) }} />
        )}
        {marks.map((m, i) => (
          <div key={i} title={`${m.t.toFixed(1)} s  ${m.label}`} className="group absolute inset-y-0" style={{ left: pct(m.t) }}>
            <div className={clsx('absolute top-2 bottom-2 w-[3px] -translate-x-1/2 rounded-full',
              m.tone === 'bad' && 'bg-bad', m.tone === 'warn' && 'bg-warn', m.tone === 'ok' && 'bg-ok', m.tone === 'info' && 'bg-faint')} />
          </div>
        ))}
        {verdict && (
          <div className="absolute inset-y-0" style={{ left: pct(verdict.t) }}>
            <div className="absolute inset-y-0 w-[2px] -translate-x-1/2 bg-bad" />
            <div className="absolute -top-[7px] h-0 w-0 -translate-x-1/2 border-x-[6px] border-t-[7px] border-x-transparent border-t-bad" />
          </div>
        )}
        {playhead != null && (
          <div className="absolute inset-y-0 w-[2px] -translate-x-1/2 bg-ink" style={{ left: pct(playhead) }} />
        )}
        {hover != null && onSeek && (
          <div className="pointer-events-none absolute inset-y-0 w-px bg-ink/40" style={{ left: pct(hover) }}>
            <span className="absolute -top-6 -translate-x-1/2 rounded-md bg-ink px-1.5 py-0.5 text-[11px] text-white">{hover.toFixed(1)} s</span>
          </div>
        )}
      </div>
      <div className="relative mt-1 h-4 text-[11px] text-faint">
        {ticks.map((t) => (
          <span key={t} className="absolute -translate-x-1/2" style={{ left: pct(t) }}>{t}s</span>
        ))}
      </div>
    </div>
  )
}

// ---------------------------------------------------------------------------------------------------------------
// A plugin's own page (PLUGIN_STANDARD.md §7a): a side panel with a sandboxed iframe. The server links the theme in.
export type PluginPage = { plugin: string; runId: string; episode: number | null }

export function PluginPageDrawer({ page, onClose }: { page: PluginPage | null; onClose: () => void }) {
  useEffect(() => {
    if (!page) return
    const onKey = (e: KeyboardEvent) => e.key === 'Escape' && onClose()
    // the page's "Change settings" (a sandboxed page cannot navigate the panel; it asks)
    const onMessage = (e: MessageEvent) => {
      if (e.data?.type === 'dreamkit:open-settings' && e.data.plugin === page.plugin) {
        onClose()
        window.location.hash = `/plugins/${encodeURIComponent(page.plugin)}`
      }
    }
    window.addEventListener('keydown', onKey)
    window.addEventListener('message', onMessage)
    return () => { window.removeEventListener('keydown', onKey); window.removeEventListener('message', onMessage) }
  }, [page, onClose])
  if (!page) return null
  const q = new URLSearchParams({ ...(page.runId ? { run_id: page.runId } : {}), ...(page.episode != null ? { attempt: String(page.episode) } : {}) })
  return (
    <aside className="fixed inset-y-0 right-0 z-40 flex w-[min(620px,100vw)] flex-col border-l border-line bg-base shadow-[0_0_40px_rgba(0,0,0,.12)]"
      role="dialog" aria-label={`${page.plugin} page`}>
      <div className="flex items-center gap-2 border-b border-line bg-panel px-4 py-2.5">
        <PluginIcon name={page.plugin} />
        <span className="text-[14px] font-medium">{page.plugin}</span>
        {page.episode != null && <span className="text-[12px] text-faint">attempt {page.episode}</span>}
        <button onClick={onClose} className="ml-auto rounded-md px-2 py-0.5 text-[13px] text-dim hover:bg-raised hover:text-ink" aria-label="Close">
          Close <kbd>Esc</kbd>
        </button>
      </div>
      <iframe key={`${page.plugin}-${q}`} aria-label={`${page.plugin} page`} src={`/plugin-ui/${encodeURIComponent(page.plugin)}/?${q}`}
        sandbox="allow-scripts allow-downloads" className="w-full flex-1 border-0 bg-base" />
    </aside>
  )
}

/** Every plugin has a page: its own (manifest ui) or the standard one the server gives it (§7a). */
export function pagesOf(plugins: { name: string }[] | undefined): Set<string> {
  return new Set((plugins ?? []).map((p) => p.name))
}

// ---------------------------------------------------------------------------------------------------------------
// Sparklines on a canvas, redrawn from the live store (no React re-render per point).

export function Sparkline({ live, field, label, unit, color, scale = 1, max, warnAbove }: {
  live: LiveStore
  field: string
  label: string
  unit: string
  color: string
  scale?: number
  max?: number
  warnAbove?: number
}) {
  const canvas = useRef<HTMLCanvasElement>(null)
  const value = useRef<HTMLSpanElement>(null)
  useEffect(() => {
    let last = 0
    const draw = () => {
      const now = performance.now()
      if (now - last < 150) return // ~6 redraws a second is plenty for a trend
      last = now
      const c = canvas.current
      if (!c) return
      const ctx = c.getContext('2d')!
      const w = (c.width = c.clientWidth * devicePixelRatio)
      const h = (c.height = c.clientHeight * devicePixelRatio)
      const pts = live.points.filter((p) => p.v[field] != null)
      ctx.clearRect(0, 0, w, h)
      if (pts.length < 2) return
      const window = 20 // seconds shown
      const t1 = pts[pts.length - 1].t
      const t0 = Math.max(0, t1 - window)
      const ys = pts.map((p) => p.v[field] * scale)
      const top = max ?? Math.max(...ys, (warnAbove ?? 0) * 1.1, 1e-6)
      const x = (t: number) => ((t - t0) / window) * w
      const y = (v: number) => h - 4 * devicePixelRatio - (Math.max(0, v) / top) * (h - 8 * devicePixelRatio)
      if (warnAbove != null && warnAbove < top) {
        ctx.strokeStyle = 'rgba(183,121,31,.5)'
        ctx.setLineDash([4 * devicePixelRatio, 4 * devicePixelRatio])
        ctx.beginPath(); ctx.moveTo(0, y(warnAbove)); ctx.lineTo(w, y(warnAbove)); ctx.stroke()
        ctx.setLineDash([])
      }
      ctx.strokeStyle = color
      ctx.lineWidth = 1.75 * devicePixelRatio
      ctx.beginPath()
      pts.forEach((p, i) => {
        if (p.t < t0) return
        const px = x(p.t), py = y(p.v[field] * scale)
        if (i === 0) ctx.moveTo(px, py)
        else ctx.lineTo(px, py)
      })
      ctx.stroke()
      if (value.current) value.current.textContent = (ys[ys.length - 1]).toFixed(unit === 'ms' ? 0 : 1)
    }
    draw()
    return live.subscribe(draw)
  }, [live, field, color, scale, max, warnAbove, unit])
  return (
    <div className="min-w-0">
      <div className="flex items-baseline justify-between text-[12px]">
        <span className="text-dim">{label}</span>
        <span className="text-ink"><span ref={value}>–</span> <span className="text-faint">{unit}</span></span>
      </div>
      <canvas ref={canvas} className="mt-1 h-10 w-full" />
    </div>
  )
}

// ---------------------------------------------------------------------------------------------------------------
// "…": the secondary actions of a row or a page, out of the way until asked for.

export interface MenuItem { label: string; onSelect: () => void; tone?: 'bad'; href?: string }

export function MoreMenu({ items, label = 'More actions' }: { items: MenuItem[]; label?: string }) {
  const [open, setOpen] = useState(false)
  const box = useRef<HTMLDivElement>(null)
  useEffect(() => {
    if (!open) return
    const close = (e: MouseEvent | KeyboardEvent) => {
      if (e instanceof KeyboardEvent ? e.key === 'Escape' : !box.current?.contains(e.target as Node)) setOpen(false)
    }
    window.addEventListener('mousedown', close)
    window.addEventListener('keydown', close)
    return () => { window.removeEventListener('mousedown', close); window.removeEventListener('keydown', close) }
  }, [open])
  if (!items.length) return null
  return (
    <div ref={box} className="relative" onClick={(e) => e.stopPropagation()}>
      <button aria-label={label} aria-expanded={open} onClick={() => setOpen(!open)}
        className="flex h-8 w-8 items-center justify-center rounded-lg text-[18px] leading-none text-dim hover:bg-raised hover:text-ink">…</button>
      {open && (
        <div role="menu" className="absolute right-0 z-30 mt-1 min-w-[180px] rounded-xl border border-line bg-panel py-1 shadow-[0_4px_16px_rgba(23,24,26,.08)]">
          {items.map((it) => it.href ? (
            <a key={it.label} role="menuitem" href={it.href} target="_blank" rel="noreferrer" onClick={() => setOpen(false)}
              className="block px-3.5 py-1.5 text-[14px] hover:bg-raised">{it.label}</a>
          ) : (
            <button key={it.label} role="menuitem" onClick={() => { setOpen(false); it.onSelect() }}
              className={clsx('block w-full px-3.5 py-1.5 text-left text-[14px] hover:bg-raised', it.tone === 'bad' && 'text-bad')}>{it.label}</button>
          ))}
        </div>
      )}
    </div>
  )
}

/** Secondary content, folded: a title line that opens it. */
export function Fold({ title, children, open }: { title: ReactNode; children: ReactNode; open?: boolean }) {
  return (
    <details open={open} className="group rounded-xl border border-line bg-panel">
      <summary className="flex cursor-pointer list-none items-center gap-2 px-5 py-3.5 text-[15px] font-semibold select-none">
        <span className="text-faint transition-transform group-open:rotate-90" aria-hidden>›</span>{title}
      </summary>
      <div className="border-t border-line">{children}</div>
    </details>
  )
}
