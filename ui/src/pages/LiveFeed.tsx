import { useEffect, useMemo, useRef, useState, type KeyboardEvent, type ReactNode } from 'react'
import clsx from 'clsx'
import { ChevronDown, ChevronRight } from 'lucide-react'
import { tOf, type EpisodeResult, type EventRecord } from '../api'
import { LEVEL_TONE, Panel, PanelTitle, PluginIcon } from '../ui'

/** One line of the feed: a single event, or the same event repeated (collapsed, with a count and the mean). */
type Line = { key: string; plugin: string; level: EventRecord['level']; text: string; t: number | null; episode: number | null; events: EventRecord[] }

const NUM = /-?\d+(?:\.\d+)?/g

/** "model round trip 2347 ms, slow" ×15 -> "model round trip avg 2.3 s, slow" (the mean of the first number). */
function collapsed(template: string, events: EventRecord[]): string {
  const firsts = events.map((e) => Number((e.message.match(NUM) ?? [])[0])).filter((v) => !Number.isNaN(v))
  if (!firsts.length) return template.replace(/#/g, '…')
  const avg = firsts.reduce((a, v) => a + v, 0) / firsts.length
  if (/# ms\b/.test(template)) {
    const shown = avg >= 1000 ? `avg ${(avg / 1000).toFixed(1)} s` : `avg ${Math.round(avg)} ms`
    return template.replace(/# ms\b/, shown).replace(/#/g, '…')
  }
  const whole = firsts.every((v) => Number.isInteger(v))
  return template.replace('#', `avg ${whole && Number.isInteger(avg) ? avg : avg.toFixed(1)}`).replace(/#/g, '…')
}

/** Events of one attempt as lines: identical messages (numbers aside) of a plugin become one line. */
function linesOf(events: EventRecord[]): Line[] {
  const byKey = new Map<string, EventRecord[]>()
  const order: string[] = []
  for (const e of events) {
    const key = `${e.plugin}|${e.level}|${e.message.replace(NUM, '#')}`
    if (!byKey.has(key)) { byKey.set(key, []); order.push(key) }
    byKey.get(key)!.push(e)
  }
  return order.map((key) => {
    const evs = byKey.get(key)!
    const first = evs[0]
    const template = key.split('|').slice(2).join('|')
    return { key, plugin: first.plugin, level: first.level, t: tOf(first), episode: first.episode ?? null, events: evs,
      text: evs.length > 1 ? collapsed(template, evs) : first.message }
  })
}

/** Plumbing, not news: the loop's mode, the model's round trips. Folded into one line per attempt. */
const isTechnical = (l: Line) => l.plugin === 'dreamkit' || /model round trip/.test(l.text)

/** An attempt's technical lines behind one fold. */
function Technical({ lines, pages, onPage, onWatch }: {
  lines: Line[]; pages: Set<string>; onPage: (plugin: string, episode: number | null) => void; onWatch: (episode: number) => void
}) {
  const [open, setOpen] = useState(false)
  const n = lines.reduce((a, l) => a + l.events.length, 0)
  return (
    <div className="border-t border-line/50">
      <button onClick={() => setOpen(!open)} aria-expanded={open} className="flex w-full items-center gap-2 px-4 py-1.5 pl-9 text-left text-[12px] text-faint hover:text-dim">
        {open ? '▾' : '▸'} {n} technical {n === 1 ? 'event' : 'events'} <span className="truncate">({[...new Set(lines.map((l) => l.text.replace(/ avg .*/, '').slice(0, 28)))].join(', ')})</span>
      </button>
      {open && lines.map((line) => <FeedLine key={line.key} line={line} pages={pages} onPage={onPage} onWatch={onWatch} />)}
    </div>
  )
}

const isVerdict = (e: EventRecord) => e.plugin === 'failure-monitor' && e.level === 'error' && Boolean(e.data.reason)

export default function Feed({ events, episodes, current, running, onWatch, pages, onPage }: {
  events: EventRecord[]
  episodes: EpisodeResult[]
  current: number | null  // the attempt running now
  running: boolean
  onWatch: (episode: number) => void
  pages: Set<string>  // plugins with a page (§7a): their lines open it
  onPage: (plugin: string, episode: number | null) => void
}) {
  const [only, setOnly] = useState<string>('all')  // 'all' | 'problems' | a plugin's name
  const [open, setOpen] = useState<Record<string, boolean>>({})  // attempt -> expanded, when the user chose
  const box = useRef<HTMLDivElement>(null)
  const plugins = useMemo(() => [...new Set(events.map((e) => e.plugin).filter((p) => p !== 'dreamkit'))].sort(), [events])
  const shown = useMemo(() => events.filter((e) => only === 'all' ? true : only === 'problems' ? e.level === 'warn' || e.level === 'error' : e.plugin === only), [events, only])
  const groups = useMemo(() => {
    const by = new Map<string, EventRecord[]>()
    for (const e of shown) {
      const k = e.episode == null ? 'run' : String(e.episode)
      if (!by.has(k)) by.set(k, [])
      by.get(k)!.push(e)
    }
    return [...by.entries()]
  }, [shown])
  useEffect(() => {
    const el = box.current
    if (el && el.scrollHeight - el.scrollTop - el.clientHeight < 160) el.scrollTop = el.scrollHeight
  }, [shown.length])

  const result = (ep: number) => episodes.find((e) => e.episode === ep)
  const isOpen = (k: string) => open[k] ?? (k === 'run' || (running && k === String(current)) || !result(Number(k)))
  const chip = (k: string, label: ReactNode) => (
    <button key={k} onClick={() => setOnly(k)} aria-pressed={only === k}
      className={clsx('rounded-md px-2 py-0.5 text-[12px]', only === k ? 'bg-raised text-ink' : 'text-faint hover:text-ink')}>{label}</button>
  )

  return (
    <Panel className="flex h-[calc(100vh-260px)] min-h-[360px] flex-col">
      <PanelTitle>What the plugins say</PanelTitle>
      <div className="flex flex-wrap items-center gap-1 px-4 pb-2" role="toolbar" aria-label="Show">
        {chip('all', 'All')}{chip('problems', 'Problems')}
        <span className="mx-1 h-4 w-px bg-line" aria-hidden />
        {plugins.map((p) => chip(p, <span className="flex items-center gap-1"><PluginIcon name={p} className="!h-3 !w-3" />{p}</span>))}
      </div>
      <div ref={box} className="flex-1 overflow-y-auto border-t border-line">
        {groups.length === 0 && <p className="px-4 py-8 text-center text-[13px] text-faint">{events.length ? 'Nothing here with this filter.' : 'Plugins report here as they see things happen.'}</p>}
        {groups.map(([k, evs]) => {
          const ep = k === 'run' ? null : Number(k)
          const r = ep == null ? undefined : result(ep)
          const live = ep != null && !r
          // pinned in the header: each plugin's last success/error say on this attempt
          const pinned = [...new Map(evs.filter((e) => e.level === 'success' || e.level === 'error').map((e) => [e.plugin, e])).values()]
          const expanded = isOpen(k)
          return (
            <section key={k} className="border-b border-line">
              <button onClick={() => setOpen({ ...open, [k]: !expanded })} aria-expanded={expanded}
                className={clsx('flex w-full items-center gap-2 px-4 py-2 text-left hover:bg-raised/40', live && 'bg-signal-soft/50')}>
                {expanded ? <ChevronDown className="h-3.5 w-3.5 text-faint" /> : <ChevronRight className="h-3.5 w-3.5 text-faint" />}
                <span className="text-[13px] font-semibold">{ep == null ? 'Run' : `Attempt ${ep}`}</span>
                {r && <span className={clsx('text-[12px]', r.success ? 'text-ok' : r.success === false ? 'text-bad' : 'text-dim')}>{r.success ? '✓ success' : r.success === false ? `✗ ${(r.reason ?? 'failed').replace(/_/g, ' ')}` : 'unknown'}</span>}
                {live && <span className="text-[12px] text-signal">running</span>}
                {r && <span className="text-[12px] text-faint">{r.duration_s.toFixed(1)} s{typeof r.cost_usd === 'number' ? ` · $${r.cost_usd.toFixed(2)}` : ''}</span>}
                <span className="ml-auto text-[11px] text-faint">{evs.length}</span>
              </button>
              {pinned.length > 0 && (
                <div className="flex flex-wrap gap-1.5 px-4 pb-2 pl-9">
                  {pinned.map((e) => (
                    <button key={e.plugin} onClick={() => onPage(e.plugin, e.episode ?? null)} title={`${e.plugin}: ${e.message}`}
                      className={clsx('flex max-w-full items-center gap-1 rounded-full border px-2 py-0.5 text-[11.5px]',
                        e.level === 'success' ? 'border-ok/30 bg-ok-soft text-ok' : 'border-bad/30 bg-bad-soft text-bad')}>
                      <PluginIcon name={e.plugin} className="!h-3 !w-3" />
                      <span className="truncate">{e.message.length > 48 ? e.message.slice(0, 47) + '…' : e.message}</span>
                    </button>
                  ))}
                </div>
              )}
              {expanded && (() => {
                const lines = linesOf(evs)
                const tech = lines.filter(isTechnical)
                return [...lines.filter((l) => !isTechnical(l)).map((line) => (
                  <FeedLine key={line.key} line={line} pages={pages} onPage={onPage} onWatch={onWatch} />
                )), tech.length ? <Technical key="tech" lines={tech} pages={pages} onPage={onPage} onWatch={onWatch} /> : null]
              })()}
            </section>
          )
        })}
      </div>
    </Panel>
  )
}

function FeedLine({ line, pages, onPage, onWatch }: {
  line: Line; pages: Set<string>; onPage: (plugin: string, episode: number | null) => void; onWatch: (episode: number) => void
}) {
  const [more, setMore] = useState(false)
  const e = line.events[0]
  const verdict = isVerdict(e)
  const many = line.events.length > 1
  const clickable = pages.has(line.plugin)
  const act = () => (many ? setMore(!more) : clickable && onPage(line.plugin, line.episode))
  return (
    <div className={clsx('border-t border-line/50', verdict && 'bg-bad-soft')}>
      <div role="button" tabIndex={0} title={many ? (more ? 'Hide the repeats' : 'Show every one') : clickable ? `Open ${line.plugin}'s page` : undefined}
        onClick={act} onKeyDown={(k: KeyboardEvent) => { if (k.key === 'Enter' || k.key === ' ') { k.preventDefault(); act() } }}
        className={clsx('flex gap-2.5 px-4 py-1.5 pl-9', (many || clickable) && 'cursor-pointer hover:bg-raised/50')}>
        <PluginIcon name={line.plugin} className={clsx('mt-0.5', LEVEL_TONE[line.level])} />
        <div className="min-w-0 flex-1">
          <p className={clsx('text-[13px]', verdict ? 'font-medium text-bad' : line.level === 'info' ? 'text-ink' : LEVEL_TONE[line.level])}>
            {line.text}{many && <span className="ml-1.5 rounded bg-raised px-1.5 text-[11px] text-dim">×{line.events.length}</span>}
          </p>
          <p className="text-[11px] text-faint">{line.plugin}{line.t != null ? `, ${many ? 'from ' : ''}${line.t.toFixed(1)} s` : ''}</p>
          {verdict && line.episode != null && (
            <button onClick={(ev) => { ev.stopPropagation(); onWatch(line.episode!) }} className="mt-0.5 text-[12px] text-signal hover:underline">
              Watch from {Math.max(0, Number(e.data.at_s ?? e.data.at) - 1).toFixed(1)} s
            </button>
          )}
        </div>
        {many ? <span className="self-center text-[12px] text-faint" aria-hidden>{more ? '▾' : '▸'}</span>
          : clickable && <span className="self-center text-[16px] text-faint" aria-hidden>›</span>}
      </div>
      {more && line.events.map((x, i) => (
        <div key={i} className="flex gap-2.5 px-4 py-1 pl-[60px] text-[12px] text-dim">
          <span className="w-12 shrink-0 text-right text-faint">{tOf(x) != null ? `${tOf(x)!.toFixed(1)} s` : ''}</span>
          <span className="min-w-0 flex-1">{x.message}</span>
          {clickable && <button onClick={() => onPage(x.plugin, x.episode ?? null)} className="text-faint hover:text-ink" aria-label={`Open ${x.plugin}'s page`}>›</button>}
        </div>
      ))}
    </div>
  )
}
