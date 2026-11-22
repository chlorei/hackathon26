import { useMemo, useRef, useState, type KeyboardEvent, type ReactNode } from 'react'
import clsx from 'clsx'
import { ChevronDown, ChevronRight } from 'lucide-react'
import { recoveriesOf, tOf, type EpisodeResult, type EventRecord } from '../api'
import { LEVEL_TONE, Panel, PanelTitle, PluginIcon } from '../ui'

/** One line of the feed: a single event, or the same event repeated (collapsed, with a count and the mean). */
type Line = { key: string; plugin: string; level: EventRecord['level']; text: string; t: number | null; episode: number | null; events: EventRecord[] }

const NUM = /-?\d+(?:\.\d+)?/g

/** "model round trip 2347 ms, slow" ×15 -> "model round trip avg 2.3 s, slow". An average means something only for
a latency; anything else (a count "(2/3)", a second "at 14.9 s") shows its latest message. */
function collapsed(template: string, events: EventRecord[]): string {
  if (!/# ms\b/.test(template)) return events[events.length - 1].message
  const ms = events.map((e) => Number((e.message.match(/(-?\d+(?:\.\d+)?) ms\b/) ?? [])[1])).filter((v) => !Number.isNaN(v))
  const avg = ms.reduce((a, v) => a + v, 0) / Math.max(1, ms.length)
  const shown = avg >= 1000 ? `avg ${(avg / 1000).toFixed(1)} s` : `avg ${Math.round(avg)} ms`
  return template.replace(/# ms\b/, shown).replace(/#/g, '…')
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
        <span className="shrink-0 whitespace-nowrap">{open ? '▾' : '▸'} {n} technical {n === 1 ? 'event' : 'events'}</span> <span className="min-w-0 truncate">({[...new Set(lines.map((l) => l.text.replace(/ avg .*/, '').slice(0, 28)))].join(', ')})</span>
      </button>
      {open && lines.map((line) => <FeedLine key={line.key} line={line} pages={pages} onPage={onPage} onWatch={onWatch} />)}
    </div>
  )
}

/** What to do when attempts keep failing the same way. */
const STREAK_HINT: Record<string, string> = {
  model_error: 'the model keeps answering with an error: check the Dream API key, the credits or the network',
  slow_model: 'the model keeps answering too slowly: check the network, or try a lighter model',
  camera_failure: 'a camera keeps failing: check its cable and what it sees',
}

/** Groups in feed order, with runs of 3+ finished attempts that failed for the same reason folded into one. */
type Item = { kind: 'group'; key: string; events: EventRecord[] } | { kind: 'streak'; key: string; reason: string; first: number; last: number }
function foldStreaks(groups: [string, EventRecord[]][], result: (ep: number) => EpisodeResult | undefined): Item[] {
  const out: Item[] = []
  let i = 0
  while (i < groups.length) {
    const r = groups[i][0] === 'run' ? undefined : result(Number(groups[i][0]))
    let j = i
    while (r && r.success === false && j + 1 < groups.length) {
      const n = groups[j + 1][0] === 'run' ? undefined : result(Number(groups[j + 1][0]))
      if (!n || n.success !== false || n.reason !== r.reason) break
      j++
    }
    if (r && j - i + 1 >= 3) {
      const eps = groups.slice(i, j + 1).map(([k]) => Number(k))
      out.push({ kind: 'streak', key: `streak-${eps[0]}`, reason: r.reason ?? 'failed', first: Math.min(...eps), last: Math.max(...eps) })
      i = j + 1
    } else {
      out.push({ kind: 'group', key: groups[i][0], events: groups[i][1] })
      i++
    }
  }
  return out
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
      const k = !e.episode ? 'run' : String(e.episode)
      if (!by.has(k)) by.set(k, [])
      by.get(k)!.push(e)
    }
    const all = [...by.entries()]
    return [...all.filter(([k]) => k !== 'run').reverse(), ...all.filter(([k]) => k === 'run')]  // newest attempt on top
  }, [shown])

  const result = (ep: number) => episodes.find((e) => e.episode === ep)
  const isOpen = (k: string) => open[k] ?? (k !== 'run' && ((running && k === String(current)) || !result(Number(k))))
  const chip = (k: string, label: ReactNode) => (
    <button key={k} onClick={() => setOnly(k)} aria-pressed={only === k}
      className={clsx('rounded-md px-2 py-0.5 text-[12px]', only === k ? 'bg-raised text-ink' : 'text-faint hover:text-ink')}>{label}</button>
  )

  return (
    <Panel className="flex h-[calc(100vh-260px)] min-h-[360px] flex-col">
      <PanelTitle>What the plugins say</PanelTitle>
      <div className="flex flex-wrap items-center gap-1 px-4 pb-2" role="toolbar" aria-label="Show">
        {chip('all', 'All')}{chip('problems', 'Problems')}
        {plugins.length > 0 && (
          <select aria-label="One plugin" value={plugins.includes(only) ? only : ''} onChange={(e) => setOnly(e.target.value || 'all')}
            className={clsx('ml-1 rounded-md border border-line bg-panel px-1.5 py-0.5 text-[12px]', plugins.includes(only) ? 'text-ink' : 'text-faint')}>
            <option value="">One plugin…</option>
            {plugins.map((p) => <option key={p} value={p}>{p}</option>)}
          </select>
        )}
      </div>
      <div ref={box} className="flex-1 overflow-y-auto border-t border-line">
        {groups.length === 0 && <p className="px-4 py-8 text-center text-[13px] text-faint">{events.length ? 'Nothing here with this filter.' : 'Plugins report here as they see things happen.'}</p>}
        {foldStreaks(groups, result).map((item) => {
          if (item.kind === 'streak') {
            const n = item.last - item.first + 1
            return (
              <section key={item.key} className="border-b border-line bg-bad-soft/40 px-4 py-2">
                <p className="text-[13px]"><span className="font-semibold">Attempts {item.first}–{item.last}</span>{' '}
                  <span className="text-bad">✗ {item.reason.replace(/_/g, ' ')} ×{n}</span></p>
                {STREAK_HINT[item.reason] && <p className="mt-0.5 text-[12px] text-dim">{n} in a row: {STREAK_HINT[item.reason]}</p>}
              </section>
            )
          }
          const [k, evs] = [item.key, item.events]
          const ep = k === 'run' ? null : Number(k)
          const r = ep == null ? undefined : result(ep)
          const live = ep != null && !r && running && ep === current
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
                {r && <span className="text-[12px] text-faint">{r.duration_s.toFixed(1)} s{typeof r.cost_usd === 'number' && r.cost_usd > 0 ? ` · $${r.cost_usd.toFixed(2)}` : ''}</span>}
                {r && r.success && recoveriesOf(r) > 0 && <span className="text-[12px] text-signal">saved by recovery</span>}
                <span className="ml-auto text-[11px] text-faint">{evs.length}</span>
              </button>
              {pinned.length > 0 && expanded && (
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
