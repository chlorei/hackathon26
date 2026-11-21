import { useEffect, useMemo, useRef, useState, type KeyboardEvent } from 'react'
import { useQuery } from '@tanstack/react-query'
import clsx from 'clsx'
import { CameraOff, FolderOpen } from 'lucide-react'
import { api, atOf, REASON_LABEL, tOf, type EpisodeResult, type EventRecord } from '../api'
import { useLive, useRunStream, type LiveStore } from '../useRunStream'
import { AttemptStrip, Button, ICON_BY_PLUGIN, LEVEL_TONE, Outcome, pagesOf, Panel, PanelTitle, PluginIcon, PluginPageDrawer, Sparkline, Tag, type Mark, type PluginPage } from '../ui'

const CAMS: Record<string, string> = { top: 'Overhead', left: 'Left wrist', right: 'Right wrist' }

const SNIPPET = `import dream, dreamkit

client = dreamkit.wrap(dream.DreamClient(), config="dreamkit.yaml")
client.policy({"checkpoint": "dreamlabs/dream-1-beta"})

with client.episode("put apple into the bowl", env=env):
    while client.t < 60:
        client.set_observation(frames, observation)
        reply = client.get_action()
        for left, right in rows(reply):
            env.step(left, right)
            client.step(env, images=env.images(), command=(left, right))`

/** Watches whatever run the user's loop is writing. It never starts anything. */
export default function LivePage({ onOpenRun }: { onOpenRun: (id: string, episode?: number) => void }) {
  const live = useQuery({ queryKey: ['live'], queryFn: api.live, refetchInterval: 2000 })
  const plugins = useQuery({ queryKey: ['plugins'], queryFn: api.plugins })
  for (const p of plugins.data?.plugins ?? []) ICON_BY_PLUGIN[p.name] = p.icon
  const pages = useMemo(() => pagesOf(plugins.data?.plugins), [plugins.data])
  const [page, setPage] = useState<PluginPage | null>(null)
  const [runId, setRunId] = useState<string | null>(null)
  useEffect(() => {
    const id = live.data?.id
    if (id && id !== runId) setRunId(id) // a newer run took over
  }, [live.data?.id, runId])

  const stream = useRunStream(runId)
  const running = Boolean(live.data?.id) && live.data?.id === runId
  const current = stream.current
  const marks = useMemo<Mark[]>(() => (current ? marksFrom(stream.events, current.episode) : []), [stream.events, current])

  if (!runId) return <Waiting />

  return (
    <div className="grid gap-4 xl:grid-cols-[minmax(0,1fr)_380px]">
      <div className="min-w-0 space-y-4">
        <Cameras live={stream.live} running={running} />
        <Panel>
          <PanelTitle right={current && <LiveClock live={stream.live} />}>
            {current ? `Attempt ${current.episode}, scene ${current.scene + 1}${current.attempt > 1 ? `, retry ${current.attempt - 1}` : ''}` : running ? 'Between attempts' : 'This run has ended'}
          </PanelTitle>
          <div className="px-4 pb-3">
            {current ? <LiveStrip live={stream.live} marks={marks} /> : (
              <p className="pb-2 text-[13px] text-faint">{running ? 'The next attempt draws here as soon as your loop starts it.' : 'Open it in History to watch the attempts back.'}</p>
            )}
          </div>
          <div className="grid grid-cols-2 gap-x-6 gap-y-3 border-t border-line px-4 py-3 md:grid-cols-4">
            {stream.mode?.startsWith('robot') ? (
              <>
                <Sparkline live={stream.live} field="grip" label="Gripper opening" unit="" scale={1} color="#2f8fb3" max={1} />
                <Sparkline live={stream.live} field="track" label="Tracking error" unit="rad" scale={1} color="#c8402f" warnAbove={0.12} />
                <div className="text-[12px]"><span className="text-dim">Positions</span><p className="mt-2 text-faint">Robot only: no world</p></div>
              </>
            ) : (
              <>
                <Sparkline live={stream.live} field="d" label="Hand to object" unit="cm" scale={100} color="#2f8fb3" />
                <Sparkline live={stream.live} field="z" label="Object lifted" unit="cm" scale={100} color="#1f8a5b" max={20} />
                <Sparkline live={stream.live} field="dt" label="Object to target" unit="cm" scale={100} color="#8a5cc2" />
              </>
            )}
            {live.data?.mock ? (
              <div className="text-[12px]"><span className="text-dim">Model round trip</span><p className="mt-2 text-faint">Mock: no model calls</p></div>
            ) : <Sparkline live={stream.live} field="rtt" label="Model round trip" unit="ms" color="#d9822b" warnAbove={700} />}
          </div>
        </Panel>
        <Attempts episodes={stream.episodes} current={current} onOpen={(ep) => onOpenRun(runId, ep)} />
      </div>

      <div className="space-y-4 xl:sticky xl:top-[68px] xl:self-start">
        <Panel>
          <div className="flex items-start gap-3 px-4 py-3">
            <span className={clsx('mt-1.5 h-2 w-2 shrink-0 rounded-full', running ? 'animate-pulse bg-signal' : 'bg-faint')} />
            <div className="min-w-0 flex-1">
              <p className="text-[14px]">{stream.task || live.data?.task || 'Run'}</p>
              {stream.mode && <p className="text-[12px] text-dim">{stream.mode}</p>}
              <p className="text-[12px] text-dim">
                {running ? 'Your loop is running' : 'Finished'}{live.data?.mock ? ', mock policy' : live.data?.model ? `, ${live.data.model}` : ''}
              </p>
            </div>
            {live.data?.mock && <Tag tone="warn" title="The mock policy fails on purpose so the plugins can be tested; no model is called">mock: scripted failures</Tag>}
          </div>
          <div className="flex gap-2 border-t border-line px-4 py-2.5">
            <Button variant="quiet" onClick={() => api.reveal(runId)}><FolderOpen className="h-4 w-4" /> Open folder</Button>
            <Button variant="quiet" onClick={() => onOpenRun(runId)}>Open in History</Button>
          </div>
        </Panel>
        <Feed events={stream.events} onWatch={(ep) => onOpenRun(runId, ep)} pages={pages}
          onPage={(plugin, episode) => setPage({ plugin, runId, episode })} />
        <PluginPageDrawer page={page} onClose={() => setPage(null)} />
      </div>
    </div>
  )
}

function Waiting() {
  const [copied, setCopied] = useState(false)
  return (
    <div className="mx-auto max-w-[760px] pt-10">
      <div className="flex items-center gap-3">
        <span className="h-2.5 w-2.5 animate-pulse rounded-full bg-signal" />
        <h1 className="text-[22px] font-semibold">Waiting for a run</h1>
      </div>
      <p className="mt-2 max-w-[62ch] text-[15px] leading-relaxed text-dim">
        Start your loop with <code className="rounded bg-raised px-1.5 py-0.5 text-ink">dreamkit.wrap(...)</code>. This page picks it up on its
        own and shows the cameras, what every plugin says, and each attempt as it happens.
      </p>
      <Panel className="mt-6">
        <PanelTitle right={
          <button className="text-[12px] text-dim hover:text-ink" onClick={() => { void navigator.clipboard.writeText(SNIPPET); setCopied(true) }}>
            {copied ? 'Copied' : 'Copy'}
          </button>
        }>your_loop.py</PanelTitle>
        <pre className="overflow-x-auto border-t border-line px-4 py-3 font-mono text-[12.5px] leading-relaxed text-ink">{SNIPPET}</pre>
      </Panel>
      <p className="mt-4 text-[13px] text-dim">
        Or try it on Dream Labs' simulator without credits: <code className="rounded bg-raised px-1.5 py-0.5 text-ink">python examples/sim_loop.py --mock</code>
      </p>
    </div>
  )
}

function marksFrom(events: EventRecord[], episode: number): Mark[] {
  return events
    .filter((e) => e.episode === episode && tOf(e) != null && e.plugin !== 'dreamkit')
    .filter((e) => e.plugin !== 'failure-monitor' || e.data.kind)
    .map((e) => ({ t: tOf(e)!, label: `${e.plugin}: ${e.message}`, tone: e.level === 'error' ? 'bad' : e.level === 'warn' ? 'warn' : e.level === 'success' ? 'ok' : 'info' }))
}

/** One big camera and two small ones; click a small one to swap. Frames are written straight into <img>. */
function Cameras({ live, running }: { live: LiveStore; running: boolean }) {
  const [main, setMain] = useState('top')
  const others = Object.keys(CAMS).filter((c) => c !== main)
  return (
    <div className="grid gap-3 md:grid-cols-[minmax(0,3fr)_minmax(0,1fr)]">
      <CameraView live={live} cam={main} running={running} large />
      <div className="grid grid-cols-2 content-start gap-3 md:grid-cols-1">
        {others.map((c) => (
          <button key={c} onClick={() => setMain(c)} className="block text-left" title={`Show ${CAMS[c].toLowerCase()} large`}>
            <CameraView live={live} cam={c} running={running} />
          </button>
        ))}
      </div>
    </div>
  )
}

function CameraView({ live, cam, running, large }: { live: LiveStore; cam: string; running: boolean; large?: boolean }) {
  const img = useRef<HTMLImageElement>(null)
  const [has, setHas] = useState(Boolean(live.frames[cam]))
  useEffect(() => {
    const paint = () => {
      const src = live.frames[cam]
      if (!src || document.hidden) return
      if (img.current && img.current.src !== src) img.current.src = src
      setHas(true)
    }
    paint()
    return live.subscribe(paint)
  }, [live, cam])
  return (
    <div className={clsx('relative aspect-video overflow-hidden rounded-lg border border-line bg-black', !large && 'hover:border-dim')}>
      <img ref={img} alt={CAMS[cam]} className={clsx('h-full w-full object-cover', !has && 'hidden')} />
      {!has && (
        <div className="flex h-full flex-col items-center justify-center gap-1.5 text-faint">
          <CameraOff className="h-5 w-5" strokeWidth={1.5} />
          <span className="text-[12px]">{running ? 'Waiting for frames' : 'Cameras show here during a run'}</span>
        </div>
      )}
      <span className={clsx('absolute left-2 top-2 flex items-center gap-1.5 rounded-md bg-black/60 px-1.5 py-0.5 text-[12px] text-white', large && 'text-[13px]')}>
        {running && has && <span className="h-1.5 w-1.5 rounded-full bg-signal" />}
        {CAMS[cam]}
      </span>
    </div>
  )
}

function LiveClock({ live }: { live: LiveStore }) {
  useLive(live)
  return <span className="text-[13px] text-signal">{live.t.toFixed(1)} s</span>
}

function LiveStrip({ live, marks }: { live: LiveStore; marks: Mark[] }) {
  useLive(live)
  const span = Math.max(30, Math.ceil((live.t + 6) / 10) * 10) // the strip grows with the attempt
  return <AttemptStrip duration={span} marks={marks} playhead={live.t} live />
}

function Attempts({ episodes, current, onOpen }: {
  episodes: EpisodeResult[]
  current: { episode: number; scene: number; attempt: number } | null
  onOpen: (episode: number) => void
}) {
  if (!episodes.length && !current) return null
  const ok = episodes.filter((e) => e.success).length
  return (
    <Panel>
      <PanelTitle right={<span className="text-[13px] text-dim">{ok} of {episodes.length} worked, {new Set(episodes.map((e) => e.scene)).size} scenes so far</span>}>
        Attempts
      </PanelTitle>
      <ul className="divide-y divide-line border-t border-line">
        {episodes.map((e) => (
          <li key={e.episode}>
            <button onClick={() => onOpen(e.episode)} className="flex w-full items-center gap-3 px-4 py-2.5 text-left hover:bg-raised/50">
              <Outcome success={e.success} />
              <span className="w-24 shrink-0 text-[13px] text-dim">Scene {e.scene + 1}{e.attempt > 1 ? `, retry ${e.attempt - 1}` : ''}</span>
              <span className={clsx('min-w-0 flex-1 truncate text-[13px]', e.success ? 'text-ink' : 'text-bad')}>
                {e.success ? `Done in ${e.duration_s.toFixed(1)} s` : e.recap?.headline?.replace(/^Attempt \d+: /, '') ?? `${e.message ?? REASON_LABEL[e.reason ?? 'unknown']}${atOf(e) != null ? ` at ${atOf(e)!.toFixed(1)} s` : ''}`}
              </span>
              <span className="shrink-0 text-[12px] text-steel">{e.success ? 'Replay' : 'Watch the failure'}</span>
            </button>
          </li>
        ))}
        {current && (
          <li className="flex items-center gap-3 px-4 py-2.5 text-[13px]">
            <span className="h-2 w-2 animate-pulse rounded-full bg-signal" />
            <span className="text-dim">Scene {current.scene + 1}{current.attempt > 1 ? `, retry ${current.attempt - 1}` : ''} is running</span>
          </li>
        )}
      </ul>
    </Panel>
  )
}

function Feed({ events, onWatch, pages, onPage }: {
  events: EventRecord[]
  onWatch: (episode: number) => void
  pages: Set<string>  // plugins with a page of their own (§7a): their lines open it
  onPage: (plugin: string, episode: number | null) => void
}) {
  const [only, setOnly] = useState<'all' | 'problems'>('all')
  const box = useRef<HTMLDivElement>(null)
  const shown = useMemo(() => (only === 'problems' ? events.filter((e) => e.level === 'warn' || e.level === 'error') : events).slice(-150), [events, only])
  useEffect(() => {
    const el = box.current
    if (el && el.scrollHeight - el.scrollTop - el.clientHeight < 120) el.scrollTop = el.scrollHeight
  }, [shown.length])
  return (
    <Panel className="flex h-[calc(100vh-460px)] min-h-[320px] flex-col">
      <PanelTitle right={
        <div className="flex rounded-md bg-base p-0.5 text-[12px]">
          {(['all', 'problems'] as const).map((k) => (
            <button key={k} onClick={() => setOnly(k)} className={clsx('rounded px-2 py-0.5', only === k ? 'bg-raised text-ink' : 'text-faint')}>
              {k === 'all' ? 'All' : 'Problems'}
            </button>
          ))}
        </div>
      }>What the plugins say</PanelTitle>
      <div ref={box} className="flex-1 overflow-y-auto border-t border-line">
        {shown.length === 0 && <p className="px-4 py-8 text-center text-[13px] text-faint">Plugins report here as they see things happen.</p>}
        {shown.map((e, i) => {
          const verdict = e.plugin === 'failure-monitor' && e.level === 'error' && Boolean(e.data.reason)
          return (
            <div key={`${e.ts}-${i}`} className={clsx('flex gap-2.5 border-b border-line/60 px-4 py-2', verdict && 'bg-bad-soft',
              pages.has(e.plugin) && 'cursor-pointer hover:bg-raised/50')}
              {...(pages.has(e.plugin) ? {
                role: 'button', tabIndex: 0, title: `Open ${e.plugin}'s page`,
                onClick: () => onPage(e.plugin, e.episode ?? null),
                onKeyDown: (k: KeyboardEvent) => { if (k.key === 'Enter' || k.key === ' ') { k.preventDefault(); onPage(e.plugin, e.episode ?? null) } },
              } : {})}>
              <PluginIcon name={e.plugin} className={clsx('mt-0.5', LEVEL_TONE[e.level])} />
              <div className="min-w-0 flex-1">
                <p className={clsx('text-[13px]', verdict ? 'font-medium text-bad' : e.level === 'info' ? 'text-ink' : LEVEL_TONE[e.level])}>{e.message}</p>
                <p className="text-[11px] text-faint">{e.plugin}{tOf(e) != null ? `, ${tOf(e)!.toFixed(1)} s` : ''}{e.episode ? ` in attempt ${e.episode}` : ''}</p>
                {verdict && e.episode != null && (
                  <button onClick={() => onWatch(e.episode!)} className="mt-1 text-[12px] text-signal hover:underline">Watch from {Math.max(0, Number(e.data.at_s ?? e.data.at) - 1).toFixed(1)} s</button>
                )}
              </div>
              {pages.has(e.plugin) && <span className="self-center text-[16px] text-faint" aria-hidden>›</span>}
            </div>
          )
        })}
      </div>
    </Panel>
  )
}
