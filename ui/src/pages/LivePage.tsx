import { useEffect, useMemo, useRef, useState, type ReactNode } from 'react'
import { useQuery } from '@tanstack/react-query'
import clsx from 'clsx'
import { CameraOff, FolderOpen } from 'lucide-react'
import { api, atOf, REASON_LABEL, tOf, type EpisodeResult, type EventRecord } from '../api'
import { useLive, useRunStream, type LiveStore } from '../useRunStream'
import Feed from './LiveFeed'
import { AttemptStrip, Button, ICON_BY_PLUGIN, Outcome, pagesOf, Panel, PanelTitle, PluginPageDrawer, Sparkline, Tag, type Mark, type PluginPage } from '../ui'

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

  const [showCams, setShowCams] = useState(() => localStorage.getItem('dk-cams') === '1')
  const stream = useRunStream(runId, showCams)
  const running = Boolean(live.data?.id) && live.data?.id === runId
  const current = stream.current
  const marks = useMemo<Mark[]>(() => (current ? marksFrom(stream.events, current.episode) : []), [stream.events, current])
  const toggleCams = () => { localStorage.setItem('dk-cams', showCams ? '0' : '1'); setShowCams(!showCams) }

  if (!runId) return <Waiting />

  return (
    <div className="space-y-4">
    <StatusBar task={stream.task || live.data?.task || 'Run'} mode={stream.mode} running={running} mock={Boolean(live.data?.mock)}
      model={live.data?.model} current={current} episodes={stream.episodes} live={stream.live}
      onFolder={() => api.reveal(runId)} onHistory={() => onOpenRun(runId)} />
    <div className="grid gap-4 xl:grid-cols-[minmax(0,1fr)_400px]">
      <div className="min-w-0 space-y-4">
        <NowCard events={stream.events} episodes={stream.episodes} current={current} running={running} live={stream.live} />
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
        <Panel>
          <PanelTitle right={<Button variant="quiet" onClick={toggleCams}>{showCams ? 'Hide cameras' : 'Show cameras'}</Button>}>
            Cameras
          </PanelTitle>
          {showCams ? <div className="px-4 pb-4"><Cameras live={stream.live} running={running} /></div> : (
            <p className="px-5 pb-4 text-dim">Hidden: you watch the arm itself (the robot, or the simulator window). Open them to check
              what the model sees; the loop renders frames for the panel only while they are open.</p>
          )}
        </Panel>
      </div>

      <div className="space-y-4 xl:sticky xl:top-[68px] xl:self-start">
        <Feed events={stream.events} episodes={stream.episodes} current={current?.episode ?? null} running={running}
          onWatch={(ep) => onOpenRun(runId, ep)} pages={pages} onPage={(plugin, episode) => setPage({ plugin, runId, episode })} />
        <PluginPageDrawer page={page} onClose={() => setPage(null)} />
      </div>
    </div>
    </div>
  )
}

/** The run in one line: task, attempt, clock, successes, the model's mean round trip, money spent when known. */
function StatusBar({ task, mode, running, mock, model, current, episodes, live, onFolder, onHistory }: {
  task: string; mode?: string; running: boolean; mock: boolean; model?: string
  current: { episode: number; scene: number; attempt: number } | null
  episodes: EpisodeResult[]; live: LiveStore
  onFolder: () => void; onHistory: () => void
}) {
  const ok = episodes.filter((e) => e.success).length
  const lat = episodes.map((e) => ((e as unknown as Record<string, unknown>)['failure-monitor'] as { latency?: { mean_ms?: number; calls?: number } } | undefined)?.latency)
    .filter((l): l is { mean_ms: number; calls: number } => Boolean(l && l.mean_ms != null && l.calls))
  const calls = lat.reduce((a, l) => a + l.calls, 0)
  const meanMs = calls ? lat.reduce((a, l) => a + l.mean_ms * l.calls, 0) / calls : null
  const item = (label: string, value: ReactNode, title?: string) => (
    <div className="min-w-0" title={title}>
      <p className="text-[11px] tracking-wide text-faint uppercase">{label}</p>
      <p className="truncate text-[15px] font-semibold tabular-nums">{value}</p>
    </div>
  )
  return (
    <Panel>
      <div className="flex flex-wrap items-center gap-x-8 gap-y-3 px-5 py-3">
        <div className="flex min-w-[220px] flex-1 items-start gap-2.5">
          <span className={clsx('mt-2 h-2 w-2 shrink-0 rounded-full', running ? 'animate-pulse bg-signal' : 'bg-faint')} />
          <div className="min-w-0">
            <p className="truncate text-[15px] font-semibold">{task}</p>
            <p className="text-[12px] text-dim">{running ? 'running' : 'finished'}{mock ? ' · mock policy' : model ? ` · ${model}` : ''}{mode ? ` · ${mode}` : ''}</p>
          </div>
          {mock && <Tag tone="warn" title="The mock policy fails on purpose so the plugins can be tested; no model is called">mock</Tag>}
        </div>
        {item('Attempt', current ? `${current.episode}${current.attempt > 1 ? ` (retry ${current.attempt - 1})` : ''} · scene ${current.scene + 1}` : '–')}
        {item('Time', running && current ? <LiveClock live={live} /> : '–')}
        {item('Succeeded', `${ok} / ${episodes.length}`)}
        {item('Model round trip', meanMs == null ? (mock ? 'mock' : '–') : meanMs >= 1000 ? `${(meanMs / 1000).toFixed(1)} s avg` : `${Math.round(meanMs)} ms avg`, calls ? `${calls} calls` : undefined)}
        <Spent live={live} episodes={episodes} mock={mock} running={running} />
        <div className="ml-auto flex gap-1">
          <Button variant="quiet" onClick={onFolder}><FolderOpen className="h-4 w-4" /> Folder</Button>
          <Button variant="quiet" onClick={onHistory}>History</Button>
        </div>
      </div>
    </Panel>
  )
}

/** What is happening right now, in one glance: the attempt, the clock, and the last thing that mattered. */
function NowCard({ events, episodes, current, running, live }: {
  events: EventRecord[]
  episodes: EpisodeResult[]
  current: { episode: number; scene: number; attempt: number } | null
  running: boolean
  live: LiveStore
}) {
  useLive(live)
  const ep = current?.episode
  const mine = events.filter((e) => e.episode === ep && e.plugin !== 'dreamkit')
  const verdict = [...mine].reverse().find((e) => e.data?.kind === 'verdict' || (e.level === 'error' && e.data?.reason))
  const recovering = [...mine].reverse().find((e) => e.plugin === 'recovery' && e.data?.action)
  const last = [...mine].reverse().find((e) => e.level !== 'info' || e.plugin === 'failure-monitor')
  const done = episodes.length
  const ok = episodes.filter((e) => e.success).length
  const tone = !current ? 'idle' : recovering && (!verdict || (tOf(recovering) ?? 0) >= (tOf(verdict) ?? 0)) ? 'recover'
    : verdict ? 'bad' : 'ok'
  const headline = !current ? (running ? 'Between attempts' : 'Run finished')
    : tone === 'recover' ? recovering!.message
    : verdict ? verdict.message.replace(/^verdict: /, '')
    : last ? last.message : 'Running, nothing wrong so far'
  return (
    <Panel className={clsx('px-6 py-5', tone === 'bad' && 'border-bad/50', tone === 'recover' && 'border-signal/50')}>
      <div className="flex flex-wrap items-baseline gap-x-4 gap-y-1">
        <span className="text-[13px] font-semibold uppercase tracking-[.06em] text-dim">
          {current ? `Attempt ${current.episode} · scene ${current.scene + 1}${current.attempt > 1 ? ` · retry ${current.attempt - 1}` : ''}` : 'Now'}
        </span>
        {current && <span className="text-[15px] font-semibold text-signal">{live.t.toFixed(1)} s</span>}
        <span className="ml-auto text-dim">{ok} of {done} attempts worked so far</span>
      </div>
      <p className={clsx('mt-2 text-[26px] font-semibold leading-tight tracking-[-0.01em]',
        tone === 'bad' && 'text-bad', tone === 'recover' && 'text-signal', tone === 'ok' && 'text-ink', tone === 'idle' && 'text-dim')}>
        {headline}
      </p>
      {current && (
        <p className="mt-1 text-dim">
          {tone === 'recover' ? 'recovery is fixing it from where the robot is' : verdict ? `${verdict.plugin} · ${tOf(verdict)?.toFixed(1)} s` : last ? `${last.plugin} · ${tOf(last)?.toFixed(1)} s` : 'plugins are watching'}
        </p>
      )}
    </Panel>
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

/** Money as it goes (dreamkit.cost, telemetry spent_usd); with a budget (budget-guard's budget_usd), how far in. */
function Spent({ live, episodes, mock, running }: { live: LiveStore; episodes: EpisodeResult[]; mock: boolean; running: boolean }) {
  useLive(live)
  const v = live.points.length ? live.points[live.points.length - 1].v : {}
  // what each attempt cost: the core's figure, or budget-guard's when it bills a mock for a test (count_mock)
  const costOf = (e: EpisodeResult) => e.cost_usd || ((e as unknown as Record<string, { attempt_usd?: number }>)['budget-guard']?.attempt_usd ?? 0)
  const done = episodes.reduce((a, e) => a + costOf(e), 0)
  const spent = Math.max(done, running && typeof v.spent_usd === 'number' ? v.spent_usd : 0)
  const budget = typeof v.budget_usd === 'number' && v.budget_usd > 0 ? v.budget_usd : null
  const perAttempt = episodes.length ? done / episodes.length : 0
  const left = budget && perAttempt > 0 ? Math.max(0, Math.floor((budget - spent) / perAttempt)) : null
  const share = budget ? spent / budget : 0
  return (
    <div className="min-w-[120px]" title={mock ? 'a mock run calls no model: nothing is billed' : 'model time, at the price per robot-hour in dreamkit.yaml'}>
      <p className="text-[11px] tracking-wide text-faint uppercase">Spent</p>
      <p className="text-[15px] font-semibold tabular-nums">${spent.toFixed(2)}
        {budget != null && <span className="text-[12px] font-normal text-dim"> of ${budget.toFixed(2)}{left != null ? ` · ~${left} attempts left` : ''}</span>}
        {mock && budget == null && <span className="text-[12px] font-normal text-dim"> mock</span>}
      </p>
      {budget != null && (
        <div className="mt-1 h-1.5 w-full overflow-hidden rounded bg-raised" role="progressbar" aria-valuenow={Math.round(share * 100)} aria-valuemin={0} aria-valuemax={100} aria-label="budget used">
          <div className={clsx('h-full rounded', share >= 0.95 ? 'bg-bad' : share >= 0.8 ? 'bg-warn' : 'bg-ok')} style={{ width: `${Math.min(100, share * 100)}%` }} />
        </div>
      )}
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
