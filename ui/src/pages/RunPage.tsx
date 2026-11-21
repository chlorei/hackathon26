import { useEffect, useMemo, useRef, useState } from 'react'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import clsx from 'clsx'
import { CameraOff, FolderOpen, Play, Square } from 'lucide-react'
import { api, REASON_LABEL, type EpisodeResult, type EventRecord } from '../api'
import { useLive, useRunStream, type LiveStore } from '../useRunStream'
import { AttemptStrip, Button, Field, Input, LEVEL_TONE, Outcome, Panel, PanelTitle, PluginIcon, Sparkline, Toggle, type Mark } from '../ui'

const CAMS: Record<string, string> = { top: 'Overhead', left: 'Left wrist', right: 'Right wrist' }
const TASKS = ['put apple into the bowl', 'put orange into the bowl', 'put banana on the plate']

export default function RunPage({ onOpenRun }: { onOpenRun: (id: string, episode?: number) => void }) {
  const queryClient = useQueryClient()
  const config = useQuery({ queryKey: ['config'], queryFn: api.config })
  const runs = useQuery({ queryKey: ['runs'], queryFn: api.runs, refetchInterval: 5000 })
  const [runId, setRunId] = useState<string | null>(null)
  const [form, setForm] = useState({ task: '', model: '', episodes: 2, seconds: 40, mock: true, light: false })
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    if (config.data && !form.task) setForm((f) => ({ ...f, task: config.data.task, model: config.data.model, light: config.data.light }))
  }, [config.data, form.task])
  useEffect(() => {
    if (!runId && runs.data?.active) setRunId(runs.data.active)
  }, [runs.data?.active, runId])

  const stream = useRunStream(runId)
  const running = stream.status === 'running' || stream.status === 'connecting'
  const runPath = config.data && runId ? `${config.data.runs_path}/${runId}` : null

  const start = useMutation({
    mutationFn: () => api.startRun({
      task: form.task, model: form.model, episodes: form.episodes, seconds: form.seconds, mock: form.mock,
      overrides: { light: form.light },
    }),
    onSuccess: ({ id }) => { setError(null); setRunId(id); queryClient.invalidateQueries({ queryKey: ['runs'] }) },
    onError: (e: Error) => setError(e.message),
  })
  const stop = useMutation({ mutationFn: () => api.stopRun(runId!) })

  const current = stream.current
  const marks = useMemo<Mark[]>(() => (current ? marksFrom(stream.events, current.episode) : []), [stream.events, current])

  return (
    <div className="grid gap-4 xl:grid-cols-[minmax(0,1fr)_380px]">
      <div className="min-w-0 space-y-4">
        <Cameras live={stream.live} running={running} />

        <Panel>
          <PanelTitle right={current && <LiveClock live={stream.live} />}>
            {current ? `Attempt ${current.episode}, scene ${current.scene + 1}${current.attempt > 1 ? `, retry ${current.attempt - 1}` : ''}` : 'Attempt timeline'}
          </PanelTitle>
          <div className="px-4 pb-3">
            {current ? (
              <LiveStrip live={stream.live} seconds={form.seconds} marks={marks} />
            ) : (
              <p className="pb-2 text-[13px] text-faint">Each attempt draws here as it runs: reach, grasp, lift, release, and where it went wrong.</p>
            )}
          </div>
          <div className="grid grid-cols-2 gap-x-6 gap-y-3 border-t border-line px-4 py-3 md:grid-cols-4">
            <Sparkline live={stream.live} field="d" label="Hand to object" unit="cm" scale={100} color="#7fb2e5" />
            <Sparkline live={stream.live} field="z" label="Object lifted" unit="cm" scale={100} color="#4ade80" max={20} />
            <Sparkline live={stream.live} field="dt" label="Object to target" unit="cm" scale={100} color="#c4a7f5" />
            <Sparkline live={stream.live} field="rtt" label="Model round trip" unit="ms" color="#f0b43c" warnAbove={700} />
          </div>
        </Panel>

        <Attempts episodes={stream.episodes} planned={form.episodes} current={current} onOpen={(ep) => runId && onOpenRun(runId, ep)} />
      </div>

      <div className="space-y-4 xl:sticky xl:top-[68px] xl:self-start">
        <Panel>
          <PanelTitle>New run</PanelTitle>
          <div className="space-y-3 px-4 pb-4">
            <Field label="Task">
              <Input value={form.task} list="tasks" disabled={running} onChange={(e) => setForm({ ...form, task: e.target.value })} />
              <datalist id="tasks">{TASKS.map((t) => <option key={t} value={t} />)}</datalist>
            </Field>
            <div className="grid grid-cols-2 gap-3">
              <Field label="Attempts"><Input type="number" min={1} max={100} value={form.episodes} disabled={running} onChange={(e) => setForm({ ...form, episodes: Number(e.target.value) })} /></Field>
              <Field label="Seconds each"><Input type="number" min={5} max={600} value={form.seconds} disabled={running} onChange={(e) => setForm({ ...form, seconds: Number(e.target.value) })} /></Field>
            </div>
            <Field label="Model" hint={form.mock ? 'Mock is on: a scripted arm, no API calls, no credits.' : 'Every model call uses credits.'}>
              <Input value={form.mock ? 'scripted mock' : form.model} disabled={running || form.mock} onChange={(e) => setForm({ ...form, model: e.target.value })} />
            </Field>
            <div className="flex flex-col gap-2 pt-1">
              <Toggle checked={form.mock} disabled={running} onChange={(v) => setForm({ ...form, mock: v })} label="Mock policy" />
              <Toggle checked={form.light} disabled={running} onChange={(v) => setForm({ ...form, light: v })} label="Light mode: fewer camera frames, less CPU" />
            </div>
            {running ? (
              <Button variant="stop" className="w-full !py-2" onClick={() => stop.mutate()} disabled={stop.isPending}>
                <Square className="h-4 w-4" /> Stop after this step
              </Button>
            ) : (
              <Button variant="go" className="w-full !py-2" onClick={() => start.mutate()} disabled={start.isPending || !form.task.trim()}>
                <Play className="h-4 w-4" /> Start {form.episodes} attempt{form.episodes > 1 ? 's' : ''}
              </Button>
            )}
            {error && <p className="text-[13px] text-bad">{error}</p>}
            {config.data && !config.data.sim_found && (
              <p className="text-[13px] text-signal">Simulator not found. Set sim_path in {config.data.config_path}.</p>
            )}
            {runPath && (
              <button onClick={() => runId && api.reveal(runId)} className="flex w-full items-start gap-2 rounded-md bg-base px-2.5 py-2 text-left text-[12px] text-dim hover:text-ink" title="Open in Finder">
                <FolderOpen className="mt-px h-4 w-4 shrink-0" />
                <span className="min-w-0">Videos and results are saved to<br /><span className="break-all font-mono text-[11px] text-faint">{runPath}</span></span>
              </button>
            )}
          </div>
        </Panel>
        <Feed events={stream.events} onWatch={(ep) => runId && onOpenRun(runId, ep)} />
      </div>
    </div>
  )
}

function marksFrom(events: EventRecord[], episode: number): Mark[] {
  return events
    .filter((e) => e.episode === episode && e.t != null && e.plugin !== 'dreamkit')
    .filter((e) => e.plugin !== 'failure-monitor' || e.data.kind)
    .map((e) => ({ t: e.t!, label: `${e.plugin}: ${e.message}`, tone: e.level === 'error' ? 'bad' : e.level === 'warn' ? 'warn' : e.level === 'success' ? 'ok' : 'info' }))
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
      <span className={clsx('absolute left-2 top-2 flex items-center gap-1.5 rounded bg-black/55 px-1.5 py-0.5 text-[12px] text-ink', large && 'text-[13px]')}>
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

function LiveStrip({ live, seconds, marks }: { live: LiveStore; seconds: number; marks: Mark[] }) {
  useLive(live)
  return <AttemptStrip duration={seconds} marks={marks} playhead={live.t} live />
}

function Attempts({ episodes, planned, current, onOpen }: {
  episodes: EpisodeResult[]
  planned: number
  current: { episode: number; scene: number; attempt: number } | null
  onOpen: (episode: number) => void
}) {
  if (!episodes.length && !current) return null
  const ok = episodes.filter((e) => e.success).length
  return (
    <Panel>
      <PanelTitle right={<span className="text-[13px] text-dim">{ok} of {episodes.length} worked, {new Set(episodes.map((e) => e.scene)).size} of {planned} scenes done</span>}>
        Attempts
      </PanelTitle>
      <ul className="divide-y divide-line border-t border-line">
        {episodes.map((e) => (
          <li key={e.episode}>
            <button onClick={() => onOpen(e.episode)} className="flex w-full items-center gap-3 px-4 py-2.5 text-left hover:bg-raised/50">
              <Outcome success={e.success} />
              <span className="w-24 shrink-0 text-[13px] text-dim">Scene {e.scene + 1}{e.attempt > 1 ? `, retry ${e.attempt - 1}` : ''}</span>
              <span className={clsx('min-w-0 flex-1 truncate text-[13px]', e.success ? 'text-ink' : 'text-bad')}>
                {e.success ? `Done in ${e.duration_s.toFixed(1)} s` : e.recap?.headline?.replace(/^Attempt \d+: /, '') ?? `${e.message ?? REASON_LABEL[e.reason ?? 'unknown']}${e.at != null ? ` at ${e.at.toFixed(1)} s` : ''}`}
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

function Feed({ events, onWatch }: { events: EventRecord[]; onWatch: (episode: number) => void }) {
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
        {shown.length === 0 && <p className="px-4 py-8 text-center text-[13px] text-faint">Start a run. Each plugin reports here as it sees things happen.</p>}
        {shown.map((e, i) => {
          const verdict = e.plugin === 'failure-monitor' && e.level === 'error' && Boolean(e.data.reason)
          return (
            <div key={`${e.ts}-${i}`} className={clsx('flex gap-2.5 border-b border-line/60 px-4 py-2', verdict && 'bg-bad/8')}>
              <PluginIcon name={e.plugin} className={clsx('mt-0.5', LEVEL_TONE[e.level])} />
              <div className="min-w-0 flex-1">
                <p className={clsx('text-[13px]', verdict ? 'font-medium text-bad' : e.level === 'info' ? 'text-ink' : LEVEL_TONE[e.level])}>{e.message}</p>
                <p className="text-[11px] text-faint">{e.plugin}{e.t != null ? `, ${e.t.toFixed(1)} s` : ''}{e.episode ? ` in attempt ${e.episode}` : ''}</p>
                {verdict && e.episode != null && (
                  <button onClick={() => onWatch(e.episode!)} className="mt-1 text-[12px] text-steel hover:underline">Watch from {Math.max(0, Number(e.data.at) - 1).toFixed(1)} s</button>
                )}
              </div>
            </div>
          )
        })}
      </div>
    </Panel>
  )
}
