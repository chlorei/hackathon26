import { useEffect, useMemo, useState, type ReactNode } from 'react'
import { useMutation, useQuery } from '@tanstack/react-query'
import clsx from 'clsx'
import { FolderOpen } from 'lucide-react'
import { api, atOf, REASON_LABEL, tOf, type DemoState, type EpisodeResult, type EventRecord } from '../api'
import { toast } from '../toast'
import { useLive, useRunStream, type LiveStore } from '../useRunStream'
import Feed from './LiveFeed'
import { AttemptStrip, Button, ICON_BY_PLUGIN, Outcome, pagesOf, Panel, PanelTitle, PluginPageDrawer, Tag, type Mark, type PluginPage } from '../ui'

const plural = (n: number, word: string) => `${n} ${word}${n === 1 ? '' : 's'}`


// The same API as README.md: five lines, the loop around them is yours.
const SNIPPET = `import dream, dreamkit

client = dreamkit.wrap(dream.DreamClient())        # the one line
with client.episode("put apple into the bowl", env=env):
    reply = client.get_action()                       # plugins run around the model
    env.step(left, right)                             # each row of the reply
    client.step(env, command=(left, right))           # robot: state=... instead of env`

/** Watches whatever run the user's loop is writing. It never starts anything. */
export default function LivePage({ onOpenRun }: { onOpenRun: (id: string, episode?: number) => void }) {
  const live = useQuery({ queryKey: ['live'], queryFn: api.live, refetchInterval: 2000 })
  const demo = useQuery({ queryKey: ['demo'], queryFn: api.demo, refetchInterval: 3000 })
  const startDemo = useMutation({ mutationFn: api.startDemo, onSuccess: () => demo.refetch(), onError: (e: Error) => toast(`Could not start: ${e.message}`) })
  const stopDemo = useMutation({ mutationFn: api.stopDemo, onSuccess: () => { demo.refetch(); live.refetch() } })
  const plugins = useQuery({ queryKey: ['plugins'], queryFn: api.plugins })
  for (const p of plugins.data?.plugins ?? []) ICON_BY_PLUGIN[p.name] = p.icon
  const pages = useMemo(() => pagesOf(plugins.data?.plugins), [plugins.data])
  const [page, setPage] = useState<PluginPage | null>(null)
  const [runId, setRunId] = useState<string | null>(null)
  useEffect(() => {
    const id = live.data?.id
    if (id && id !== runId) setRunId(id) // a newer run took over
  }, [live.data?.id, runId])

  const stream = useRunStream(runId, false)  // no camera frames: people watch the arm itself; videos are in History
  const running = Boolean(live.data?.id) && live.data?.id === runId
  const current = stream.current
  const marks = useMemo<Mark[]>(() => (current ? marksFrom(stream.events, current.episode) : []), [stream.events, current])

  if (!runId) return <Waiting demo={demo.data} onDemo={(policy) => startDemo.mutate(policy)} />

  return (
    <div className="space-y-4">
    <StatusBar task={stream.task || live.data?.task || 'Run'} mode={stream.mode} running={running} mock={Boolean(live.data?.mock)}
      model={live.data?.model} current={current} episodes={stream.episodes} live={stream.live}
      onFolder={() => api.reveal(runId)} onHistory={() => onOpenRun(runId)}
      onStop={demo.data?.running && running ? () => stopDemo.mutate() : undefined} stopping={stopDemo.isPending}
      onNewDemo={!running && demo.data?.available && !demo.data.running ? () => startDemo.mutate('mock') : undefined} />
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
          {current && <RobotNow live={stream.live} robotOnly={Boolean(stream.mode?.startsWith('robot'))} mock={Boolean(live.data?.mock)} />}
        </Panel>
        <Attempts episodes={stream.episodes} current={current} onOpen={(ep) => onOpenRun(runId, ep)} />
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
function StatusBar({ task, mode, running, mock, model, current, episodes, live, onFolder, onHistory, onStop, stopping, onNewDemo }: {
  task: string; mode?: string; running: boolean; mock: boolean; model?: string
  current: { episode: number; scene: number; attempt: number } | null
  episodes: EpisodeResult[]; live: LiveStore
  onFolder: () => void; onHistory: () => void
  onStop?: () => void; stopping?: boolean  // the demo the panel started
  onNewDemo?: () => void  // a finished run: start the demo again
}) {
  const ok = episodes.filter((e) => e.success).length
  const lat = episodes.map((e) => ((e as unknown as Record<string, unknown>)['failure-monitor'] as { latency?: { mean_ms?: number; calls?: number } } | undefined)?.latency)
    .filter((l): l is { mean_ms: number; calls: number } => Boolean(l && l.mean_ms != null && l.calls))
  const calls = lat.reduce((a, l) => a + l.calls, 0)
  const meanMs = calls ? lat.reduce((a, l) => a + l.mean_ms * l.calls, 0) / calls : null
  const item = (label: string, value: ReactNode, title?: string) => (
    <div className="min-w-0" title={title}>
      <p className="text-[12px] text-dim">{label}</p>
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
          {onStop && <Button variant="stop" onClick={onStop} disabled={stopping}>{stopping ? 'Stopping…' : 'Stop'}</Button>}
          {onNewDemo && <Button variant="quiet" onClick={onNewDemo} title="Try demo again (free, simulator)">New demo</Button>}
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
  const last = [...mine].reverse().find((e) => (e.level !== 'info' || e.plugin === 'failure-monitor') && !/model round trip/.test(e.message))
  const done = episodes.length
  const ok = episodes.filter((e) => e.success).length
  const tone = !current ? 'idle' : recovering && (!verdict || (tOf(recovering) ?? 0) >= (tOf(verdict) ?? 0)) ? 'recover'
    : verdict ? 'bad' : 'ok'
  const headline = !current ? (running ? 'Between attempts' : 'Run finished')
    : tone === 'recover' ? recovering!.message
    : verdict ? verdict.message.replace(/^verdict: /, '')
    : last ? last.message : 'Running, nothing wrong so far'
  // the moment it goes wrong, said big for a few seconds: what failed, when, and what recovery does about it
  const fresh = verdict && current && Date.now() / 1000 - (verdict.ts ?? 0) < 8
  const fix = fresh ? [...mine].find((e) => e.plugin === 'recovery' && e.data?.action && (tOf(e) ?? 0) >= (tOf(verdict) ?? 0)) : undefined
  return (
    <>
    {fresh && (
      <div role="alert" className="flex flex-wrap items-baseline gap-x-3 gap-y-1 rounded-xl border border-bad/50 bg-bad-soft px-6 py-3.5">
        <span className="text-[20px] font-semibold text-bad">Attempt {current!.episode}: {verdict!.message.replace(/^verdict: /, '')}</span>
        {fix && <span className="text-[16px] text-signal">→ recovery: {fix.message.replace(/^[^:]+: /, '')}</span>}
      </div>
    )}
    <Panel className={clsx('px-6 py-5', tone === 'bad' && 'border-bad/50', tone === 'recover' && 'border-signal/50')}>
      <div className="flex flex-wrap items-baseline gap-x-4 gap-y-1">
        <span className="text-[14px] font-semibold text-dim">
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
    </>
  )
}

const STEPS = [
  { n: 1, title: 'Install dreamkit and make a config', code: 'pip install dreamkit && dreamkit init' },
  { n: 2, title: 'Wrap your Dream-1 client: one line', code: 'client = dreamkit.wrap(dream.DreamClient())' },
  { n: 3, title: 'Run your loop', code: 'python my_robot.py', after: 'this page comes alive' },
]

/** No run yet: try the demo here, or connect your own robot in three steps. */
/** What to do about a demo that would not start, from its last lines. */
function demoHint(log: string): string {
  if (/simulator not found/i.test(log)) return 'dreamkit cannot find Dream Labs\' simulator: set sim_path in dreamkit.yaml, or pip install dreamlabs.'
  if (/No module named/i.test(log)) return 'A package is missing in this Python: reinstall (uv pip install -e ".[dev]").'
  if (/DREAM_API_KEY|api key|401|403/i.test(log)) return 'The Dream API key was refused: check DREAM_API_KEY in .env.'
  return 'Its last lines are below; ~/.dreamkit/demo.log has all of it.'
}

function Waiting({ demo, onDemo }: { demo?: DemoState; onDemo: (policy: 'mock' | 'dream') => void }) {
  const [copied, setCopied] = useState<string | null>(null)
  const [confirm, setConfirm] = useState(false)
  const copy = (what: string) => { void navigator.clipboard.writeText(what); setCopied(what) }
  const starting = demo?.running
  const copyButton = (what: string) => (
    <button className="shrink-0 text-[13px] text-dim hover:text-ink" onClick={() => copy(what)}>{copied === what ? 'Copied' : 'Copy'}</button>
  )
  return (
    <div className="mx-auto max-w-[760px] space-y-5 pt-10">
      <div>
        <h1 className="text-[24px] font-semibold">No robot running yet</h1>
        <p className="mt-1.5 text-[15px] text-dim">Watch dreamkit at work on Dream Labs' simulator, or connect your own robot.</p>
      </div>
      {demo?.available !== false && (
        <Panel className="flex flex-wrap items-center gap-3 px-5 py-4">
          <Button variant="go" disabled={starting} onClick={() => onDemo('mock')}>
            {starting && demo?.policy === 'mock' ? 'Starting the demo…' : 'Try demo (free, simulator)'}
          </Button>
          {!confirm ? (
            <Button disabled={starting || !demo?.dream_key} onClick={() => setConfirm(true)}
              title={demo?.dream_key ? 'Dream-1 drives the simulated arms: 2 attempts' : 'Add DREAM_API_KEY to .env (or ~/.dreamlabs/credentials) first'}>
              Run with Dream-1
            </Button>
          ) : (
            <span className="flex flex-wrap items-center gap-2 rounded-lg bg-raised px-3 py-1.5 text-[13.5px]">
              This uses Dream-1 credits (2 attempts, about $0.05).
              <Button variant="go" onClick={() => { setConfirm(false); onDemo('dream') }}>Run</Button>
              <Button variant="quiet" onClick={() => setConfirm(false)}>Cancel</Button>
            </span>
          )}
          <p className="w-full text-[13px] text-dim">
            {starting ? 'The simulator is starting; this page switches to the run in a few seconds.'
              : 'The demo drops the apple on purpose in attempts 1 and 3, so you see the plugins notice it and recovery step in.'}
            {!demo?.dream_key && ' Dream-1 needs a Dream API key.'}
          </p>
          {!starting && demo?.failed && (
            <div role="alert" className="w-full rounded-lg border border-bad/40 bg-bad-soft px-3 py-2 text-[13.5px]">
              <p className="font-medium text-bad">The demo stopped before it began. {demoHint(demo.failed)}</p>
              <pre className="mt-1.5 max-h-40 overflow-auto font-mono text-[12px] whitespace-pre-wrap text-dim">{demo.failed}</pre>
            </div>
          )}
        </Panel>
      )}
      <Panel>
        <PanelTitle>Or connect your own robot in 3 steps</PanelTitle>
        <ol className="space-y-3 border-t border-line px-5 py-4">
          {STEPS.map((s) => (
            <li key={s.n} className="flex gap-3">
              <span className="flex h-6 w-6 shrink-0 items-center justify-center rounded-full bg-ink text-[12px] font-bold text-panel">{s.n}</span>
              <div className="min-w-0 flex-1">
                <p className="text-[14px] font-medium">{s.title}</p>
                <div className="mt-1 flex items-center gap-3">
                  <code className="min-w-0 flex-1 truncate rounded bg-raised px-2.5 py-1 font-mono text-[13px]">{s.code}</code>
                  {copyButton(s.code)}
                  {s.after && <span className="shrink-0 text-[13px] text-dim">→ {s.after}</span>}
                </div>
                {s.n === 2 && (
                  <details className="mt-2 text-[13px]">
                    <summary className="cursor-pointer text-dim hover:text-ink">The full loop</summary>
                    <div className="mt-2 flex items-start gap-3">
                      <pre className="min-w-0 flex-1 overflow-x-auto rounded bg-raised px-3 py-2.5 font-mono text-[12.5px] leading-relaxed">{SNIPPET}</pre>
                      {copyButton(SNIPPET)}
                    </div>
                  </details>
                )}
              </div>
            </li>
          ))}
        </ol>
      </Panel>
      {demo && (
        <p className="text-[13px] text-dim">
          Plugins enabled: {demo.plugins.length ? demo.plugins.join(', ') : 'none'} · <a href="#/plugins" className="text-signal hover:underline">Change →</a>
        </p>
      )}
    </div>
  )
}

function marksFrom(events: EventRecord[], episode: number): Mark[] {
  return events
    .filter((e) => e.episode === episode && tOf(e) != null && e.plugin !== 'dreamkit')
    .filter((e) => e.plugin !== 'failure-monitor' || e.data.kind)
    .map((e) => ({ t: tOf(e)!, label: `${e.plugin}: ${e.message}`, tone: e.level === 'error' ? 'bad' : e.level === 'warn' ? 'warn' : e.level === 'success' ? 'ok' : 'info' }))
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
      <p className="text-[12px] text-dim">Spent</p>
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

/** An attempt's failure in a few plain words: no 'at 0.0 s', a chain of moments cut to its first and last. */
function whyFailed(e: EpisodeResult): string {
  if (e.success == null) return `ran ${e.duration_s.toFixed(0)} s; did it work? The arm alone cannot tell: mark it in History`
  if (e.reason === 'model_error') return 'the model did not answer'
  const at = atOf(e)
  let text = e.recap?.headline?.replace(/^Attempt \d+: /, '') ?? `${e.message ?? REASON_LABEL[e.reason ?? 'unknown']}${at ? ` at ${at.toFixed(1)} s` : ''}`
  text = text.replace(/ at 0\.0 s\.?$/, '').replace(/\.$/, '')
  const parts = text.split(/, (?:then )?/)
  return parts.length > 2 ? `${parts[0]} … then ${parts[parts.length - 1]}` : text
}

/** Where the robot is right now, in one sentence (the latest point the loop sent). */
function RobotNow({ live, robotOnly, mock }: { live: LiveStore; robotOnly: boolean; mock: boolean }) {
  useLive(live)
  const v = live.points[live.points.length - 1]?.v ?? {}
  const cm = (x: number) => `${Math.round(x * 100)} cm`
  const parts = robotOnly
    ? [v.grip != null ? `gripper ${v.grip > 0.6 ? 'open' : v.grip < 0.15 ? 'closed' : 'half open'} (${v.grip.toFixed(2)})` : '',
       v.track != null ? (v.track > 0.12 ? `the arm lags its commands (${v.track.toFixed(2)} rad)` : 'the arm follows its commands') : '']
    : [v.d != null ? `hand ${cm(v.d)} from the object` : '', v.z != null ? (v.z > 0.01 ? `object lifted ${cm(v.z)}` : 'object on the table') : '',
       v.dt != null ? `${cm(v.dt)} from the target` : '']
  if (!mock && v.rtt != null) parts.push(`model answers in ${Math.round(v.rtt)} ms${v.rtt > 700 ? ' (slow)' : ''}`)
  const text = parts.filter(Boolean).join(' · ')
  return <p className="border-t border-line px-4 py-2.5 text-[13.5px] text-dim">{text ? text[0].toUpperCase() + text.slice(1) : 'Waiting for the first readings'}</p>
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
      <PanelTitle right={<span className="text-[13px] text-dim">{ok} of {episodes.length} worked, {plural(new Set(episodes.map((e) => e.scene)).size, 'scene')} so far</span>}>
        Attempts
      </PanelTitle>
      <ul className="divide-y divide-line border-t border-line">
        {episodes.map((e) => (
          <li key={e.episode}>
            <button onClick={() => onOpen(e.episode)} className="flex w-full items-center gap-3 px-4 py-2.5 text-left hover:bg-raised/50">
              <Outcome success={e.success} />
              <span className="w-24 shrink-0 text-[13px] text-dim">Scene {e.scene + 1}{e.attempt > 1 ? `, retry ${e.attempt - 1}` : ''}</span>
              <span className={clsx('min-w-0 flex-1 truncate text-[13px]', e.success ? 'text-ink' : e.success === false ? 'text-bad' : 'text-dim')}>
                {e.success ? `Done in ${e.duration_s.toFixed(1)} s` : whyFailed(e)}
              </span>
              <span className="shrink-0 text-[12px] text-steel">{e.success ? 'Replay' : e.success === false ? 'Watch the failure' : 'Watch and mark it'}</span>
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
