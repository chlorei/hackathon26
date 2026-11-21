import { useEffect, useMemo, useRef, useState } from 'react'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { AnimatePresence, motion } from 'framer-motion'
import clsx from 'clsx'
import { CameraOff, ListChecks, Play, Radio, Square, Video } from 'lucide-react'
import { api, REASON_LABEL, type EpisodeResult, type EventRecord } from '../api'
import { useRunStream } from '../useRunStream'
import { Badge, Button, Card, CardHeader, Input, Label, LEVEL_STYLE, PluginIcon, ResultMark, Switch } from '../ui'

const CAMS = [
  { key: 'top', label: 'overhead' },
  { key: 'left', label: 'left wrist' },
  { key: 'right', label: 'right wrist' },
]

export default function RunPage({ onOpenRun }: { onOpenRun: (id: string, episode?: number) => void }) {
  const queryClient = useQueryClient()
  const config = useQuery({ queryKey: ['config'], queryFn: api.config })
  const runs = useQuery({ queryKey: ['runs'], queryFn: api.runs, refetchInterval: 4000 })
  const [runId, setRunId] = useState<string | null>(null)
  const [form, setForm] = useState({ task: '', model: '', episodes: 3, seconds: 40, mock: true })
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    if (config.data && !form.task) {
      setForm((f) => ({ ...f, task: config.data.task, model: config.data.model, episodes: config.data.episodes, seconds: config.data.seconds }))
    }
  }, [config.data, form.task])

  // Re-attach to a run that is still going (page reload, another tab started it).
  useEffect(() => {
    if (!runId && runs.data?.active) setRunId(runs.data.active)
  }, [runs.data?.active, runId])

  const stream = useRunStream(runId)
  const running = stream.status === 'running' || stream.status === 'connecting'

  const start = useMutation({
    mutationFn: () => api.startRun(form),
    onSuccess: ({ id }) => {
      setError(null)
      setRunId(id)
      queryClient.invalidateQueries({ queryKey: ['runs'] })
    },
    onError: (e: Error) => setError(e.message),
  })
  const stop = useMutation({ mutationFn: () => api.stopRun(runId!) })

  return (
    <div className="grid grid-cols-12 gap-4">
      <div className="col-span-12 space-y-4 xl:col-span-8">
        <div className="grid grid-cols-3 gap-3">
          {CAMS.map((cam) => (
            <CameraTile key={cam.key} label={cam.label} src={stream.frames[cam.key]} live={running} />
          ))}
        </div>
        <Card>
          <CardHeader
            icon={<Play className="h-4 w-4" />}
            title="Run"
            right={
              stream.current ? (
                <div className="flex items-center gap-2 text-xs text-zinc-400">
                  <Radio className="h-3.5 w-3.5 animate-pulse text-red-400" />
                  attempt {stream.current.episode} · scene {stream.current.scene + 1} · try {stream.current.attempt} ·{' '}
                  <span className="font-mono text-zinc-200">{stream.current.t.toFixed(1)} s</span>
                </div>
              ) : null
            }
          />
          <div className="grid grid-cols-12 items-end gap-3 p-4">
            <div className="col-span-12 md:col-span-5">
              <Label>Task</Label>
              <Input value={form.task} onChange={(e) => setForm({ ...form, task: e.target.value })} disabled={running} list="tasks" />
              <datalist id="tasks">
                <option value="put apple into the bowl" />
                <option value="put orange into the bowl" />
                <option value="put banana on the plate" />
              </datalist>
            </div>
            <div className="col-span-6 md:col-span-3">
              <Label>Model</Label>
              <Input value={form.model} onChange={(e) => setForm({ ...form, model: e.target.value })} disabled={running || form.mock} />
            </div>
            <div className="col-span-3 md:col-span-1">
              <Label>Attempts</Label>
              <Input type="number" min={1} max={100} value={form.episodes}
                onChange={(e) => setForm({ ...form, episodes: Number(e.target.value) })} disabled={running} />
            </div>
            <div className="col-span-3 md:col-span-1">
              <Label>Sec</Label>
              <Input type="number" min={5} max={600} value={form.seconds}
                onChange={(e) => setForm({ ...form, seconds: Number(e.target.value) })} disabled={running} />
            </div>
            <div className="col-span-12 flex items-center justify-between gap-3 md:col-span-2 md:justify-end">
              <div className="flex items-center gap-2 text-sm text-zinc-400" title="Scripted policy: no API calls, no credits">
                <Switch checked={form.mock} onChange={(v) => setForm({ ...form, mock: v })} disabled={running} /> mock
              </div>
              {running ? (
                <Button variant="danger" onClick={() => stop.mutate()} disabled={stop.isPending}>
                  <Square className="h-4 w-4" /> Stop
                </Button>
              ) : (
                <Button onClick={() => start.mutate()} disabled={start.isPending || !form.task.trim()}>
                  <Play className="h-4 w-4" /> Start
                </Button>
              )}
            </div>
          </div>
          {error && <div className="border-t border-zinc-800 px-4 py-2 text-sm text-red-400">{error}</div>}
          {config.data && !config.data.sim_found && (
            <div className="border-t border-zinc-800 px-4 py-2 text-sm text-amber-400">
              Dream Labs simulator not found — set <code>sim_path</code> in {config.data.config_path}
            </div>
          )}
          <Progress planned={form.episodes} episodes={stream.episodes} current={stream.current} onOpen={(ep) => runId && onOpenRun(runId, ep)} />
        </Card>
        {stream.status !== 'idle' && !running && runId && <RunDone runId={runId} episodes={stream.episodes} onOpenRun={onOpenRun} />}
      </div>
      <div className="col-span-12 xl:col-span-4">
        <EventFeed events={stream.events} plugins={stream.plugins} onSeek={(ep) => runId && onOpenRun(runId, ep)} />
      </div>
    </div>
  )
}

function CameraTile({ label, src, live }: { label: string; src?: string; live: boolean }) {
  return (
    <div className="relative aspect-video overflow-hidden rounded-xl border border-zinc-800 bg-zinc-900">
      {src ? (
        <img src={src} alt={label} className="h-full w-full object-cover" />
      ) : (
        <div className="flex h-full w-full flex-col items-center justify-center gap-2 text-zinc-600">
          <CameraOff className="h-6 w-6" />
          <span className="text-xs">no signal</span>
        </div>
      )}
      <div className="absolute left-2 top-2 flex items-center gap-1.5 rounded-md bg-black/60 px-2 py-0.5 text-xs text-zinc-200 backdrop-blur">
        {live && src && <span className="h-1.5 w-1.5 animate-pulse rounded-full bg-red-500" />}
        {label}
      </div>
    </div>
  )
}

function Progress({ planned, episodes, current, onOpen }: {
  planned: number
  episodes: EpisodeResult[]
  current: RunStreamCurrent
  onOpen: (episode: number) => void
}) {
  if (!episodes.length && !current) return null
  return (
    <div className="flex flex-wrap items-center gap-2 border-t border-zinc-800 px-4 py-3">
      <ListChecks className="h-4 w-4 text-zinc-500" />
      {episodes.map((e) => (
        <button key={e.episode} onClick={() => onOpen(e.episode)}
          className={clsx('rounded-md border px-2 py-1 text-xs transition hover:bg-zinc-800',
            e.success ? 'border-emerald-500/40 text-emerald-300' : e.success === false ? 'border-red-500/40 text-red-300' : 'border-zinc-700 text-zinc-400')}>
          <ResultMark success={e.success} /> {e.scene + 1}.{e.attempt}{' '}
          {e.success ? '' : <span className="opacity-80">{REASON_LABEL[e.reason ?? 'unknown'] ?? e.reason}{e.at != null ? ` @${e.at.toFixed(1)}s` : ''}</span>}
        </button>
      ))}
      {current && (
        <span className="animate-pulse rounded-md border border-sky-500/40 px-2 py-1 text-xs text-sky-300">
          ▶ {current.scene + 1}.{current.attempt}
        </span>
      )}
      <span className="ml-auto text-xs text-zinc-500">
        {new Set(episodes.map((e) => e.scene)).size}/{planned} scenes
      </span>
    </div>
  )
}
type RunStreamCurrent = ReturnType<typeof useRunStream>['current']

function RunDone({ runId, episodes, onOpenRun }: { runId: string; episodes: EpisodeResult[]; onOpenRun: (id: string, episode?: number) => void }) {
  const ok = episodes.filter((e) => e.success).length
  const firstFail = episodes.find((e) => e.success === false)
  return (
    <Card className="flex items-center justify-between px-4 py-3">
      <div className="text-sm">
        <span className={ok ? 'text-emerald-400' : 'text-red-400'}>{ok}/{episodes.length}</span> attempts succeeded
        {firstFail?.message && <span className="text-zinc-400"> · first failure: {firstFail.message} at {firstFail.at?.toFixed(1)} s</span>}
      </div>
      <div className="flex gap-2">
        {firstFail && (
          <Button variant="outline" onClick={() => onOpenRun(runId, firstFail.episode)}>
            <Video className="h-4 w-4" /> Watch the failure
          </Button>
        )}
        <Button variant="ghost" onClick={() => onOpenRun(runId)}>Details</Button>
      </div>
    </Card>
  )
}

function EventFeed({ events, plugins, onSeek }: { events: EventRecord[]; plugins: string[]; onSeek: (episode: number) => void }) {
  const [filter, setFilter] = useState<string | null>(null)
  const bottom = useRef<HTMLDivElement>(null)
  const shown = useMemo(() => (filter ? events.filter((e) => e.plugin === filter) : events), [events, filter])
  useEffect(() => {
    void bottom.current?.scrollIntoView({ behavior: 'smooth', block: 'end' })
  }, [shown.length])
  return (
    <Card className="flex h-[calc(100vh-7.5rem)] flex-col">
      <CardHeader icon={<Radio className="h-4 w-4" />} title="Plugin events" right={<span className="text-xs text-zinc-500">{events.length}</span>} />
      {plugins.length > 0 && (
        <div className="flex flex-wrap gap-1 border-b border-zinc-800 px-3 py-2">
          {plugins.map((p) => (
            <button key={p} onClick={() => setFilter(filter === p ? null : p)}
              className={clsx('flex items-center gap-1 rounded-md px-1.5 py-0.5 text-xs', filter === p ? 'bg-zinc-200 text-zinc-900' : 'bg-zinc-800 text-zinc-400 hover:text-zinc-200')}>
              <PluginIcon name={p} className="h-3 w-3" /> {p}
            </button>
          ))}
        </div>
      )}
      <div className="flex-1 space-y-1.5 overflow-y-auto p-3">
        {shown.length === 0 && <div className="py-10 text-center text-sm text-zinc-600">Start a run — every plugin reports here.</div>}
        <AnimatePresence initial={false}>
          {shown.map((e, i) => {
            const style = LEVEL_STYLE[e.level] ?? LEVEL_STYLE.info
            const verdict = e.plugin === 'failure-monitor' && e.level === 'error'
            return (
              <motion.div key={`${e.ts}-${i}`} initial={{ opacity: 0, y: 8, scale: 0.98 }} animate={{ opacity: 1, y: 0, scale: 1 }}
                transition={{ duration: 0.2 }}
                className={clsx('rounded-lg border bg-zinc-950/60 px-3 py-2', style.border, verdict && 'bg-red-950/40 ring-1 ring-red-500/40')}>
                <div className="flex items-center gap-2 text-xs text-zinc-500">
                  <PluginIcon name={e.plugin} className={style.text} />
                  <span className={clsx('font-medium', style.text)}>{e.plugin}</span>
                  <span className="ml-auto font-mono">{e.episode != null ? `#${e.episode}` : ''} {e.t != null ? `${e.t.toFixed(1)}s` : ''}</span>
                </div>
                <div className={clsx('mt-0.5 text-sm', verdict ? 'font-medium text-red-200' : 'text-zinc-200')}>{e.message}</div>
                {verdict && e.episode != null && (
                  <button onClick={() => onSeek(e.episode!)} className="mt-1 flex items-center gap-1 text-xs text-red-300 hover:text-red-100">
                    <Video className="h-3 w-3" /> watch at {Number(e.data.at ?? e.t).toFixed(1)} s
                  </button>
                )}
              </motion.div>
            )
          })}
        </AnimatePresence>
        <div ref={bottom} />
      </div>
      {events.some((e) => e.level === 'error') && (
        <div className="border-t border-zinc-800 px-3 py-2 text-xs text-zinc-500">
          <Badge tone="red">{events.filter((e) => e.level === 'error').length} errors</Badge>{' '}
          <Badge tone="yellow">{events.filter((e) => e.level === 'warn').length} warnings</Badge>
        </div>
      )}
    </Card>
  )
}
