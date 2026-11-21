import { useEffect, useMemo, useRef, useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import clsx from 'clsx'
import {
  Bar, BarChart, CartesianGrid, Cell, Legend, Line, LineChart, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis,
} from 'recharts'
import { ArrowLeft, BarChart3, FileText, History, Video } from 'lucide-react'
import { api, REASON_LABEL, type EpisodeResult, type RunInfo } from '../api'
import { Badge, Button, Card, CardHeader, formatTime, ResultMark } from '../ui'

const TOOLTIP = { contentStyle: { background: '#18181b', border: '1px solid #3f3f46', borderRadius: 8, fontSize: 12 } }

export default function HistoryPage({ runId, episode, onOpen, onBack }: {
  runId: string | null
  episode: number | null
  onOpen: (id: string, episode?: number) => void
  onBack: () => void
}) {
  if (runId) return <RunDetail id={runId} initialEpisode={episode} onBack={onBack} />
  return <RunList onOpen={onOpen} />
}

function RunList({ onOpen }: { onOpen: (id: string) => void }) {
  const runs = useQuery({ queryKey: ['runs'], queryFn: api.runs, refetchInterval: 5000 })
  const list = runs.data?.runs ?? []
  const all = list.flatMap((r) => r.episodes ?? [])
  return (
    <div className="space-y-4">
      <Charts runs={list} episodes={all} />
      <Card>
        <CardHeader icon={<History className="h-4 w-4" />} title="Runs" right={<span className="text-xs text-zinc-500">{list.length}</span>} />
        <table className="w-full text-sm">
          <thead className="text-left text-xs uppercase tracking-wide text-zinc-500">
            <tr>
              <th className="px-4 py-2 font-medium">when</th>
              <th className="px-4 py-2 font-medium">task</th>
              <th className="px-4 py-2 font-medium">model</th>
              <th className="px-4 py-2 font-medium">result</th>
              <th className="px-4 py-2 font-medium">failures</th>
              <th className="px-4 py-2 font-medium">status</th>
            </tr>
          </thead>
          <tbody>
            {list.map((r) => (
              <tr key={r.id} onClick={() => onOpen(r.id)} className="cursor-pointer border-t border-zinc-800 hover:bg-zinc-800/40">
                <td className="whitespace-nowrap px-4 py-2.5 text-zinc-400">{formatTime(r.started)}</td>
                <td className="px-4 py-2.5">{r.task}</td>
                <td className="px-4 py-2.5">{r.mock ? <Badge tone="violet">mock</Badge> : <span className="text-zinc-400">{r.model}</span>}</td>
                <td className="px-4 py-2.5">
                  <div className="flex gap-0.5">
                    {(r.episodes ?? []).map((e) => <ResultMark key={e.episode} success={e.success} />)}
                  </div>
                </td>
                <td className="px-4 py-2.5">
                  <div className="flex flex-wrap gap-1">
                    {Object.entries(r.summary?.reasons ?? {}).map(([k, n]) => (
                      <Badge key={k} tone="red">{REASON_LABEL[k] ?? k} ×{n}</Badge>
                    ))}
                  </div>
                </td>
                <td className="px-4 py-2.5"><StatusBadge status={r.status} /></td>
              </tr>
            ))}
            {!list.length && (
              <tr><td colSpan={6} className="px-4 py-10 text-center text-zinc-600">No runs yet.</td></tr>
            )}
          </tbody>
        </table>
      </Card>
    </div>
  )
}

function StatusBadge({ status }: { status: string }) {
  const tone = status === 'finished' ? 'green' : status === 'running' ? 'blue' : status === 'error' ? 'red' : 'zinc'
  return <Badge tone={tone}>{status}</Badge>
}

function Charts({ runs, episodes }: { runs: RunInfo[]; episodes: EpisodeResult[] }) {
  const byTask = useMemo(() => {
    const m = new Map<string, { task: string; ok: number; fail: number }>()
    for (const e of episodes) {
      const row = m.get(e.task) ?? { task: e.task.replace(/^put /, ''), ok: 0, fail: 0 }
      if (e.success) row.ok += 1
      else row.fail += 1
      m.set(e.task, row)
    }
    return [...m.values()]
  }, [episodes])
  const reasons = useMemo(() => {
    const m: Record<string, number> = {}
    for (const e of episodes) if (e.success === false) m[e.reason ?? 'unknown'] = (m[e.reason ?? 'unknown'] ?? 0) + 1
    return Object.entries(m).map(([reason, n]) => ({ reason: REASON_LABEL[reason] ?? reason, n })).sort((a, b) => b.n - a.n)
  }, [episodes])
  const latency = useMemo(
    () => runs.slice().reverse().flatMap((r) => (r.episodes ?? []).filter((e) => e.latency).map((e) => ({
      name: `${r.id.slice(9, 13)}#${e.episode}`, mean: e.latency!.mean_ms, p95: e.latency!.p95_ms,
    }))).slice(-40),
    [runs],
  )
  const ok = episodes.filter((e) => e.success).length
  return (
    <div className="grid grid-cols-12 gap-4">
      <Card className="col-span-12 p-4 md:col-span-3">
        <div className="text-xs uppercase tracking-wide text-zinc-500">success rate</div>
        <div className={clsx('mt-1 text-4xl font-semibold', ok / Math.max(1, episodes.length) >= 0.5 ? 'text-emerald-400' : 'text-red-400')}>
          {episodes.length ? Math.round((100 * ok) / episodes.length) : 0}%
        </div>
        <div className="text-sm text-zinc-500">{ok}/{episodes.length} attempts · {runs.length} runs</div>
      </Card>
      <ChartCard title="success by task" className="md:col-span-3">
        <BarChart data={byTask}>
          <CartesianGrid stroke="#27272a" vertical={false} />
          <XAxis dataKey="task" tick={{ fill: '#a1a1aa', fontSize: 11 }} />
          <YAxis allowDecimals={false} tick={{ fill: '#71717a', fontSize: 11 }} width={24} />
          <Tooltip {...TOOLTIP} cursor={{ fill: '#27272a' }} />
          <Bar dataKey="ok" stackId="a" fill="#22c55e" name="success" />
          <Bar dataKey="fail" stackId="a" fill="#ef4444" name="fail" radius={[4, 4, 0, 0]} />
        </BarChart>
      </ChartCard>
      <ChartCard title="why attempts failed" className="md:col-span-3">
        <BarChart data={reasons} layout="vertical" margin={{ left: 20 }}>
          <XAxis type="number" allowDecimals={false} tick={{ fill: '#71717a', fontSize: 11 }} />
          <YAxis type="category" dataKey="reason" tick={{ fill: '#d4d4d8', fontSize: 11 }} width={90} />
          <Tooltip {...TOOLTIP} cursor={{ fill: '#27272a' }} />
          <Bar dataKey="n" name="attempts" radius={[0, 4, 4, 0]}>
            {reasons.map((r) => <Cell key={r.reason} fill="#ef4444" />)}
          </Bar>
        </BarChart>
      </ChartCard>
      <ChartCard title="model latency (ms)" className="md:col-span-3">
        <LineChart data={latency}>
          <CartesianGrid stroke="#27272a" vertical={false} />
          <XAxis dataKey="name" hide />
          <YAxis tick={{ fill: '#71717a', fontSize: 11 }} width={36} />
          <Tooltip {...TOOLTIP} />
          <ReferenceLine y={700} stroke="#f59e0b" strokeDasharray="4 4" />
          <Line dataKey="mean" stroke="#60a5fa" dot={false} strokeWidth={2} />
          <Line dataKey="p95" stroke="#a78bfa" dot={false} strokeWidth={1.5} />
        </LineChart>
      </ChartCard>
    </div>
  )
}

function ChartCard({ title, className, children }: { title: string; className?: string; children: React.ReactElement }) {
  return (
    <Card className={clsx('col-span-12 p-3', className)}>
      <div className="mb-1 flex items-center gap-1.5 px-1 text-xs uppercase tracking-wide text-zinc-500">
        <BarChart3 className="h-3.5 w-3.5" /> {title}
      </div>
      <div className="h-36">
        <ResponsiveContainer width="100%" height="100%">{children}</ResponsiveContainer>
      </div>
    </Card>
  )
}

function RunDetail({ id, initialEpisode, onBack }: { id: string; initialEpisode: number | null; onBack: () => void }) {
  const run = useQuery({ queryKey: ['run', id], queryFn: () => api.run(id), refetchInterval: (q) => (q.state.data?.live ? 3000 : false) })
  const episodes = run.data?.episodes ?? []
  const [selected, setSelected] = useState<number | null>(initialEpisode)
  useEffect(() => {
    if (selected == null && episodes.length) {
      setSelected((episodes.find((e) => e.success === false) ?? episodes[0]).episode)
    }
  }, [episodes, selected])
  const ep = episodes.find((e) => e.episode === selected) ?? null
  if (!run.data) return <div className="text-zinc-500">Loading…</div>
  const r = run.data
  return (
    <div className="space-y-4">
      <div className="flex items-center gap-3">
        <Button variant="ghost" onClick={onBack}><ArrowLeft className="h-4 w-4" /> Runs</Button>
        <div>
          <div className="text-lg font-semibold">{r.task}</div>
          <div className="text-xs text-zinc-500">{r.id} · {r.mock ? 'mock' : r.model} · {formatTime(r.started)} · <StatusBadge status={r.status} /></div>
        </div>
        <div className="ml-auto flex items-center gap-3">
          <div className="text-right">
            <div className="text-2xl font-semibold">{r.summary.successes}/{r.summary.attempts}</div>
            <div className="text-xs text-zinc-500">attempts · {r.summary.scene_successes}/{r.summary.scenes} scenes</div>
          </div>
          {r.report && (
            <a href={`/media/${id}/report.html`} target="_blank" rel="noreferrer">
              <Button variant="outline"><FileText className="h-4 w-4" /> Report</Button>
            </a>
          )}
        </div>
      </div>
      <div className="grid grid-cols-12 gap-4">
        <Card className="col-span-12 lg:col-span-3">
          <CardHeader title="Attempts" />
          <div className="max-h-[70vh] space-y-1 overflow-y-auto p-2">
            {episodes.map((e) => (
              <button key={e.episode} onClick={() => setSelected(e.episode)}
                className={clsx('w-full rounded-lg px-3 py-2 text-left text-sm transition', selected === e.episode ? 'bg-zinc-800' : 'hover:bg-zinc-800/50')}>
                <div className="flex items-center gap-2">
                  <ResultMark success={e.success} />
                  <span className="text-zinc-300">#{e.episode}</span>
                  <span className="text-xs text-zinc-500">scene {e.scene + 1} · try {e.attempt}</span>
                  <span className="ml-auto font-mono text-xs text-zinc-500">{e.duration_s.toFixed(1)}s</span>
                </div>
                {e.success === false && (
                  <div className="mt-0.5 text-xs text-red-300">{e.message ?? REASON_LABEL[e.reason ?? 'unknown']}{e.at != null && ` · ${e.at.toFixed(1)} s`}</div>
                )}
              </button>
            ))}
          </div>
        </Card>
        <div className="col-span-12 space-y-4 lg:col-span-9">{ep && <EpisodeView runId={id} ep={ep} key={ep.episode} />}</div>
      </div>
    </div>
  )
}

function EpisodeView({ runId, ep }: { runId: string; ep: EpisodeResult }) {
  const cams = ep.videos.map((v) => v.split('/').pop()!.replace('.mp4', ''))
  const [cam, setCam] = useState(cams.includes('top') ? 'top' : cams[0])
  const video = useRef<HTMLVideoElement>(null)
  const [now, setNow] = useState(0)
  const seekTo = ep.success === false && ep.at != null ? Math.max(0, ep.at - 1) : 0
  const trace = useQuery({ queryKey: ['trace', runId, ep.episode], queryFn: () => api.trace(runId, ep.episode), retry: false })

  const seek = (t: number) => {
    if (video.current) {
      video.current.currentTime = t
      void video.current.play().catch(() => undefined)
    }
  }
  const onLoaded = () => {
    if (video.current) {
      video.current.currentTime = now || seekTo
      if (seekTo || now) void video.current.play().catch(() => undefined)
    }
  }

  return (
    <>
      <Card>
        <CardHeader
          icon={<Video className="h-4 w-4" />}
          title={ep.success === false ? (
            <span className="text-red-300">{ep.message ?? ep.reason}{ep.at != null && <> at <span className="font-mono">{ep.at.toFixed(1)} s</span></>}</span>
          ) : ep.success ? <span className="text-emerald-300">success</span> : 'attempt'}
          right={
            <div className="flex gap-1">
              {cams.map((c) => (
                <button key={c} onClick={() => { setNow(video.current?.currentTime ?? 0); setCam(c) }}
                  className={clsx('rounded-md px-2 py-0.5 text-xs', cam === c ? 'bg-zinc-200 text-zinc-900' : 'bg-zinc-800 text-zinc-400')}>{c}</button>
              ))}
            </div>
          }
        />
        {cam ? (
          <div className="relative bg-black">
            <video ref={video} src={`/media/${runId}/ep${ep.episode}/${cam}.mp4`} controls muted onLoadedMetadata={onLoaded}
              onTimeUpdate={(e) => setNow(e.currentTarget.currentTime)} className="mx-auto max-h-[52vh] w-full" />
            {ep.success === false && ep.at != null && (
              <button onClick={() => seek(Math.max(0, ep.at! - 1))}
                className="absolute right-3 top-3 rounded-md bg-red-500/90 px-2 py-1 text-xs font-medium text-white shadow">
                ⏮ jump to {ep.at.toFixed(1)} s
              </button>
            )}
          </div>
        ) : (
          <div className="p-8 text-center text-sm text-zinc-600">No video — enable the recorder plugin.</div>
        )}
        {ep.timeline && ep.timeline.length > 0 && (
          <div className="flex flex-wrap gap-1.5 border-t border-zinc-800 px-4 py-3">
            {ep.timeline.map((m, i) => (
              <button key={i} onClick={() => seek(m.t)}
                className={clsx('rounded-md px-2 py-1 text-xs hover:brightness-125',
                  ['separated', 'fell', 'collision', 'stalled'].includes(m.kind) ? 'bg-amber-500/15 text-amber-300' : 'bg-zinc-800 text-zinc-300')}>
                <span className="font-mono text-zinc-500">{m.t.toFixed(1)}s</span> {m.message}
              </button>
            ))}
          </div>
        )}
      </Card>
      {trace.data && trace.data.rows.length > 0 && (
        <Card className="p-3">
          <div className="mb-1 px-1 text-xs uppercase tracking-wide text-zinc-500">
            what failure-monitor measured · {trace.data.object} → {trace.data.target} · click to seek
          </div>
          <div className="h-48">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={trace.data.rows} onClick={(s) => { const t = Number(s?.activeLabel); if (Number.isFinite(t)) seek(t) }}>
                <CartesianGrid stroke="#27272a" vertical={false} />
                <XAxis dataKey="t" type="number" domain={[0, 'dataMax']} tick={{ fill: '#71717a', fontSize: 11 }} unit="s" />
                <YAxis tick={{ fill: '#71717a', fontSize: 11 }} width={40} />
                <Tooltip {...TOOLTIP} labelFormatter={(t) => `${Number(t).toFixed(2)} s`} formatter={(v) => Number(v).toFixed(3)} />
                <Legend wrapperStyle={{ fontSize: 11 }} />
                {ep.at != null && ep.success === false && <ReferenceLine x={ep.at} stroke="#ef4444" strokeWidth={2} label={{ value: ep.reason ?? '', fill: '#fca5a5', fontSize: 11, position: 'top' }} />}
                <ReferenceLine x={Number(now.toFixed(2))} stroke="#e4e4e7" strokeDasharray="2 2" />
                <Line dataKey="z" name="object height (m)" stroke="#22c55e" dot={false} strokeWidth={2} isAnimationActive={false} />
                <Line dataKey="d" name="hand → object (m)" stroke="#60a5fa" dot={false} strokeWidth={1.5} isAnimationActive={false} />
                <Line dataKey="dt" name="object → target (m)" stroke="#f59e0b" dot={false} strokeWidth={1.5} isAnimationActive={false} />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </Card>
      )}
      {ep.candidates && ep.candidates.length > 1 && (
        <Card className="px-4 py-3 text-sm text-zinc-400">
          Also seen: {ep.candidates.filter((c) => c.reason !== ep.reason).map((c) => `${REASON_LABEL[c.reason] ?? c.reason} @${c.at.toFixed(1)}s`).join(' · ')}
          <span className="text-zinc-600"> — the first in time is reported</span>
        </Card>
      )}
    </>
  )
}
