import { useEffect, useMemo, useRef, useState } from 'react'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import clsx from 'clsx'
import { Bar, BarChart, CartesianGrid, Line, LineChart, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'
import { ArrowLeft, Clapperboard, Database, FileText, FolderOpen, Trash2 } from 'lucide-react'
import { api, atOf, recoveriesOf, REASON_LABEL, type EpisodeResult, type RunInfo } from '../api'
import { AttemptStrip, Button, ICON_BY_PLUGIN, Outcome, pagesOf, Panel, PanelTitle, PluginIcon, PluginPageDrawer, Tag, when, type Mark, type PluginPage } from '../ui'

const TIP = { contentStyle: { background: '#ffffff', border: '1px solid #e4e2dc', borderRadius: 8, fontSize: 12, color: '#17181a' } }
const BAD_KINDS = new Set(['separated', 'fell', 'collision', 'stalled'])

export default function HistoryPage({ runId, episode, onOpen, onBack }: {
  runId: string | null
  episode: number | null
  onOpen: (id: string, episode?: number) => void
  onBack: () => void
}) {
  return runId ? <RunDetail id={runId} initial={episode} onBack={onBack} onOpen={onOpen} /> : <RunList onOpen={onOpen} />
}

function RunList({ onOpen }: { onOpen: (id: string) => void }) {
  const runs = useQuery({ queryKey: ['runs'], queryFn: api.runs, refetchInterval: 6000 })
  const list = runs.data?.runs ?? []
  const all = list.flatMap((r) => r.episodes ?? [])
  const reasons = useMemo(() => {
    const m: Record<string, number> = {}
    for (const e of all) if (e.success === false) m[e.reason ?? 'unknown'] = (m[e.reason ?? 'unknown'] ?? 0) + 1
    return Object.entries(m).map(([r, n]) => ({ reason: REASON_LABEL[r] ?? r, n })).sort((a, b) => b.n - a.n)
  }, [all])
  const ok = all.filter((e) => e.success).length
  return (
    <div className="space-y-4">
      <div className="grid gap-4 lg:grid-cols-[260px_minmax(0,1fr)]">
        <Panel className="px-4 py-4">
          <p className="text-[13px] text-dim">Across all runs</p>
          <p className="mt-1 text-[40px] font-semibold leading-none">{all.length ? Math.round((100 * ok) / all.length) : 0}<span className="text-[20px] text-dim">%</span></p>
          <p className="mt-2 text-[13px] text-dim">{ok} of {all.length} attempts worked in {list.length} runs</p>
        </Panel>
        <Panel>
          <PanelTitle>Why attempts failed</PanelTitle>
          <div className="h-36 px-2 pb-2">
            {reasons.length ? (
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={reasons} layout="vertical" margin={{ left: 8, right: 16 }}>
                  <XAxis type="number" allowDecimals={false} tick={{ fill: '#6b6f76', fontSize: 11 }} axisLine={false} tickLine={false} />
                  <YAxis type="category" dataKey="reason" tick={{ fill: '#6b6f76', fontSize: 12 }} width={110} axisLine={false} tickLine={false} />
                  <Tooltip {...TIP} cursor={{ fill: '#efede8' }} />
                  <Bar dataKey="n" name="attempts" fill="#c8402f" radius={[0, 3, 3, 0]} isAnimationActive={false} barSize={14} />
                </BarChart>
              </ResponsiveContainer>
            ) : <p className="px-2 pt-6 text-[13px] text-faint">No failures recorded yet.</p>}
          </div>
        </Panel>
      </div>
      <Panel>
        <PanelTitle>Runs</PanelTitle>
        <table className="w-full text-[13px]">
          <thead className="border-y border-line text-left text-[12px] text-faint">
            <tr><th className="px-4 py-2 font-normal">Started</th><th className="px-4 py-2 font-normal">Task</th><th className="px-4 py-2 font-normal">Attempts</th><th className="px-4 py-2 font-normal">Failures</th><th className="px-4 py-2 font-normal">Policy</th></tr>
          </thead>
          <tbody>
            {list.map((r) => (
              <tr key={r.id} tabIndex={0} onClick={() => onOpen(r.id)} onKeyDown={(e) => e.key === 'Enter' && onOpen(r.id)}
                className="cursor-pointer border-b border-line/60 hover:bg-raised/50">
                <td className="whitespace-nowrap px-4 py-2.5 text-dim">{when(r.started)}{r.status === 'running' && <Tag tone="signal">running</Tag>}</td>
                <td className="px-4 py-2.5">{r.task}</td>
                <td className="px-4 py-2.5"><span className="tracking-[2px]">{(r.episodes ?? []).map((e) => <Outcome key={e.episode} success={e.success} />)}</span></td>
                <td className="px-4 py-2.5"><div className="flex flex-wrap gap-1">{Object.entries(r.summary?.reasons ?? {}).map(([k, n]) => <Tag key={k} tone="bad">{REASON_LABEL[k] ?? k}{n > 1 ? ` ×${n}` : ''}</Tag>)}</div></td>
                <td className="px-4 py-2.5 text-dim">{r.mock ? <Tag tone="warn">mock: scripted failures</Tag> : r.model}</td>
              </tr>
            ))}
            {!list.length && <tr><td colSpan={5} className="px-4 py-10 text-center text-faint">No runs yet. Start one on the Run page.</td></tr>}
          </tbody>
        </table>
      </Panel>
    </div>
  )
}

function RunDetail({ id, initial, onBack, onOpen }: { id: string; initial: number | null; onBack: () => void; onOpen: (id: string, ep?: number) => void }) {
  const run = useQuery({ queryKey: ['run', id], queryFn: () => api.run(id), refetchInterval: (q) => (q.state.data?.live ? 3000 : false) })
  const plugins = useQuery({ queryKey: ['plugins'], queryFn: api.plugins })
  for (const p of plugins.data?.plugins ?? []) ICON_BY_PLUGIN[p.name] = p.icon
  const pages = useMemo(() => pagesOf(plugins.data?.plugins), [plugins.data])
  const [page, setPage] = useState<PluginPage | null>(null)
  const episodes = run.data?.episodes ?? []
  const selected = initial ?? (episodes.find((e) => e.success === false) ?? episodes[0])?.episode ?? null
  const ep = episodes.find((e) => e.episode === selected) ?? null
  if (!run.data) return <p className="text-dim">Loading the run…</p>
  const r: RunInfo = run.data
  return (
    <div className="space-y-4">
      <div className="flex flex-wrap items-center gap-3">
        <Button variant="quiet" onClick={onBack}><ArrowLeft className="h-4 w-4" /> All runs</Button>
        <div className="min-w-0">
          <h1 className="text-[18px] font-semibold">{r.task}</h1>
          {r.mock && <div className="mb-1"><Tag tone="warn" title="The mock policy fails on purpose so the plugins can be tested; no model is called">mock: scripted failures</Tag></div>}
          <p className="text-[12px] text-dim">{when(r.started)}, {r.mock ? 'mock policy' : r.model}, {r.summary.successes} of {r.summary.attempts} attempts worked</p>
        </div>
        <div className="ml-auto flex gap-2">
          <Button onClick={() => api.reveal(id)}><FolderOpen className="h-4 w-4" /> Open folder</Button>
          <DeleteRun id={id} live={Boolean(r.live)} onDone={onBack} />
          {r.report && <a href={`/media/${id}/${r.report}`} target="_blank" rel="noreferrer"><Button><FileText className="h-4 w-4" /> Report</Button></a>}
        </div>
      </div>
      <Harvest id={id} />
      {r.recap?.text && (
        <Panel className="flex gap-3 px-4 py-3">
          <Clapperboard className="mt-0.5 h-4 w-4 shrink-0 text-ink" strokeWidth={1.75} />
          <p className="text-[14px] leading-relaxed">{r.recap.text}</p>
        </Panel>
      )}
      <div className="grid gap-4 lg:grid-cols-[250px_minmax(0,1fr)]">
        <Panel className="self-start">
          <PanelTitle>Attempts</PanelTitle>
          <ul className="border-t border-line pb-1">
            {episodes.map((e) => (
              <li key={e.episode}>
                <button onClick={() => onOpen(id, e.episode)} className={clsx('w-full px-4 py-2 text-left', selected === e.episode ? 'bg-raised' : 'hover:bg-raised/50')}>
                  <div className="flex items-center gap-2 text-[13px]">
                    <Outcome success={e.success} />
                    <span>Scene {e.scene + 1}{e.attempt > 1 ? `, retry ${e.attempt - 1}` : ''}</span>
                    {recoveriesOf(e) > 0 && <Tag tone="ok">recovered{recoveriesOf(e) > 1 ? ` ×${recoveriesOf(e)}` : ''}</Tag>}
                    <span className="ml-auto text-[12px] text-faint">{e.duration_s.toFixed(0)} s</span>
                  </div>
                  {e.success == null && <LabelButtons runId={id} episode={e.episode} />}
                  {e.labeled_by === 'person' && <p className="pl-5 text-[11.5px] text-dim">marked by a person</p>}
                  {e.success === false && <p className="mt-0.5 truncate pl-5 text-[12px] text-bad">{REASON_LABEL[e.reason ?? 'unknown'] ?? e.reason}{atOf(e) != null && ` at ${atOf(e)!.toFixed(1)} s`}</p>}
                </button>
              </li>
            ))}
          </ul>
        </Panel>
        {ep ? <EpisodeView key={ep.episode} runId={id} ep={ep} seconds={r.seconds} pages={pages}
          onPage={(plugin) => setPage({ plugin, runId: id, episode: ep.episode })} /> : <p className="text-faint">This run has no finished attempts.</p>}
        <PluginPageDrawer page={page} onClose={() => setPage(null)} />
      </div>
    </div>
  )
}

function Harvest({ id }: { id: string }) {
  const queryClient = useQueryClient()
  const harvest = useQuery({ queryKey: ['harvest', id], queryFn: () => api.harvest(id) })
  const check = useMutation({
    mutationFn: () => api.validateHarvest(id),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ['harvest', id] }),
  })
  const h = harvest.data
  if (!h || (h.success === 0 && h.failed === 0)) return null
  const v = h.validation
  return (
    <Panel className="flex flex-wrap items-center gap-x-4 gap-y-2 px-4 py-3">
      <Database className="h-4 w-4 shrink-0 text-ok" strokeWidth={1.75} />
      <p className="text-[14px]">
        <span className="font-medium text-ok">{h.success} successful episode{h.success === 1 ? '' : 's'}</span> ready for fine-tuning
        <span className="text-dim">{h.failed ? `, ${h.failed} failed kept for analysis` : ''}</span>
      </p>
      <div className="ml-auto flex items-center gap-3">
        {v && (v.ok ? (
          <span className="text-[13px] text-ok">validate_dataset passed: {v.episodes} episodes, {v.frames} frames, {v.fps} fps</span>
        ) : <span className="max-w-[48ch] text-[13px] text-bad">validate_dataset: {v.error}</span>)}
        {h.success > 0 && (
          <Button onClick={() => check.mutate()} disabled={check.isPending}>
            {check.isPending ? 'Checking…' : v ? 'Check again' : 'Check with validate_dataset'}
          </Button>
        )}
      </div>
      <p className="basis-full text-[12px] text-faint">
        Local check only, nothing is uploaded. To gather several runs: <code className="text-dim">dreamkit harvest export {id}</code>
      </p>
    </Panel>
  )
}

/** No world, nobody judged it: a person says ✓ or ✗ (the harvester's dataset follows). */
function LabelButtons({ runId, episode }: { runId: string; episode: number }) {
  const queryClient = useQueryClient()
  const mark = useMutation({
    mutationFn: (success: boolean) => api.label(runId, episode, success),
    onSuccess: () => { queryClient.invalidateQueries({ queryKey: ['run', runId] }); queryClient.invalidateQueries({ queryKey: ['harvest', runId] }) },
  })
  return (
    <div className="mt-1 flex items-center gap-1.5 pl-5" onClick={(e) => e.stopPropagation()}>
      <span className="text-[11.5px] text-warn">outcome unknown</span>
      <button disabled={mark.isPending} onClick={() => mark.mutate(true)}
        className="rounded-md border border-line bg-panel px-1.5 text-[12px] text-ok hover:bg-ok-soft" title="It worked">✓</button>
      <button disabled={mark.isPending} onClick={() => mark.mutate(false)}
        className="rounded-md border border-line bg-panel px-1.5 text-[12px] text-bad hover:bg-bad-soft" title="It failed">✗</button>
    </div>
  )
}

function DeleteRun({ id, live, onDone }: { id: string; live: boolean; onDone: () => void }) {
  const queryClient = useQueryClient()
  const [sure, setSure] = useState(false)
  const remove = useMutation({
    mutationFn: () => api.deleteRun(id),
    onSuccess: () => { queryClient.invalidateQueries({ queryKey: ['runs'] }); onDone() },
  })
  if (live) return null
  return sure ? (
    <Button variant="stop" onClick={() => remove.mutate()} disabled={remove.isPending} onBlur={() => setSure(false)}>
      <Trash2 className="h-4 w-4" /> Delete videos and results
    </Button>
  ) : (
    <Button variant="quiet" onClick={() => setSure(true)}><Trash2 className="h-4 w-4" /> Delete run</Button>
  )
}

function EpisodeView({ runId, ep, seconds, pages, onPage }: {
  runId: string; ep: EpisodeResult; seconds: number
  pages: Set<string>  // plugins with a page of their own (§7a)
  onPage: (plugin: string) => void
}) {
  // this attempt's lines from plugins that ship a page: their badges, or their name; a click opens the page
  const pageRows = [...pages].filter((p) => ep.ui?.[p] || (ep as unknown as Record<string, unknown>)[p])
    .map((p) => ({ plugin: p, text: (ep.ui?.[p]?.badges ?? []).map((b) => b.text).join(' · ') || `${p}: open the page` }))
  // recorder/ep003_top.mp4 (standard) or ep3/top.mp4 (older runs): camera name -> path in the run folder
  const videoOf: Record<string, string> = Object.fromEntries(ep.videos.map((v) => {
    const stem = v.split('/').pop()!.replace('.mp4', '')
    return [stem.includes('_') ? stem.split('_').pop()! : stem, v]
  }).filter(([c]) => c !== 'recap'))
  const cams = Object.keys(videoOf)
  const [cam, setCam] = useState(cams.includes('top') ? 'top' : cams[0])
  const [mode, setMode] = useState<'full' | 'recap'>('full')
  const video = useRef<HTMLVideoElement>(null)
  const [now, setNow] = useState(0)
  const at = atOf(ep)
  const failed = ep.success === false && at != null
  const start = failed ? Math.max(0, at! - 1) : 0
  const trace = useQuery({ queryKey: ['trace', runId, ep.episode], queryFn: () => api.trace(runId, ep.episode), retry: false })

  const marks: Mark[] = (ep.timeline ?? []).map((m) => ({ t: m.t, label: m.message, tone: BAD_KINDS.has(m.kind) ? 'warn' : 'info' }))
  if (ep.success) marks.push({ t: ep.duration_s, label: 'task done', tone: 'ok' })
  const seek = (t: number) => {
    setMode('full')
    const v = video.current
    if (!v) return
    v.currentTime = t
    void v.play().catch(() => undefined)
  }
  const src = mode === 'recap' && ep.recap?.video ? `/media/${runId}/${ep.recap.video}` : cam ? `/media/${runId}/${videoOf[cam]}` : null

  useEffect(() => {
    // follow the video with a cheap timer instead of re-rendering on every timeupdate
    const id = setInterval(() => video.current && setNow(video.current.currentTime), 200)
    return () => clearInterval(id)
  }, [])

  return (
    <div className="min-w-0 space-y-4">
      <Panel>
        <div className="flex flex-wrap items-center gap-3 px-4 pt-3 pb-2">
          <h2 className={clsx('text-[15px] font-medium', failed ? 'text-bad' : ep.success ? 'text-ok' : 'text-ink')}>
            {failed ? `${ep.message ?? REASON_LABEL[ep.reason ?? 'unknown']} at ${at!.toFixed(1)} s` : ep.success ? `Done in ${ep.duration_s.toFixed(1)} s` : 'Outcome unknown: mark it on the left'}
          </h2>
          {failed && (ep.details as { source?: string } | undefined)?.source && (
            <Tag tone="plain" title="What the verdict rests on: the robot's own signals, or the simulator's world">
              from {(ep.details as { source?: string }).source === 'world' ? 'the simulator world' : 'robot signals'}
            </Tag>
          )}
          <div className="ml-auto flex items-center gap-1 rounded-md bg-base p-0.5 text-[12px]">
            {ep.recap?.video && (
              <button onClick={() => setMode(mode === 'recap' ? 'full' : 'recap')} className={clsx('rounded px-2 py-0.5', mode === 'recap' ? 'bg-signal text-base' : 'text-dim')}>Recap</button>
            )}
            {cams.map((c) => (
              <button key={c} onClick={() => { setMode('full'); setCam(c) }} className={clsx('rounded px-2 py-0.5', mode === 'full' && cam === c ? 'bg-raised text-ink' : 'text-dim')}>{c}</button>
            ))}
          </div>
        </div>
        {src ? (
          <video key={src} ref={video} src={src} controls muted playsInline className="max-h-[54vh] w-full bg-black"
            onLoadedMetadata={(e) => {
              if (mode === 'full') e.currentTarget.currentTime = now || start
              if (mode === 'recap' || start) void e.currentTarget.play().catch(() => undefined)
            }} />
        ) : (
          <p className="px-4 py-10 text-center text-[13px] text-faint">No video for this attempt. Switch on the recorder plugin.</p>
        )}
        {mode === 'full' && (
          <div className="px-4 pt-4 pb-2">
            <AttemptStrip duration={Math.max(ep.duration_s, Math.min(seconds, ep.duration_s + 2))} marks={marks}
              verdict={failed ? { t: at!, label: ep.reason ?? '' } : null} playhead={now} onSeek={seek} />
          </div>
        )}
        {ep.recap?.headline && <p className="border-t border-line px-4 py-3 text-[13px] text-dim">{ep.recap.headline}</p>}
      </Panel>

      {((ep.timeline?.length ?? 0) > 0 || failed || pageRows.length > 0) && (
        <Panel>
          <PanelTitle>Moments</PanelTitle>
          {pageRows.map((row) => (
            <button key={row.plugin} onClick={() => onPage(row.plugin)} title={`Open ${row.plugin}'s page`}
              className="flex w-full items-center gap-3 border-t border-line px-4 py-2 text-left text-[13px] hover:bg-raised/50">
              <PluginIcon name={row.plugin} className="text-dim" />
              <span className="min-w-0 flex-1 truncate">{row.text}</span>
              <span className="text-[11px] text-faint">{row.plugin}</span>
              <span className="text-[16px] text-faint" aria-hidden>›</span>
            </button>
          ))}
          {failed && (
            <button onClick={() => seek(start)} className="flex w-full gap-3 border-y border-line bg-bad-soft px-4 py-2 text-left text-[13px] hover:brightness-95">
              <span className="w-12 shrink-0 text-right text-bad">{at!.toFixed(1)} s</span>
              <span className="font-medium text-bad">Verdict: {ep.message ?? REASON_LABEL[ep.reason ?? 'unknown']}</span>
            </button>
          )}
          <ol className={clsx(!failed && 'border-t border-line')}>
            {[...(ep.timeline ?? [])].sort((a, b) => a.t - b.t).map((m, i) => (
              <li key={i}>
                <button onClick={() => seek(Math.max(0, m.t - 0.5))} className="flex w-full gap-3 px-4 py-1.5 text-left text-[13px] hover:bg-raised/50">
                  <span className="w-12 shrink-0 text-right text-faint">{m.t.toFixed(1)} s</span>
                  <span className={BAD_KINDS.has(m.kind) ? 'text-warn' : 'text-ink'}>{m.message}</span>
                </button>
              </li>
            ))}
          </ol>
        </Panel>
      )}

      {trace.data && trace.data.rows.length > 1 && (
        <Panel>
          <PanelTitle>What failure-monitor measured</PanelTitle>
          <div className="h-44 px-2 pb-3">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={trace.data.rows} onClick={(s) => { const t = Number(s?.activeLabel); if (Number.isFinite(t)) seek(t) }}>
                <CartesianGrid stroke="#e4e2dc" vertical={false} />
                <XAxis dataKey="t" type="number" domain={[0, 'dataMax']} tick={{ fill: '#6b6f76', fontSize: 11 }} unit=" s" axisLine={false} tickLine={false} />
                <YAxis domain={[0, 'auto']} tickFormatter={(v) => `${Math.round(v * 100)}`} unit=" cm" tick={{ fill: '#6b6f76', fontSize: 11 }} width={48} axisLine={false} tickLine={false} />
                <Tooltip {...TIP} labelFormatter={(t) => `${Number(t).toFixed(1)} s`} formatter={(v) => `${(100 * Number(v)).toFixed(1)} cm`} />
                {failed && <ReferenceLine x={at} stroke="#c8402f" strokeWidth={2} />}
                <ReferenceLine x={Number(now.toFixed(1))} stroke="#2f5bd3" />
                <Line dataKey="d" name="hand to object" stroke="#2f8fb3" dot={false} strokeWidth={1.75} isAnimationActive={false} />
                <Line dataKey="z" name="object lifted" stroke="#1f8a5b" dot={false} strokeWidth={1.75} isAnimationActive={false} />
                <Line dataKey="dt" name="object to target" stroke="#8a5cc2" dot={false} strokeWidth={1.5} isAnimationActive={false} />
              </LineChart>
            </ResponsiveContainer>
          </div>
          <p className="px-4 pb-3 text-[12px] text-faint">
            <span className="text-steel">Blue</span> hand to object, <span className="text-ok">green</span> how high the object is, <span className="text-[#8a5cc2]">violet</span> object to target. Click the chart to jump the video there.
          </p>
        </Panel>
      )}
    </div>
  )
}
