import { useEffect, useMemo, useRef, useState, type ReactNode } from 'react'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import clsx from 'clsx'
import { CartesianGrid, Line, LineChart, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'
import { ArrowLeft, FolderOpen, Trash2 } from 'lucide-react'
import { api, ApiError, atOf, recoveriesOf, type EpisodeResult, type RunInfo } from '../api'
import { count, POLICY_FILTER, reasonLabel, SOURCE_LABEL, TERM } from '../labels'
import { AttemptStrip, Button, Fold, ICON_BY_PLUGIN, MoreMenu, Outcome, pagesOf, Panel, PanelTitle, PluginIcon, PluginPageDrawer, Tag, when, type Mark, type MenuItem, type PluginPage } from '../ui'

const TIP = { contentStyle: { background: 'var(--dk-panel)', border: '1px solid var(--dk-line)', borderRadius: 8, fontSize: 12, color: 'var(--dk-ink)' } }
const BAD_KINDS = new Set(['separated', 'fell', 'collision', 'stalled'])

export default function HistoryPage({ runId, episode, onOpen, onBack }: {
  runId: string | null
  episode: number | null
  onOpen: (id: string, episode?: number) => void
  onBack: () => void
}) {
  return runId ? <RunDetail id={runId} initial={episode} onBack={onBack} onOpen={onOpen} /> : <RunList onOpen={onOpen} />
}

type Filter = 'dream' | 'scripted' | 'all'

/** An attempt that counts: it ran, and dreamkit could read what success meant. The rest is shown grey as invalid. */
const valid = (e: EpisodeResult) => e.task_readable !== false
const ms = (v: number) => (v >= 1000 ? `${(v / 1000).toFixed(1)} s` : `${Math.round(v)} ms`)

const DASH = '—'
const field = (e: EpisodeResult, key: string) => (e as unknown as Record<string, Record<string, unknown> | undefined>)[key]
/** A real run bills the model's time; mock and scripted ones bill nothing. */
const isReal = (r: RunInfo) => r.kind === 'dream' && !r.mock
const costText = (r: RunInfo) => (!isReal(r) ? 'free' : r.cost_usd == null ? DASH : `$${r.cost_usd.toFixed(2)}`)
/** What a run's camera judge (vision-judge) did: off, judged (and agreed), or not judged and why. */
const GO_PLUGINS = (name: string) => <a href={`#/plugins/${name}`} className="text-signal hover:underline">switch it on in Plugins</a>
/** A failure of the camera judge, and what to do about it. */
function judgeHint(why: string): string {
  if (/quota|429|exhausted/i.test(why)) return 'the Gemini quota ran out: it resets tomorrow, or add ANTHROPIC_API_KEY to .env'
  if (/GEMINI_API_KEY|api key/i.test(why)) return 'no model key: add GEMINI_API_KEY to .env'
  return why
}
function judgeText(r: RunInfo): ReactNode {
  if (!(r.plugins ?? []).includes('vision-judge')) return <>off in this run · {GO_PLUGINS('vision-judge')}</>
  const eps = (r.episodes ?? []).filter(valid)
  const judged = eps.filter((e) => typeof field(e, 'vision-judge')?.success === 'boolean' && typeof e.success === 'boolean')
  const agree = judged.filter((e) => field(e, 'vision-judge')!.success === e.success).length
  const why = [...new Set(eps.map((e) => field(e, 'vision-judge')?.error as string | undefined).filter(Boolean))]
  const parts = [judged.length ? `${agree} of ${judged.length} agree with success-check` : '']
  if (why.length) parts.push(`${eps.length - judged.length} not judged: ${[...new Set(why.map((w) => judgeHint(w ?? "")))].join('; ')}`)
  return parts.filter(Boolean).join(' · ') || 'no verdict recorded'
}
function savesText(r: RunInfo): ReactNode {
  if (!(r.plugins ?? []).includes('recovery')) return <>off in this run · {GO_PLUGINS('recovery')}</>
  const helped = (r.episodes ?? []).filter((e) => recoveriesOf(e) > 0)
  return helped.length ? `${helped.filter((e) => e.success).length} of ${helped.length} attempts it stepped in` : 'never needed'
}

/** One run in numbers, for the comparison. */
function statsOf(r: RunInfo) {
  const st = r.stats  // dreamkit.stats on the server: the same numbers as reliability and the Data tab
  const done = (r.episodes ?? []).filter((e) => valid(e) && e.success)
  return {
    rate: st?.attempts ? st.success_rate : null, ok: st?.successes ?? 0, n: st?.attempts ?? 0,
    time: done.length ? done.reduce((a, e) => a + e.duration_s, 0) / done.length : null,
    cost: costText(r), latency: st?.mean_latency_ms == null ? DASH : ms(st.mean_latency_ms),
    reason: st?.top_reason ? reasonLabel(st.top_reason) : DASH,
  }
}

/** Two runs side by side, and what changed between them. */
function Compare({ a, b, onClear }: { a: RunInfo; b: RunInfo; onClear: () => void }) {
  const [x, y] = [statsOf(a), statsOf(b)]
  const pct = (v: number | null) => (v == null ? DASH : `${Math.round(100 * v)}%`)
  const rows: [string, string, string][] = [
    ['Success', `${pct(x.rate)} (${x.ok} of ${x.n})`, `${pct(y.rate)} (${y.ok} of ${y.n})`],
    ['Time to success', x.time == null ? DASH : `${x.time.toFixed(1)} s`, y.time == null ? DASH : `${y.time.toFixed(1)} s`],
    ['Most failures', x.reason, y.reason],
    ['Cost', x.cost, y.cost],
    ['Model latency', x.latency, y.latency],
    ['Policy', a.mock ? 'mock' : a.model || DASH, b.mock ? 'mock' : b.model || DASH],
  ]
  const delta = x.rate != null && y.rate != null ? Math.round(100 * (y.rate - x.rate)) : null
  return (
    <Panel>
      <PanelTitle right={<Button variant="quiet" onClick={onClear}>Clear</Button>}>Comparing two runs</PanelTitle>
      <table className="mx-5 mb-3 w-[calc(100%-40px)] table-fixed text-[14px]">
        <thead className="text-left text-[13px] text-dim">
          <tr><th className="w-[170px] py-1.5 font-normal" /><th className="py-1.5 font-normal">{when(a.started)} · {a.task}</th><th className="py-1.5 font-normal">{when(b.started)} · {b.task}</th></tr>
        </thead>
        <tbody>
          {rows.map(([k, va, vb]) => (
            <tr key={k} className="border-t border-line/60"><td className="py-1.5 text-dim">{k}</td><td className="py-1.5">{va}</td><td className="py-1.5">{vb}</td></tr>
          ))}
        </tbody>
      </table>
      {delta != null && (
        <p className="px-5 pb-4 text-[14px]">
          The later run {delta === 0 ? 'succeeded as often' : <b className={delta > 0 ? 'text-ok' : 'text-bad'}>{delta > 0 ? `+${delta}` : delta} points</b>}
          {Math.min(x.n, y.n) < 20 && <span className="text-dim"> · with fewer than 20 attempts each this can be luck; reliability tells for sure</span>}
        </p>
      )}
    </Panel>
  )
}

/** The success rate of each run, oldest first: did it get better? */
function Trend({ runs }: { runs: RunInfo[] }) {
  const points = runs.filter((r) => (r.episodes ?? []).some(valid)).slice().reverse().map((r) => {
    const st = statsOf(r)
    return { when: when(r.started), rate: Math.round(100 * (st.rate ?? 0)), n: st.n }
  })
  if (points.length < 2) return null
  return (
    <Panel>
      <PanelTitle>Success over time</PanelTitle>
      <div className="h-[150px] px-2 pb-3">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={points} margin={{ left: 0, right: 16, top: 8 }}>
            <CartesianGrid stroke="var(--dk-line)" vertical={false} />
            <XAxis dataKey="when" tick={{ fill: 'var(--dk-dim)', fontSize: 11 }} interval="preserveStartEnd" axisLine={false} tickLine={false} />
            <YAxis domain={[0, 100]} ticks={[0, 50, 100]} unit="%" width={44} tick={{ fill: 'var(--dk-dim)', fontSize: 11 }} axisLine={false} tickLine={false} />
            <Tooltip {...TIP} formatter={(v, _n, p) => [`${v}% of ${(p.payload as { n: number }).n} attempts`, 'success']} />
            <Line dataKey="rate" stroke="var(--dk-ok)" strokeWidth={2} dot={{ r: 3 }} isAnimationActive={false} />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </Panel>
  )
}

function RunList({ onOpen }: { onOpen: (id: string) => void }) {
  const runs = useQuery({ queryKey: ['runs'], queryFn: api.runs, refetchInterval: 6000 })
  const [filter, setFilter] = useState<Filter>(() => {
    const saved = localStorage.getItem('dk-history-filter') as Filter | null
    return saved && saved in POLICY_FILTER ? saved : 'dream'
  })
  const choose = (f: Filter) => { localStorage.setItem('dk-history-filter', f); setFilter(f) }
  const [picked, setPicked] = useState<string[]>([])  // up to two runs to compare
  const pick = (id: string) => setPicked((p) => (p.includes(id) ? p.filter((x) => x !== id) : [...p, id].slice(-2)))
  const summary = useQuery({ queryKey: ['stats', filter], queryFn: () => api.stats(filter), refetchInterval: 6000 })
  if (runs.isError) return <LoadError error={runs.error} what="the runs" />
  const everything = runs.data?.runs ?? []
  const list = everything.filter((r) => filter === 'all' || r.kind === filter)
  // every number here comes from the server (dreamkit.stats): the same as reliability and the Data tab
  const st = summary.data
  const nJudged = st?.attempts ?? 0
  const ok = st?.successes ?? 0
  const byLabel = Object.entries(st?.failures ?? {}).reduce<Record<string, number>>((acc, [code, n]) => {
    acc[reasonLabel(code)] = (acc[reasonLabel(code)] ?? 0) + n
    return acc
  }, {})
  const reasons = Object.entries(byLabel).map(([reason, n]) => ({ reason, n })).sort((a, b) => b.n - a.n)
  const invalid = (st?.invalid ?? 0) + (st?.empty_runs ?? 0)
  const counts = { dream: everything.filter((r) => r.kind === 'dream').length, scripted: everything.filter((r) => r.kind === 'scripted').length, all: everything.length }
  // money only where it was spent: real runs, whichever filter is on
  const real = everything.filter(isReal)
  const spent = st?.real.cost_usd ?? 0
  const realOk = st?.real.successes ?? 0
  const top = reasons[0]
  const topLat = st?.top_reason === 'slow_model' ? st.mean_latency_ms : null
  return (
    <div className="space-y-4">
      <div className="flex flex-wrap items-center gap-3">
        <div className="flex rounded-lg border border-line bg-panel p-0.5 text-[13px]" role="tablist" aria-label="Which runs">
          {(['dream', 'scripted', 'all'] as const).map((f) => (
            <button key={f} role="tab" aria-selected={filter === f} onClick={() => choose(f)}
              className={clsx('rounded-md px-3 py-1', filter === f ? 'bg-raised font-medium text-ink' : 'text-dim hover:text-ink')}>
              {POLICY_FILTER[f]} <span className="text-faint">{counts[f]}</span>
            </button>
          ))}
        </div>
        <p className="text-[13px] text-dim">
          {filter === 'dream' ? 'Runs where Dream-1 drove the robot.' : filter === 'scripted' ? 'Mock and scripted policies: they test the machinery, not the model.' : 'Every run.'}
        </p>
      </div>

      {list.length > 0 && (
        <div className="grid gap-4 lg:grid-cols-[minmax(0,1fr)_minmax(0,1.4fr)]">
          <Panel className="px-6 py-5">
            <p className="text-[14px] text-dim">{count(nJudged, TERM.attempt)} in {count(list.length, TERM.run)}</p>
            <p className="mt-1 flex items-baseline gap-3">
              <span className="text-[44px] font-semibold leading-none tabular-nums">{Math.round(100 * (st?.success_rate ?? 0))}%</span>
              <span className="text-[15px] text-dim">worked ({ok} of {nJudged})</span>
            </p>
            <p className="mt-3 text-[15px]">
              {top ? <>Most failures: <b className="text-bad">{top.reason}</b>{topLat != null ? `, avg ${ms(topLat)}` : ''} ({top.n} of {nJudged - ok})</>
                : nJudged ? 'No failures.' : 'No attempt was judged yet.'}
            </p>
            <p className="mt-3 border-t border-line pt-3 text-[13px] text-dim">
              {real.length > 0 && <>Spent <b className="text-ink tabular-nums">${spent.toFixed(2)}</b>{realOk ? <> · ${(spent / realOk).toFixed(2)} per success</> : null} <span className="text-faint">(real runs only)</span>. </>}
              {invalid > 0 && <span title="Runs without attempts, and attempts whose task names no object and target dreamkit can read (nothing judged them)">{count(invalid, 'invalid attempt')} not counted.</span>}
            </p>
          </Panel>
          <Panel>
            <PanelTitle>Why attempts failed</PanelTitle>
            {!reasons.length ? <p className="px-5 pb-4 text-[14px] text-dim">No failures in these runs.</p> : (
              <ul className="px-5 pb-4 text-[13.5px]">
                {reasons.slice(0, 6).map((x) => (
                  <li key={x.reason} className="grid grid-cols-[minmax(0,150px)_1fr_32px] items-center gap-3 py-1">
                    <span className="truncate">{x.reason}</span>
                    <span className="h-2 rounded bg-raised"><span className="block h-2 rounded bg-bad" style={{ width: `${(100 * x.n) / reasons[0].n}%` }} /></span>
                    <span className="text-right tabular-nums">{x.n}</span>
                  </li>
                ))}
                {reasons.length > 6 && <li className="pt-1 text-[12px] text-faint">and {reasons.length - 6} more</li>}
              </ul>
            )}
          </Panel>
        </div>
      )}

      {list.length > 1 && <Trend runs={list} />}
      {picked.length === 2 && (() => {
        const [a, b] = picked.map((id) => everything.find((r) => r.id === id)).filter(Boolean).sort((p, q) => p!.started - q!.started) as RunInfo[]
        return a && b ? <Compare a={a} b={b} onClear={() => setPicked([])} /> : null
      })()}
      {picked.length === 1 && <p className="text-[13px] text-dim">Pick one more run to compare.</p>}
      <Panel>
        {!list.length ? (
          <div className="px-6 py-8">
            <p className="text-[15px]">{everything.length ? `No ${POLICY_FILTER[filter]} runs yet.` : 'No runs yet.'}</p>
            {everything.length && filter !== 'all'
              ? <button className="mt-2 text-[14px] text-signal hover:underline" onClick={() => choose('all')}>Show all runs</button>
              : <a href="#/live" className="mt-2 inline-block text-[14px] text-signal hover:underline">Start one: see Live</a>}
          </div>
        ) : (
          <table className="w-full table-fixed text-[14px]">
            <colgroup><col className="w-[40px]" /><col className="w-[150px]" /><col /><col className="w-[170px]" /><col className="w-[90px]" /><col className="w-[80px]" /><col className="w-[200px]" /></colgroup>
            <thead className="border-b border-line text-left text-[13px] text-dim">
              <tr><th className="pl-4 font-normal" title="Tick two runs to compare them" /><th className="px-3 py-2.5 font-normal">Started</th>{['Task', 'Result', 'Duration', 'Cost', 'Failure reason'].map((h) => <th key={h} className="px-3 py-2.5 font-normal">{h}</th>)}</tr>
            </thead>
            <tbody>
              {list.map((r) => {
                const eps = r.episodes ?? []
                const rs = r.stats
                const others = Object.keys(rs?.failures ?? {}).length - 1
                return (
                  <tr key={r.id} tabIndex={0} onClick={() => onOpen(r.id)} onKeyDown={(e) => e.key === 'Enter' && onOpen(r.id)}
                    className="cursor-pointer border-b border-line/60 last:border-0 hover:bg-raised/50">
                    <td className="pl-4" onClick={(e) => e.stopPropagation()}>
                      <input type="checkbox" aria-label={`Compare the run of ${when(r.started)}`} checked={picked.includes(r.id)} onChange={() => pick(r.id)} />
                    </td>
                    <td className="py-2.5 pr-3 pl-3 whitespace-nowrap text-dim">{when(r.started)}</td>
                    <td className="truncate px-3 py-2.5" title={r.task}>{r.task || <span className="text-faint">no task</span>}</td>
                    <td className="truncate px-3 py-2.5 tracking-[2px]">
                      {r.status === 'running' && <span className="mr-2 tracking-normal"><Tag tone="signal">running</Tag></span>}
                      {eps.map((e) => valid(e) ? <Outcome key={e.episode} success={e.success} />
                        : <span key={e.episode} className="text-faint" title="invalid: dreamkit cannot read an object and a target in this task">–</span>)}
                      {!eps.length && <span className="tracking-normal text-faint" title="the run recorded no attempt">invalid</span>}
                    </td>
                    <td className="px-3 py-2.5 whitespace-nowrap text-dim tabular-nums">{eps.length ? `${eps.reduce((a, e) => a + (e.duration_s ?? 0), 0).toFixed(0)} s` : DASH}</td>
                    <td className={clsx('px-3 py-2.5 whitespace-nowrap tabular-nums', isReal(r) ? 'text-ink' : 'text-faint')}>{costText(r)}</td>
                    <td className="truncate px-3 py-2.5" title={rs?.attempts ? `${rs.successes} of ${rs.attempts} worked` : undefined}>
                      {rs?.top_reason ? <span className="text-bad">{reasonLabel(rs.top_reason)}{others > 0 ? ` +${others}` : ''}</span> : <span className="text-faint">{DASH}</span>}
                    </td>
                  </tr>
                )
              })}
            </tbody>
          </table>
        )}
      </Panel>
    </div>
  )
}

function LoadError({ error, what }: { error: unknown; what: string }) {
  return (
    <div className="pt-6">
      <p className="text-bad">Could not load {what}: {(error as Error).message}</p>
      {error instanceof ApiError && (
        <details className="mt-2 text-[13px] text-dim"><summary className="cursor-pointer">Details</summary><pre className="mt-1 font-mono whitespace-pre-wrap">{error.details}</pre></details>
      )}
    </div>
  )
}

function RunDetail({ id, initial, onBack, onOpen }: { id: string; initial: number | null; onBack: () => void; onOpen: (id: string, ep?: number) => void }) {
  const queryClient = useQueryClient()
  const run = useQuery({ queryKey: ['run', id], queryFn: () => api.run(id), refetchInterval: (q) => (q.state.data?.live ? 3000 : false) })
  const plugins = useQuery({ queryKey: ['plugins'], queryFn: api.plugins })
  for (const p of plugins.data?.plugins ?? []) ICON_BY_PLUGIN[p.name] = p.icon
  const pages = useMemo(() => pagesOf(plugins.data?.plugins), [plugins.data])
  const [page, setPage] = useState<PluginPage | null>(null)
  const [deleting, setDeleting] = useState(false)
  const remove = useMutation({
    mutationFn: () => api.deleteRun(id),
    onSuccess: () => { queryClient.invalidateQueries({ queryKey: ['runs'] }); onBack() },
  })
  const episodes = run.data?.episodes ?? []
  const selected = initial ?? (episodes.find((e) => e.success === false) ?? episodes[0])?.episode ?? null
  const ep = episodes.find((e) => e.episode === selected) ?? null
  if (run.isError) return <LoadError error={run.error} what="this run" />
  if (!run.data) return <p className="text-dim">Loading the run…</p>
  const r: RunInfo = run.data
  const menu: MenuItem[] = [
    ...(r.report ? [{ label: 'Open the report', href: `/media/${id}/${r.report}`, onSelect: () => undefined }] : []),
    ...(r.live ? [] : [{ label: 'Delete run…', tone: 'bad' as const, onSelect: () => setDeleting(true) }]),
  ]
  return (
    <div className="space-y-4">
      <div className="flex flex-wrap items-center gap-3">
        <Button variant="quiet" onClick={onBack}><ArrowLeft className="h-4 w-4" /> All runs</Button>
        <div className="min-w-0 flex-1">
          <h1 className="truncate text-[20px] font-semibold" title={r.task}>{r.task || 'Run without a task'}</h1>
          <p className="text-[13px] text-dim">
            {when(r.started)} · {r.mock ? 'mock policy' : r.model || 'unknown policy'} · {SOURCE_LABEL[r.source] ?? r.source}
            {r.mock && <span title="The mock fails on purpose so the plugins can be tested; no model is called"> · scripted failures</span>}
          </p>
        </div>
        <Button onClick={() => api.reveal(id)}><FolderOpen className="h-4 w-4" /> Open folder</Button>
        <MoreMenu items={menu} label="More for this run" />
      </div>
      {deleting && (
        <Panel className="flex flex-wrap items-center gap-3 border-bad/40 px-5 py-3">
          <p className="flex-1 text-[14px]">Delete this run's videos, datasets and results? This cannot be undone.</p>
          <Button variant="quiet" onClick={() => setDeleting(false)}>Keep it</Button>
          <Button variant="stop" disabled={remove.isPending} onClick={() => remove.mutate()}><Trash2 className="h-4 w-4" /> Delete run</Button>
        </Panel>
      )}

      <p className="text-[17px]">
        {r.stats?.attempts ? <><b>{r.stats.successes} of {count(r.stats.attempts, TERM.attempt)}</b> worked</>
          : <span className="text-dim">No attempt in this run could be judged.</span>}
        {r.stats?.top_reason && <span className="text-dim"> · most failures: {reasonLabel(r.stats.top_reason)}</span>}
      </p>

      <RunFacts r={r} onPage={(plugin) => setPage({ plugin, runId: id, episode: null })} />

      <div className="grid gap-4 lg:grid-cols-[240px_minmax(0,1fr)]">
        <Panel className="self-start">
          <PanelTitle>Attempts</PanelTitle>
          <ul className="border-t border-line pb-1">
            {episodes.map((e) => (
              <li key={e.episode}>
                <button onClick={() => onOpen(id, e.episode)} className={clsx('w-full px-4 py-2 text-left', selected === e.episode ? 'bg-raised' : 'hover:bg-raised/50')}>
                  <div className="flex items-center gap-2 text-[14px]">
                    {valid(e) ? <Outcome success={e.success} /> : <span className="text-faint">–</span>}
                    <span>Attempt {e.episode}</span>
                    {recoveriesOf(e) > 0 && (e.success
                      ? <span className="text-[12px] text-ok">recovered</span>
                      : <span className="text-[12px] text-dim" title="recovery stepped in, the attempt still failed">recovery tried</span>)}
                    <span className="ml-auto text-[12px] text-faint tabular-nums">{e.duration_s.toFixed(0)} s</span>
                  </div>
                  {!valid(e) && <p className="pl-5 text-[12px] text-faint">invalid: task not readable</p>}
                  {valid(e) && e.success == null && <LabelButtons runId={id} episode={e.episode} />}
                  {e.labeled_by === 'person' && <p className="pl-5 text-[12px] text-dim">marked by a person</p>}
                  {e.disagreement && <p className="pl-5 text-[12px] text-warn" title={`the world says ${e.disagreement.world ? 'success' : 'failure'}, ${e.disagreement.by} sees ${e.disagreement.vision ? 'success' : 'failure'}; the world's answer stands`}>physics ≠ vision</p>}
                  {valid(e) && e.success === false && <p className="mt-0.5 truncate pl-5 text-[12px] text-bad">{reasonLabel(e.reason)}{atOf(e) != null && ` at ${atOf(e)!.toFixed(1)} s`}</p>}
                </button>
              </li>
            ))}
          </ul>
        </Panel>
        <div className="min-w-0 space-y-4">
          {ep ? <EpisodeView key={ep.episode} runId={id} ep={ep} seconds={r.seconds} pages={pages}
            onPage={(plugin) => setPage({ plugin, runId: id, episode: ep.episode })} /> : <p className="text-dim">This run has no finished attempts.</p>}
          {ep && <Files key={`files-${ep.episode}`} runId={id} episode={ep.episode} />}
          {r.recap?.text && <Fold title="Run summary"><p className="px-5 py-4 text-[14px] leading-relaxed">{r.recap.text}</p></Fold>}
        </div>
        <PluginPageDrawer page={page} onClose={() => setPage(null)} />
      </div>
    </div>
  )
}

/** What the run cost and how its helpers did; each plugin's say about it, with its page one click away. */
function RunFacts({ r, onPage }: { r: RunInfo; onPage: (plugin: string) => void }) {
  const eps = r.episodes ?? []
  const lat = r.stats?.mean_latency_ms ?? null  // dreamkit.stats
  const facts: [string, ReactNode][] = [
    ['Policy', r.mock ? 'mock (scripted failures)' : r.model || 'unknown'],
    ['Cost', costText(r)],
    ['Avg model latency', lat == null ? (r.mock ? 'mock: no model' : DASH) : ms(lat)],
    ['Camera judge', judgeText(r)],
    ['Recovery saves', savesText(r)],
  ]
  // each plugin's word on the run: the verdicts it called, its badges
  const says = (r.plugins ?? []).map((name) => {
    const badges = [...new Set(eps.flatMap((e) => ((e.ui?.[name]?.badges ?? []) as { text: string }[]).map((b) => b.text)))]
    const verdicts = eps.flatMap((e) => (((e as unknown as { verdicts?: { by?: string; reason: string }[] }).verdicts) ?? []).filter((v) => v.by === name))
    const text = [verdicts.length ? `called ${count(verdicts.length, 'failure')}: ${[...new Set(verdicts.map((v) => reasonLabel(v.reason)))].join(', ')}` : '', badges.slice(0, 3).join(' · ')].filter(Boolean).join(' — ')
    return { name, text }
  })
  return (
    <div className="grid gap-4 lg:grid-cols-[minmax(0,1fr)_minmax(0,1.3fr)]">
      <Panel>
        <PanelTitle>This run</PanelTitle>
        <dl className="grid grid-cols-[150px_1fr] gap-x-4 gap-y-1.5 px-5 pb-4 text-[14px]">
          {facts.map(([k, v]) => [<dt key={k} className="text-dim">{k}</dt>, <dd key={`${k}-v`} className="min-w-0">{v}</dd>])}
        </dl>
      </Panel>
      <Panel>
        <PanelTitle>What the plugins said</PanelTitle>
        <ul className="px-3 pb-3">
          {says.map((p) => (
            <li key={p.name} className="flex items-center gap-2.5 rounded-md px-2 py-1.5 text-[13.5px] hover:bg-raised/50">
              <PluginIcon name={p.name} className="text-dim" />
              <span className="w-36 shrink-0 truncate font-medium">{p.name}</span>
              <span className="min-w-0 flex-1 truncate text-dim" title={p.text}>{p.text || 'nothing to report'}</span>
              <Button variant="quiet" className="!px-2 !py-0.5 text-[12.5px]" onClick={() => onPage(p.name)}>Open page</Button>
            </li>
          ))}
        </ul>
      </Panel>
    </div>
  )
}

const bytes = (n: number) => (n < 1024 ? `${n} B` : n < 1024 ** 2 ? `${(n / 1024).toFixed(0)} KB` : `${(n / 1024 ** 2).toFixed(1)} MB`)

/** The attempt's folder (runs/<run>/ep003/): its files, Show in Finder, and the dataset it joined. */
function Files({ runId, episode }: { runId: string; episode: number }) {
  const files = useQuery({ queryKey: ['files', runId, episode], queryFn: () => api.files(runId, episode) })
  const reveal = useMutation({ mutationFn: () => api.revealEpisode(runId, episode) })
  const f = files.data
  if (!f) return null
  const ds = f.dataset
  return (
    <>
      {ds && (
        <p className="text-[14px] text-dim">
          {ds.pending ? <>Waiting for a ✓ to join <b className="text-ink">{ds.dataset}</b></>
            : <>In dataset: <a href="#/data" className="font-medium text-ink hover:underline">{ds.dataset}</a>{ds.index != null && <> (episode {ds.index})</>}</>}
        </p>
      )}
      <Fold title={<>Files <span className="font-normal text-dim">· {count(f.files.length, 'file')} in {f.folder}/</span></>}>
        <div className="px-5 py-3">
          <ul className="text-[14px]">
            {f.files.map((x) => (
              <li key={x.path} className="flex items-baseline justify-between gap-4 border-b border-line/60 py-1.5 last:border-0">
                <a href={`/media/${runId}/${x.path}`} target="_blank" rel="noreferrer" className="truncate font-mono text-[13px] hover:underline">{x.name}</a>
                <span className="shrink-0 text-[13px] text-dim tabular-nums">{bytes(x.size_bytes)}</span>
              </li>
            ))}
            {!f.files.length && <li className="py-1.5 text-dim">No files for this attempt.</li>}
          </ul>
          <Button className="mt-3" onClick={() => reveal.mutate()}><FolderOpen className="h-4 w-4" /> Show in Finder</Button>
        </div>
      </Fold>
    </>
  )
}

/** No world, nobody judged it: a person says ✓ or ✗ (the harvester's dataset follows). */
function LabelButtons({ runId, episode }: { runId: string; episode: number }) {
  const queryClient = useQueryClient()
  const mark = useMutation({
    mutationFn: (success: boolean) => api.label(runId, episode, success),
    onSuccess: () => { queryClient.invalidateQueries({ queryKey: ['run', runId] }); queryClient.invalidateQueries({ queryKey: ['files', runId] }) },
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
  // other plugins' ctx.ui.mark (§7); failure-monitor's are its timeline above
  for (const [plugin, block] of Object.entries(ep.ui ?? {})) {
    if (plugin === 'failure-monitor') continue
    for (const m of block.marks ?? []) {
      marks.push({ t: m.t_s, label: `${plugin}: ${m.label}`, tone: m.level === 'error' ? 'bad' : m.level === 'warn' ? 'warn' : m.level === 'success' ? 'ok' : 'info' })
    }
  }
  const seek = (t: number) => {
    setMode('full')
    const v = video.current
    if (!v) return
    v.currentTime = t
    void v.play().catch(() => undefined)
  }
  const src = mode === 'recap' && ep.recap?.video ? `/media/${runId}/${ep.recap.video}` : cam ? `/media/${runId}/${videoOf[cam]}` : null

  useEffect(() => {
    // a plugin page asked to watch a moment of this attempt (on arrival, or while it is already open)
    let id: ReturnType<typeof setInterval> | undefined
    const go = () => {
      const raw = sessionStorage.getItem('dk-seek')
      if (!raw) return
      const want = JSON.parse(raw) as { run: string; episode: number; t: number }
      if (want.run !== runId || want.episode !== ep.episode) return
      sessionStorage.removeItem('dk-seek')
      // the full camera video, not the recap clip: its load handler starts it at `now`; for 3 s we make sure
      const at = Math.max(0, want.t - 1)
      setNow(at)
      setMode('full')
      const until = Date.now() + 3000
      let done = false
      clearInterval(id)
      id = setInterval(() => {
        const v = video.current
        if (done || Date.now() > until) return clearInterval(id)
        if (v && v.readyState >= 1 && !v.currentSrc.includes('recap')) {
          done = true
          v.currentTime = at
          void v.play().catch(() => undefined)
        }
      }, 100)
    }
    go()
    window.addEventListener('dk-seek', go)
    return () => { window.removeEventListener('dk-seek', go); clearInterval(id) }
  }, [runId, ep.episode])  // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => {
    // follow the video with a cheap timer instead of re-rendering on every timeupdate
    const id = setInterval(() => video.current && setNow(video.current.currentTime), 200)
    return () => clearInterval(id)
  }, [])

  return (
    <div className="min-w-0 space-y-4">
      <Panel>
        <div className="flex flex-wrap items-center gap-3 px-4 pt-3 pb-2">
          <h2 className={clsx('text-[16px] font-semibold', failed ? 'text-bad' : ep.success ? 'text-ok' : 'text-ink')}
            title={failed && (ep.details as { source?: string } | undefined)?.source
              ? `judged from ${(ep.details as { source?: string }).source === 'world' ? 'the simulator world' : 'the robot signals'}` : undefined}>
            Attempt {ep.episode}: {!valid(ep) ? 'invalid, the task names no object and target dreamkit can read'
              : failed ? `${ep.message ?? reasonLabel(ep.reason)} at ${at!.toFixed(1)} s`
              : ep.success ? `done in ${ep.duration_s.toFixed(1)} s` : 'outcome unknown, mark it on the left'}
          </h2>
          <div className="ml-auto flex items-center gap-1 rounded-md bg-base p-0.5 text-[12px]">
            {ep.recap?.video && (
              <button onClick={() => setMode(mode === 'recap' ? 'full' : 'recap')} className={clsx('rounded px-2 py-0.5', mode === 'recap' ? 'bg-raised text-ink' : 'text-dim')}>Recap</button>
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

      {((ep.timeline?.length ?? 0) > 0 || failed) && (
        <Panel>
          <PanelTitle>Moments</PanelTitle>
          {failed && (
            <button onClick={() => seek(start)} className="flex w-full gap-3 border-y border-line bg-bad-soft px-4 py-2 text-left text-[14px] hover:brightness-95">
              <span className="w-12 shrink-0 text-right text-bad tabular-nums">{at!.toFixed(1)} s</span>
              <span className="font-medium text-bad">Verdict: {ep.message ?? reasonLabel(ep.reason)}</span>
            </button>
          )}
          <ol className={clsx(!failed && 'border-t border-line', 'pb-1')}>
            {[...(ep.timeline ?? [])].sort((a, b) => a.t - b.t).map((m, i) => (
              <li key={i}>
                <button onClick={() => seek(Math.max(0, m.t - 0.5))} className="flex w-full gap-3 px-4 py-1.5 text-left text-[14px] hover:bg-raised/50">
                  <span className="w-12 shrink-0 text-right text-faint tabular-nums">{m.t.toFixed(1)} s</span>
                  <span className={BAD_KINDS.has(m.kind) ? 'text-warn' : 'text-ink'}>{m.message}</span>
                </button>
              </li>
            ))}
          </ol>
        </Panel>
      )}

      {pageRows.length > 0 && (
        <p className="flex flex-wrap items-center gap-x-3 gap-y-1 text-[14px] text-dim">
          Plugin pages:
          {pageRows.map((row) => (
            <button key={row.plugin} onClick={() => onPage(row.plugin)} className="inline-flex items-center gap-1.5 text-ink hover:underline" title={row.text}>
              <PluginIcon name={row.plugin} className="text-dim" />{row.plugin}
            </button>
          ))}
        </p>
      )}

      {trace.data && trace.data.rows.length > 1 && (
        <Fold title="What failure-monitor measured">
          <div className="h-44 px-2 pt-3 pb-3">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={trace.data.rows} onClick={(s) => { const t = Number(s?.activeLabel); if (Number.isFinite(t)) seek(t) }}>
                <CartesianGrid stroke="var(--dk-line)" vertical={false} />
                <XAxis dataKey="t" type="number" domain={[0, 'dataMax']} tick={{ fill: 'var(--dk-dim)', fontSize: 11 }} unit=" s" axisLine={false} tickLine={false} />
                <YAxis domain={[0, 'auto']} tickFormatter={(v) => `${Math.round(v * 100)}`} unit=" cm" tick={{ fill: 'var(--dk-dim)', fontSize: 11 }} width={48} axisLine={false} tickLine={false} />
                <Tooltip {...TIP} labelFormatter={(t) => `${Number(t).toFixed(1)} s`} formatter={(v) => `${(100 * Number(v)).toFixed(1)} cm`} />
                {failed && <ReferenceLine x={at} stroke="var(--dk-bad)" strokeWidth={2} />}
                <ReferenceLine x={Number(now.toFixed(1))} stroke="var(--dk-ink)" strokeDasharray="2 3" />
                <Line dataKey="d" name="hand to object" stroke="var(--dk-ink)" dot={false} strokeWidth={1.75} isAnimationActive={false} />
                <Line dataKey="z" name="object lifted" stroke="var(--dk-dim)" strokeDasharray="5 3" dot={false} strokeWidth={1.75} isAnimationActive={false} />
                <Line dataKey="dt" name="object to target" stroke="var(--dk-faint)" strokeDasharray="1 3" dot={false} strokeWidth={1.75} isAnimationActive={false} />
              </LineChart>
            </ResponsiveContainer>
          </div>
          <p className="px-5 pb-4 text-[13px] text-dim">
            Solid: hand to object · dashed: how high the object is · dotted: object to target. The red line is the verdict; click the chart to jump the video there.
          </p>
        </Fold>
      )}
    </div>
  )
}
