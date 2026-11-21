import { useEffect, useMemo, useState } from 'react'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import clsx from 'clsx'
import { Download, GitBranch, Loader2, Trash2 } from 'lucide-react'
import { api, type PluginInfo, type SchemaProp } from '../api'
import { Button, ICON_BY_PLUGIN, Input, Panel, PanelTitle, PluginIcon, PluginPageDrawer, StatusDot, Tag, Toggle, type PluginPage } from '../ui'

/** Plugins grouped by what they are for, in the order a failure goes through them: Detect → Recover → Learn. */
const STAGES: { key: 'detect' | 'recover' | 'learn'; n: number; title: string; hint: string; lead: string }[] = [
  { key: 'detect', n: 1, title: 'Detect', hint: 'Notice what went wrong, and when', lead: 'failure-monitor' },
  { key: 'recover', n: 2, title: 'Recover', hint: 'Act on it: fix the attempt, or keep the robot safe', lead: 'recovery' },
  { key: 'learn', n: 3, title: 'Learn', hint: 'Keep the videos and data that come out of it', lead: 'harvester' },
]
const CAPABILITY: Record<string, string> = { observe: 'watches', modify_actions: 'adjusts actions', control: 'controls the robot' }

function stageOf(p: PluginInfo): 'detect' | 'recover' | 'learn' {
  if (p.name === 'failure-monitor') return 'detect'
  if (p.capability === 'control' || p.capability === 'modify_actions') return 'recover'
  if (p.group === 'output' || p.name === 'harvester') return 'learn'
  return 'detect'
}

export default function PluginsPage({ focus }: { focus?: string | null }) {
  const plugins = useQuery({ queryKey: ['plugins'], queryFn: api.plugins })
  const [open, setOpen] = useState<string | null>(focus ?? null)  // #/plugins/<name>: that plugin's settings open
  useEffect(() => {
    if (!focus) return
    setOpen(focus)
    const t = setTimeout(() => document.getElementById(`plugin-${focus}`)?.scrollIntoView({ block: 'center' }), 300)
    return () => clearTimeout(t)
  }, [focus])
  const runs = useQuery({ queryKey: ['runs'], queryFn: api.runs })
  const lastRun = runs.data?.runs[0]?.id ?? null  // newest first
  const [page, setPage] = useState<PluginPage | null>(null)
  const list = useMemo(() => (plugins.data?.plugins ?? []).slice().sort((a, b) => a.order - b.order), [plugins.data])
  useEffect(() => {
    for (const p of list) ICON_BY_PLUGIN[p.name] = p.icon
  }, [list])
  if (plugins.isLoading) return <p className="text-dim">Loading plugins…</p>
  if (plugins.isError) return <p className="text-bad">Cannot reach the dreamkit server: {(plugins.error as Error).message}</p>
  const on = list.filter((p) => p.enabled).length
  const toggle = (name: string) => setOpen(open === name ? null : name)
  return (
    <div className="grid gap-6 xl:grid-cols-[minmax(0,1fr)_380px]">
      <div className="space-y-6">
        <div className="flex flex-wrap items-end justify-between gap-2">
          <div>
            <h1 className="text-[26px] font-bold tracking-[-0.015em]">Plugins</h1>
            <p className="mt-1 text-dim">{on} of {list.length} on. Changes apply the next time your loop starts (<code className="text-ink">dreamkit.wrap()</code>).</p>
          </div>
        </div>
        {STAGES.map((stage) => {
          const items = list.filter((p) => stageOf(p) === stage.key)
            .sort((a, b) => Number(b.name === stage.lead) - Number(a.name === stage.lead))
          if (!items.length) return null
          return (
            <section key={stage.key}>
              <div className="mb-2.5 flex items-baseline gap-3">
                <span className="flex h-7 w-7 items-center justify-center rounded-full bg-ink text-[13px] font-bold text-white">{stage.n}</span>
                <h2 className="text-[19px] font-semibold">{stage.title}</h2>
                <span className="text-dim">{stage.hint}</span>
              </div>
              <Panel>
                {items.map((p) => (
                  <PluginRow key={p.name} plugin={p} lead={p.name === stage.lead} open={open === p.name} onOpen={() => toggle(p.name)}
                    lastRun={lastRun} onPage={() => setPage({ plugin: p.name, runId: lastRun ?? '', episode: null })} />
                ))}
              </Panel>
            </section>
          )
        })}
        {(plugins.data?.problems.length ?? 0) > 0 && (
          <Panel className="border-bad/40">
            <PanelTitle>Refused by the plugin standard</PanelTitle>
            <pre className="mx-5 mb-4 max-h-60 overflow-auto rounded-lg bg-log p-3 font-mono text-[12.5px] whitespace-pre-wrap text-[#e8e6e1]">
              {plugins.data?.problems.join('\n\n')}
            </pre>
          </Panel>
        )}
      </div>
      <div className="space-y-5 xl:sticky xl:top-[80px] xl:self-start">
        <AddFromGitHub />
        <Catalogue />
      </div>
      <PluginPageDrawer page={page} onClose={() => setPage(null)} />
    </div>
  )
}

function useSetPlugins() {
  const queryClient = useQueryClient()
  return (data?: { plugins: PluginInfo[] }) => {
    if (data) queryClient.setQueryData(['plugins'], (old: { problems: string[] } | undefined) => ({ problems: old?.problems ?? [], plugins: data.plugins }))
    else queryClient.invalidateQueries({ queryKey: ['plugins'] })
  }
}

/** Status dot, ms per step, errors and why it was switched off — from the most recent run. */
function StatusLine({ plugin, className }: { plugin: PluginInfo; className?: string }) {
  const s = plugin.stats
  const tone = !plugin.enabled ? 'off' : plugin.refused || s?.disabled ? 'bad' : s?.errors ? 'warn' : 'ok'
  const text = !plugin.enabled ? 'Off'
    : plugin.refused ? `Not running: ${plugin.refused}`
    : s?.disabled ? s.disabled
    : s ? (s.errors ? `${s.errors} error${s.errors > 1 ? 's' : ''} last run` : 'Healthy last run') + ` · ${s.ms_per_step.toFixed(2)} ms/step`
    : 'On · no run yet'
  return (
    <div className={clsx('flex items-start gap-2 text-[13px]', className)} title={s?.last_error ?? undefined}>
      <span className="mt-[3px]"><StatusDot tone={tone} /></span>
      <span className={clsx(tone === 'bad' ? 'text-bad' : 'text-dim')}>{text}</span>
    </div>
  )
}

function PluginRow({ plugin, open, onOpen, lead, lastRun, onPage }: {
  plugin: PluginInfo; open: boolean; onOpen: () => void; lead?: boolean
  lastRun: string | null; onPage: () => void  // the plugin's own page (§7a), for the newest run
}) {
  const set = useSetPlugins()
  const queryClient = useQueryClient()
  const [values, setValues] = useState<Record<string, unknown>>(plugin.params)
  const [error, setError] = useState<string | null>(null)
  useEffect(() => setValues(plugin.params), [plugin.params])
  const update = useMutation({
    mutationFn: (body: { enabled?: boolean; params?: Record<string, unknown> }) => api.updatePlugin(plugin.name, body),
    onSuccess: (d) => { setError(null); set(d) },
    onError: (e: Error) => setError(e.message),
  })
  const remove = useMutation({
    mutationFn: () => api.removePlugin(plugin.name),
    onSuccess: () => { set(); queryClient.invalidateQueries({ queryKey: ['registry'] }) },
    onError: (e: Error) => setError(e.message),
  })
  const props = Object.entries(plugin.schema.properties ?? {})
  const dirty = JSON.stringify(values) !== JSON.stringify(plugin.params)
  const external = plugin.source && plugin.source !== 'builtin'
  return (
    <div id={`plugin-${plugin.name}`} className="border-b border-line last:border-0">
      <div className={clsx('flex items-center gap-4 px-5', lead ? 'py-4' : 'py-3.5')}>
        <span className={clsx('flex h-9 w-9 shrink-0 items-center justify-center rounded-lg', plugin.enabled ? 'bg-raised text-ink' : 'bg-raised/60 text-faint')}>
          <PluginIcon name={plugin.name} icon={plugin.icon} className="!h-[18px] !w-[18px]" />
        </span>
        <button onClick={onOpen} className="min-w-0 flex-1 text-left" aria-expanded={open}>
          <span className={clsx('flex flex-wrap items-center gap-2 font-semibold', lead ? 'text-[16.5px]' : 'text-[15px]')}>
            {plugin.name}
            {lead && <Tag tone="plain">main</Tag>}
            {plugin.capability !== 'observe' && <Tag tone={plugin.capability === 'control' ? 'signal' : 'warn'}>{CAPABILITY[plugin.capability]}</Tag>}
            {plugin.needs?.includes('world') && <Tag tone="plain" title="Reads positions and contacts, which only the simulator has: off on a real robot">sim only</Tag>}
            {external && <Tag tone="steel">{plugin.source === 'pip' ? 'installed' : plugin.source}</Tag>}
            {!plugin.installed && <Tag tone="bad">not installed</Tag>}
          </span>
          <span className="mt-0.5 block text-[13.5px] leading-snug text-dim">{plugin.description}</span>
          {plugin.capability === 'control' && plugin.conflicts.length > 0 && (
            <span className="mt-0.5 block text-[12.5px] text-dim">Conflicts with {plugin.conflicts.join(', ')}: only one control plugin can be on.</span>
          )}
        </button>
        <StatusLine plugin={plugin} className="hidden w-60 shrink-0 lg:flex" />
        <Button variant="quiet" onClick={onPage} className="!px-2.5"
          title={lastRun ? `What ${plugin.name} did in the last run (${lastRun})` : `What ${plugin.name} does`}>Open page</Button>
        <Button variant="quiet" onClick={onOpen} className="!px-2.5">{open ? 'Close' : 'Settings'}</Button>
        <Toggle checked={plugin.enabled} disabled={!plugin.installed || update.isPending} onChange={(v) => update.mutate({ enabled: v })} />
      </div>
      {open && (
        <div className="space-y-3 bg-base/70 px-5 py-4">
          {props.length === 0 && <p className="text-[13px] text-dim">This plugin has no settings.</p>}
          <div className="grid gap-x-8 gap-y-4 lg:grid-cols-2">
            {props.map(([key, prop]) => (
              <SchemaField key={key} name={key} prop={prop} value={values[key]} onChange={(v) => setValues({ ...values, [key]: v })} />
            ))}
          </div>
          {plugin.stats && Object.keys(plugin.stats.hooks).length > 0 && (
            <p className="text-[12px] text-dim">
              Last run: {Object.entries(plugin.stats.hooks).map(([h, v]) => `${h} ${v.avg_ms.toFixed(3)} ms`).join(', ')}
              {plugin.stats.suppressed_events ? `; ${plugin.stats.suppressed_events} events held back (1/s limit)` : ''}
            </p>
          )}
          <div className="flex items-center gap-2 pt-1">
            <span className="flex-1 text-[12.5px] text-bad">{error}</span>
            {external && <Button variant="stop" onClick={() => remove.mutate()} disabled={remove.isPending}><Trash2 className="h-3.5 w-3.5" /> {plugin.source === 'generated' ? 'Delete' : 'Uninstall'}</Button>}
            {props.length > 0 && <Button variant="quiet" onClick={() => setValues(plugin.defaults)}>Defaults</Button>}
            {props.length > 0 && <Button variant="go" disabled={!dirty || update.isPending} onClick={() => update.mutate({ params: values })}>Save settings</Button>}
          </div>
        </div>
      )}
    </div>
  )
}

/** One settings field, built from the Params JSON schema: description, unit, choices as chips. */
function SchemaField({ name, prop, value, onChange }: { name: string; prop: SchemaProp; value: unknown; onChange: (v: unknown) => void }) {
  const type = prop.type ?? prop.anyOf?.find((t) => t.type !== 'null')?.type
  const label = (
    <div className="mb-1 flex items-baseline justify-between gap-2">
      <span className="font-mono text-[12px] text-ink">{name}</span>
      {prop.unit ? <span className="text-[12px] text-dim">{prop.unit}</span> : null}
    </div>
  )
  const help = prop.description ? <p className="mt-1 text-[12px] leading-snug text-dim">{prop.description}</p> : null
  if (type === 'boolean') {
    return (
      <div>
        <div className="flex items-center justify-between"><span className="font-mono text-[12px]">{name}</span><Toggle checked={Boolean(value)} onChange={onChange} /></div>
        {help}
      </div>
    )
  }
  if (prop.choices && type === 'array') {
    const chosen = new Set((value as string[]) ?? [])
    return (
      <div>
        {label}
        <div className="flex flex-wrap gap-1.5">
          {prop.choices.map((c) => (
            <button key={c} type="button" onClick={() => { const n = new Set(chosen); if (n.has(c)) n.delete(c); else n.add(c); onChange(prop.choices!.filter((x) => n.has(x))) }}
              className={clsx('rounded-[20px] border px-2.5 py-0.5 text-[12.5px]', chosen.has(c) ? 'border-ink bg-ink text-white' : 'border-line bg-panel text-dim hover:text-ink')}>
              {c.replace(/_/g, ' ')}
            </button>
          ))}
        </div>
        {help}
      </div>
    )
  }
  if (prop.choices) {
    return (
      <div>
        {label}
        <div className="inline-flex rounded-lg bg-raised p-0.5">
          {prop.choices.map((c) => (
            <button key={c} type="button" onClick={() => onChange(c)}
              className={clsx('rounded-md px-2.5 py-0.5 text-[12.5px]', value === c ? 'bg-panel font-medium text-ink shadow-sm' : 'text-dim')}>{c}</button>
          ))}
        </div>
        {help}
      </div>
    )
  }
  if (type === 'number' || type === 'integer') {
    return (
      <div>
        {label}
        <Input type="number" value={(value as number) ?? ''} min={prop.minimum ?? prop.exclusiveMinimum} max={prop.maximum}
          step={type === 'integer' ? 1 : 'any'} onChange={(e) => onChange(e.target.value === '' ? null : Number(e.target.value))} className="font-mono" />
        {help}
      </div>
    )
  }
  const text = typeof value === 'string' ? value : JSON.stringify(value)
  return (
    <div>
      {label}
      <Input defaultValue={text ?? ''} className="font-mono"
        onChange={(e) => { if (type === 'string') onChange(e.target.value); else { try { onChange(JSON.parse(e.target.value)) } catch { /* keep typing */ } } }} />
      {help}
    </div>
  )
}

function useInstall() {
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: (target: string) => api.install(target),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['registry'] })
      queryClient.invalidateQueries({ queryKey: ['plugins'] })
      queryClient.invalidateQueries({ queryKey: ['apps'] })  // an app plugin: its tab appears now
    },
  })
}

function InstallLog({ result }: { result?: { ok: boolean; log: string; app?: string | null } }) {
  if (!result) return null
  return (
    <div className="space-y-2">
      <p className={clsx('text-[13px] font-medium', result.ok ? 'text-ok' : 'text-bad')}>
        {!result.ok ? 'Not installed: it did not pass the checks below.'
          : result.app ? `Installed: it passed check and its tests. Its tab "${result.app}" is in the top bar.`
          : 'Installed: it passed check, tests and bench. Switch it on in the list.'}
      </p>
      <pre className="max-h-72 overflow-auto rounded-lg bg-log p-3 font-mono text-[12px] whitespace-pre-wrap text-[#e8e6e1]">{result.log}</pre>
    </div>
  )
}

function AddFromGitHub() {
  const [target, setTarget] = useState('')
  const install = useInstall()
  return (
    <Panel>
      <PanelTitle>Add from GitHub</PanelTitle>
      <form className="space-y-2.5 px-5 pb-5" onSubmit={(e) => { e.preventDefault(); if (target.trim()) install.mutate(target.trim()) }}>
        <Input value={target} onChange={(e) => setTarget(e.target.value)} placeholder="owner/repo, a repo or folder URL, or a .py URL" />
        <Button type="submit" variant="go" className="w-full" disabled={!target.trim() || install.isPending}>
          {install.isPending ? <Loader2 className="h-4 w-4 animate-spin" /> : <GitBranch className="h-4 w-4" />} {install.isPending ? 'Checking and installing…' : 'Install'}
        </Button>
        <p className="text-[12px] leading-relaxed text-dim">
          Before anything is installed it must pass <code className="text-ink">dreamkit check</code>, its own tests and the 1 ms/step bench on
          the fixture library. A plugin is code that runs on this machine: install what you trust.
        </p>
        <InstallLog result={install.data} />
        {install.isError && <p className="text-[12.5px] text-bad">{(install.error as Error).message}</p>}
      </form>
    </Panel>
  )
}

function Catalogue() {
  const registry = useQuery({ queryKey: ['registry'], queryFn: api.registry })
  const install = useInstall()
  const entries = (registry.data?.plugins ?? []).filter((p) => p.repo !== 'builtin')
  return (
    <Panel>
      <PanelTitle right={<span className="text-[12px] text-dim">{registry.data?.source}</span>}>Catalogue</PanelTitle>
      <div className="border-t border-line">
        {entries.map((p) => (
          <div key={p.name} className="flex items-start gap-3 border-b border-line px-5 py-3 last:border-0">
            <div className="min-w-0 flex-1">
              <p className="text-[14px] font-medium">{p.name}</p>
              <p className="text-[12.5px] text-dim">{p.description}</p>
              <a href={p.repo} target="_blank" rel="noreferrer" className="text-[12px] text-signal hover:underline">{p.repo.replace('https://github.com/', '')}</a>
            </div>
            {p.installed ? <Tag tone="ok">installed</Tag> : (
              <Button onClick={() => install.mutate(p.name)} disabled={install.isPending}>
                {install.isPending && install.variables === p.name ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <Download className="h-3.5 w-3.5" />} Install
              </Button>
            )}
          </div>
        ))}
        {!entries.length && <p className="px-5 py-4 text-[13px] text-dim">The catalogue is empty.</p>}
      </div>
      {install.data && <div className="px-5 pb-4"><InstallLog result={install.data} /></div>}
    </Panel>
  )
}
