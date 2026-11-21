import { useEffect, useMemo, useState } from 'react'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import clsx from 'clsx'
import { Download, GitBranch, Loader2, Trash2 } from 'lucide-react'
import { api, type PluginInfo, type SchemaProp } from '../api'
import { Button, ICON_BY_PLUGIN, Input, Panel, PanelTitle, PluginIcon, StatusDot, Tag, Toggle } from '../ui'

const MAIN: { name: string; step: string }[] = [
  { name: 'failure-monitor', step: 'Detect' },
  { name: 'recovery', step: 'Recover' },
  { name: 'harvester', step: 'Learn' },
]
const GROUPS: { key: string; title: string; hint: string }[] = [
  { key: 'core', title: 'Core', hint: 'The loop around the model' },
  { key: 'diagnosis', title: 'Diagnosis', hint: 'Did it work, and what went wrong' },
  { key: 'safety', title: 'Safety', hint: 'Catch trouble before it reaches the robot' },
  { key: 'output', title: 'Output', hint: 'Recordings, recaps, reports' },
]
const CAPABILITY: Record<string, string> = { observe: 'observe', modify_actions: 'modifies actions', control: 'control' }

export default function PluginsPage() {
  const plugins = useQuery({ queryKey: ['plugins'], queryFn: api.plugins })
  const [open, setOpen] = useState<string | null>(null)
  const list = useMemo(() => (plugins.data?.plugins ?? []).slice().sort((a, b) => a.order - b.order), [plugins.data])
  useEffect(() => {
    for (const p of list) ICON_BY_PLUGIN[p.name] = p.icon
  }, [list])
  if (plugins.isLoading) return <p className="text-[13px] text-dim">Loading plugins…</p>
  if (plugins.isError) return <p className="text-[13px] text-bad">Cannot reach the dreamkit server: {(plugins.error as Error).message}</p>
  const main = MAIN.map((m) => ({ ...m, plugin: list.find((p) => p.name === m.name) }))
  const basic = list.filter((p) => !MAIN.some((m) => m.name === p.name))
  const toggle = (name: string) => setOpen(open === name ? null : name)
  return (
    <div className="grid gap-4 xl:grid-cols-[minmax(0,1fr)_400px]">
      <div className="space-y-4">
        <section>
          <div className="mb-3 flex items-baseline justify-between">
            <h1 className="text-[24px] font-bold tracking-[-0.015em]">Plugins</h1>
            <span className="text-[13px] text-dim">Changes apply the next time your loop calls dreamkit.wrap()</span>
          </div>
          <div className="grid gap-3 lg:grid-cols-3">
            {main.map((m, i) => (
              <Panel key={m.name} className={clsx('flex flex-col', m.plugin?.enabled && 'border-ink/25')}>
                <div className="flex items-start gap-3 px-5 pt-4">
                  <PluginIcon name={m.name} icon={m.plugin?.icon} className="mt-1 !h-5 !w-5 text-ink" />
                  <div className="min-w-0 flex-1">
                    <p className="text-[12px] font-semibold uppercase tracking-[.06em] text-dim">{i + 1}. {m.step}</p>
                    <p className="text-[16px] font-semibold">{m.name}</p>
                  </div>
                  {m.plugin && <EnableToggle plugin={m.plugin} />}
                </div>
                <p className="px-5 pt-1.5 pb-3 text-[13px] leading-snug text-dim">{m.plugin?.description ?? 'Not installed.'}</p>
                {m.plugin && <StatusLine plugin={m.plugin} className="px-5 pb-3" />}
                {m.plugin && (
                  <button onClick={() => toggle(m.name)} className="mt-auto border-t border-line px-5 py-2.5 text-left text-[13px] text-dim hover:text-ink">
                    {open === m.name ? 'Close settings' : 'Settings'}
                  </button>
                )}
              </Panel>
            ))}
          </div>
          {main.map((m) => m.plugin && open === m.name && (
            <Panel key={m.name} className="mt-3"><PluginRow plugin={m.plugin} open onOpen={() => toggle(m.name)} /></Panel>
          ))}
        </section>

        <section className="space-y-3">
          <h2 className="pt-2 text-[13px] font-semibold uppercase tracking-[.06em] text-dim">Basic</h2>
          {GROUPS.map((g) => {
            const items = basic.filter((p) => (GROUPS.some((x) => x.key === p.group) ? p.group : 'diagnosis') === g.key)
            if (!items.length) return null
            return (
              <Panel key={g.key}>
                <PanelTitle right={<span className="text-[12px] text-dim">{g.hint}</span>}>{g.title}</PanelTitle>
                <div className="border-t border-line">
                  {items.map((p) => <PluginRow key={p.name} plugin={p} open={open === p.name} onOpen={() => toggle(p.name)} />)}
                </div>
              </Panel>
            )
          })}
        </section>
        {(plugins.data?.problems.length ?? 0) > 0 && (
          <Panel className="border-bad/40">
            <PanelTitle>Refused by the plugin standard</PanelTitle>
            <pre className="mx-5 mb-4 max-h-60 overflow-auto rounded-lg bg-log p-3 font-mono text-[12px] whitespace-pre-wrap text-[#e8e6e1]">
              {plugins.data?.problems.join('\n\n')}
            </pre>
          </Panel>
        )}
      </div>
      <div className="space-y-4">
        <AddFromGitHub />
        <Catalogue />
      </div>
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

function EnableToggle({ plugin }: { plugin: PluginInfo }) {
  const set = useSetPlugins()
  const update = useMutation({ mutationFn: (enabled: boolean) => api.updatePlugin(plugin.name, { enabled }), onSuccess: (d) => set(d) })
  return <Toggle checked={plugin.enabled} disabled={!plugin.installed || update.isPending} onChange={(v) => update.mutate(v)} />
}

/** Status dot, ms per step, errors and why it was switched off — from the most recent run. */
function StatusLine({ plugin, className }: { plugin: PluginInfo; className?: string }) {
  const s = plugin.stats
  const tone = !plugin.enabled ? 'off' : plugin.refused || s?.disabled ? 'bad' : s?.errors ? 'warn' : 'ok'
  const text = !plugin.enabled ? 'off'
    : plugin.refused ? `not loaded: ${plugin.refused}`
    : s?.disabled ? s.disabled
    : s ? `${s.ms_per_step.toFixed(3)} ms/step${s.errors ? `, ${s.errors} error${s.errors > 1 ? 's' : ''}` : ', no errors'}`
    : 'on, no run yet'
  return (
    <div className={clsx('flex items-start gap-2 text-[12px]', className)} title={s?.last_error ?? undefined}>
      <span className="mt-[3px]"><StatusDot tone={tone} /></span>
      <span className={clsx(tone === 'bad' ? 'text-bad' : 'text-dim')}>{text}</span>
    </div>
  )
}

function PluginRow({ plugin, open, onOpen }: { plugin: PluginInfo; open: boolean; onOpen: () => void }) {
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
    <div className="border-b border-line last:border-0">
      <div className="flex items-center gap-3 px-5 py-3">
        <PluginIcon name={plugin.name} icon={plugin.icon} className={plugin.enabled ? 'text-ink' : 'text-faint'} />
        <button onClick={onOpen} className="min-w-0 flex-1 text-left" aria-expanded={open}>
          <span className="flex flex-wrap items-center gap-2 text-[14px] font-medium">
            {plugin.name}
            {plugin.capability !== 'observe' && <Tag tone={plugin.capability === 'control' ? 'signal' : 'warn'}>{CAPABILITY[plugin.capability]}</Tag>}
            {plugin.needs?.includes('world') && <Tag tone="plain" title="Reads positions and contacts, which only the simulator has: off on a real robot">sim only</Tag>}
            {external && <Tag tone="steel">{plugin.source === 'pip' ? 'installed' : plugin.source}</Tag>}
            {!plugin.installed && <Tag tone="bad">not installed</Tag>}
          </span>
          <span className="block truncate text-[12.5px] text-dim">{plugin.description}</span>
          {plugin.capability === 'control' && plugin.conflicts.length > 0 && (
            <span className="block text-[12px] text-dim">conflicts with {plugin.conflicts.join(', ')}: only one control plugin can be on</span>
          )}
        </button>
        <StatusLine plugin={plugin} className="hidden w-56 shrink-0 md:flex" />
        <button onClick={onOpen} className="text-[12.5px] text-dim hover:text-ink">{open ? 'Close' : 'Settings'}</button>
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
            {external && <Button variant="stop" onClick={() => remove.mutate()} disabled={remove.isPending}><Trash2 className="h-3.5 w-3.5" /> Uninstall</Button>}
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
    },
  })
}

function InstallLog({ result }: { result?: { ok: boolean; log: string } }) {
  if (!result) return null
  return (
    <div className="space-y-2">
      <p className={clsx('text-[13px] font-medium', result.ok ? 'text-ok' : 'text-bad')}>
        {result.ok ? 'Installed: it passed check, tests and bench. Switch it on in the list.' : 'Not installed: it did not pass the checks below.'}
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
