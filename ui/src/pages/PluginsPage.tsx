import { useEffect, useState } from 'react'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import clsx from 'clsx'
import { Download, GitBranch, Loader2, Trash2 } from 'lucide-react'
import { api, type PluginInfo } from '../api'
import { Button, Input, Panel, PanelTitle, PluginIcon, Tag, Toggle } from '../ui'

export default function PluginsPage() {
  const plugins = useQuery({ queryKey: ['plugins'], queryFn: api.plugins })
  const [open, setOpen] = useState<string | null>(null)
  const list = plugins.data?.plugins ?? []
  if (plugins.isLoading) return <p className="text-[13px] text-faint">Loading plugins…</p>
  if (plugins.isError) return <p className="text-[13px] text-bad">Cannot reach the dreamkit server: {(plugins.error as Error).message}</p>
  const on = list.filter((p) => p.enabled)
  const off = list.filter((p) => !p.enabled)
  return (
    <div className="grid gap-4 xl:grid-cols-[minmax(0,1fr)_400px]">
      <div className="space-y-4">
        <Panel>
          <PanelTitle right={<span className="text-[12px] text-faint">Runs in this order, top to bottom. Changes apply to the next run.</span>}>
            Switched on
          </PanelTitle>
          <ul className="border-t border-line">
            {on.map((p) => <PluginRow key={p.name} plugin={p} open={open === p.name} onOpen={() => setOpen(open === p.name ? null : p.name)} />)}
            {!on.length && <li className="px-4 py-6 text-[13px] text-faint">Nothing is switched on. Without plugins a run only moves the arm.</li>}
          </ul>
        </Panel>
        {off.length > 0 && (
          <Panel>
            <PanelTitle>Installed, switched off</PanelTitle>
            <ul className="border-t border-line">
              {off.map((p) => <PluginRow key={p.name} plugin={p} open={open === p.name} onOpen={() => setOpen(open === p.name ? null : p.name)} />)}
            </ul>
          </Panel>
        )}
        {plugins.data?.problems.map((p) => <p key={p} className="text-[13px] text-bad">{p}</p>)}
      </div>
      <div className="space-y-4">
        <AddFromGitHub />
        <Catalogue />
      </div>
    </div>
  )
}

function PluginRow({ plugin, open, onOpen }: { plugin: PluginInfo; open: boolean; onOpen: () => void }) {
  const queryClient = useQueryClient()
  const [params, setParams] = useState<Record<string, string>>({})
  const [error, setError] = useState<string | null>(null)
  useEffect(() => setParams(Object.fromEntries(Object.entries(plugin.params).map(([k, v]) => [k, JSON.stringify(v)]))), [plugin.params])
  const refresh = (data?: { plugins: PluginInfo[] }) => {
    if (data) queryClient.setQueryData(['plugins'], (old: { problems: string[] } | undefined) => ({ problems: old?.problems ?? [], plugins: data.plugins }))
    else queryClient.invalidateQueries({ queryKey: ['plugins'] })
  }
  const update = useMutation({
    mutationFn: (body: { enabled?: boolean; params?: Record<string, unknown> }) => api.updatePlugin(plugin.name, body),
    onSuccess: (d) => { setError(null); refresh(d) },
    onError: (e: Error) => setError(e.message),
  })
  const remove = useMutation({
    mutationFn: () => api.removePlugin(plugin.name),
    onSuccess: () => { refresh(); queryClient.invalidateQueries({ queryKey: ['registry'] }) },
    onError: (e: Error) => setError(e.message),
  })
  const dirty = Object.entries(params).some(([k, v]) => v !== JSON.stringify(plugin.params[k]))
  const save = () => {
    try {
      update.mutate({ params: Object.fromEntries(Object.entries(params).map(([k, v]) => [k, v.trim() === '' ? null : JSON.parse(v)])) })
    } catch {
      setError('Values are JSON: 0.5, "top", ["a", "b"], true, null')
    }
  }
  const external = plugin.source && plugin.source !== 'builtin'
  return (
    <li className="border-b border-line/60 last:border-0">
      <div className="flex items-center gap-3 px-4 py-2.5">
        <PluginIcon name={plugin.name} className={plugin.enabled ? 'text-signal' : 'text-faint'} />
        <button onClick={onOpen} className="min-w-0 flex-1 text-left" aria-expanded={open}>
          <span className="flex items-center gap-2 text-[14px]">
            {plugin.name}
            {external && <Tag tone="steel">{plugin.source === 'pip' ? 'from GitHub / pip' : 'local file'}</Tag>}
            {!plugin.installed && <Tag tone="bad">not installed</Tag>}
          </span>
          <span className="block truncate text-[12px] text-dim">{plugin.description}</span>
        </button>
        <button onClick={onOpen} className="text-[12px] text-faint hover:text-ink">{open ? 'Close' : 'Settings'}</button>
        <Toggle checked={plugin.enabled} disabled={!plugin.installed || update.isPending} onChange={(v) => update.mutate({ enabled: v })} />
      </div>
      {open && (
        <div className="space-y-2 bg-base/60 px-4 py-3">
          <div className="grid gap-x-6 gap-y-2 lg:grid-cols-2">
          {Object.keys(params).length === 0 && <p className="text-[13px] text-faint">This plugin has no settings.</p>}
          {Object.entries(params).map(([k, v]) => (
            <label key={k} className="flex items-center gap-3">
              <span className="w-40 shrink-0 truncate font-mono text-[12px] text-dim" title={k}>{k}</span>
              <Input value={v} onChange={(e) => setParams({ ...params, [k]: e.target.value })} className="font-mono !text-[12px]" />
            </label>
          ))}
          </div>
          <div className="flex items-center gap-2 pt-1">
            <span className="flex-1 text-[12px] text-bad">{error}</span>
            {external && (
              <Button variant="quiet" onClick={() => remove.mutate()} disabled={remove.isPending}><Trash2 className="h-3.5 w-3.5" /> Uninstall</Button>
            )}
            {Object.keys(params).length > 0 && (
              <Button variant="quiet" onClick={() => setParams(Object.fromEntries(Object.entries(plugin.defaults).map(([k, d]) => [k, JSON.stringify(d)])))}>Defaults</Button>
            )}
            <Button disabled={!dirty || update.isPending} onClick={save}>Save settings</Button>
          </div>
        </div>
      )}
    </li>
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

function AddFromGitHub() {
  const [target, setTarget] = useState('')
  const install = useInstall()
  return (
    <Panel>
      <PanelTitle>Add a plugin from GitHub</PanelTitle>
      <form className="space-y-2 px-4 pb-4" onSubmit={(e) => { e.preventDefault(); if (target.trim()) install.mutate(target.trim()) }}>
        <Input value={target} onChange={(e) => setTarget(e.target.value)} placeholder="owner/repo, a repo or folder URL, or a .py file URL" />
        <Button type="submit" variant="go" className="w-full" disabled={!target.trim() || install.isPending}>
          {install.isPending ? <Loader2 className="h-4 w-4 animate-spin" /> : <GitBranch className="h-4 w-4" />} {install.isPending ? 'Installing…' : 'Install'}
        </Button>
        <p className="text-[12px] leading-relaxed text-faint">
          Works with <span className="text-dim">you/dreamkit-my-check</span>, <span className="text-dim">github.com/you/repo/tree/main/plugins/x</span>, or a single <span className="text-dim">plugin.py</span>. A plugin is code that runs on this machine, so install what you trust.
        </p>
        {install.isSuccess && <p className="text-[13px] text-ok">Installed. Switch it on in the list.</p>}
        {install.isError && <pre className="max-h-40 overflow-auto whitespace-pre-wrap rounded bg-base p-2 text-[11px] text-bad">{(install.error as Error).message}</pre>}
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
      <PanelTitle>Catalogue</PanelTitle>
      <ul className="border-t border-line">
        {entries.map((p) => (
          <li key={p.name} className="flex items-start gap-3 border-b border-line/60 px-4 py-3 last:border-0">
            <div className="min-w-0 flex-1">
              <p className="text-[14px]">{p.name}</p>
              <p className="text-[12px] text-dim">{p.description}</p>
              <a href={p.repo} target="_blank" rel="noreferrer" className="text-[12px] text-steel hover:underline">{p.repo.replace('https://github.com/', '')}</a>
            </div>
            {p.installed ? <Tag tone="ok">installed</Tag> : (
              <Button onClick={() => install.mutate(p.name)} disabled={install.isPending}>
                {install.isPending && install.variables === p.name ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <Download className="h-3.5 w-3.5" />} Install
              </Button>
            )}
          </li>
        ))}
        {!entries.length && <li className="px-4 py-4 text-[13px] text-faint">The catalogue is empty.</li>}
      </ul>
      {install.isError && <p className={clsx('px-4 pb-3 text-[12px] text-bad')}>{(install.error as Error).message.slice(0, 300)}</p>}
      <p className="px-4 pb-3 text-[11px] text-faint">Source: {registry.data?.source}</p>
    </Panel>
  )
}
