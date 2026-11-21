import { useEffect, useState } from 'react'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import clsx from 'clsx'
import { Download, GitBranch, Loader2, Package, Save } from 'lucide-react'
import { api, type PluginInfo } from '../api'
import { Badge, Button, Card, Input, PluginIcon, Switch } from '../ui'

export default function PluginsPage() {
  const [tab, setTab] = useState<'installed' | 'catalog'>('installed')
  return (
    <div className="space-y-4">
      <div className="flex gap-1 rounded-lg bg-zinc-900 p-1 text-sm w-fit">
        {(['installed', 'catalog'] as const).map((t) => (
          <button key={t} onClick={() => setTab(t)}
            className={clsx('rounded-md px-3 py-1.5 capitalize', tab === t ? 'bg-zinc-700 text-white' : 'text-zinc-400 hover:text-zinc-200')}>
            {t}
          </button>
        ))}
      </div>
      {tab === 'installed' ? <Installed /> : <Catalog />}
    </div>
  )
}

function Installed() {
  const plugins = useQuery({ queryKey: ['plugins'], queryFn: api.plugins })
  return (
    <>
      <p className="text-sm text-zinc-500">
        Order = the order hooks run in (dreamkit.yaml). Changes apply to the next run.
      </p>
      <div className="grid grid-cols-1 gap-3 md:grid-cols-2 xl:grid-cols-3">
        {plugins.data?.plugins.map((p) => <PluginCard key={p.name} plugin={p} />)}
      </div>
      {plugins.data?.problems.map((p) => <div key={p} className="text-sm text-red-400">{p}</div>)}
    </>
  )
}

function PluginCard({ plugin }: { plugin: PluginInfo }) {
  const queryClient = useQueryClient()
  const [params, setParams] = useState<Record<string, string>>({})
  const [error, setError] = useState<string | null>(null)
  useEffect(() => {
    setParams(Object.fromEntries(Object.entries(plugin.params).map(([k, v]) => [k, JSON.stringify(v)])))
  }, [plugin.params])
  const update = useMutation({
    mutationFn: (body: { enabled?: boolean; params?: Record<string, unknown> }) => api.updatePlugin(plugin.name, body),
    onSuccess: (data) => {
      setError(null)
      queryClient.setQueryData(['plugins'], (old: { problems: string[] } | undefined) => ({ problems: old?.problems ?? [], plugins: data.plugins }))
    },
    onError: (e: Error) => setError(e.message),
  })
  const dirty = Object.entries(params).some(([k, v]) => v !== JSON.stringify(plugin.params[k]))
  const save = () => {
    try {
      const parsed = Object.fromEntries(Object.entries(params).map(([k, v]) => [k, v.trim() === '' ? null : JSON.parse(v)]))
      update.mutate({ params: parsed })
    } catch {
      setError('values are JSON: numbers, "strings", [lists], true/false, null')
    }
  }
  return (
    <Card className={clsx('flex flex-col', !plugin.enabled && 'opacity-70')}>
      <div className="flex items-start gap-3 p-4">
        <div className={clsx('rounded-lg p-2', plugin.enabled ? 'bg-emerald-500/15 text-emerald-400' : 'bg-zinc-800 text-zinc-500')}>
          <PluginIcon name={plugin.name} />
        </div>
        <div className="min-w-0 flex-1">
          <div className="flex items-center gap-2">
            <span className="font-medium">{plugin.name}</span>
            {plugin.source && plugin.source !== 'builtin' && <Badge tone="blue">{plugin.source}</Badge>}
            {!plugin.installed && <Badge tone="red">not installed</Badge>}
          </div>
          <p className="mt-0.5 text-sm text-zinc-400">{plugin.description}</p>
        </div>
        <Switch checked={plugin.enabled} disabled={!plugin.installed || update.isPending} onChange={(v) => update.mutate({ enabled: v })} />
      </div>
      {Object.keys(params).length > 0 && (
        <div className="mt-auto space-y-1.5 border-t border-zinc-800 px-4 py-3">
          {Object.entries(params).map(([k, v]) => (
            <div key={k} className="flex items-center gap-2">
              <span className="w-36 shrink-0 truncate font-mono text-xs text-zinc-500" title={k}>{k}</span>
              <Input value={v} onChange={(e) => setParams({ ...params, [k]: e.target.value })} className="!py-1 font-mono text-xs" />
            </div>
          ))}
          <div className="flex items-center justify-between pt-1">
            <span className="text-xs text-red-400">{error}</span>
            <Button variant="outline" className="!py-1 text-xs" disabled={!dirty || update.isPending} onClick={save}>
              <Save className="h-3.5 w-3.5" /> Save
            </Button>
          </div>
        </div>
      )}
    </Card>
  )
}

function Catalog() {
  const queryClient = useQueryClient()
  const registry = useQuery({ queryKey: ['registry'], queryFn: api.registry })
  const [log, setLog] = useState<string | null>(null)
  const [url, setUrl] = useState('')
  const install = useMutation({
    mutationFn: (name: string) => api.install(name),
    onSuccess: (r) => {
      setLog(r.output)
      queryClient.invalidateQueries({ queryKey: ['registry'] })
      queryClient.invalidateQueries({ queryKey: ['plugins'] })
    },
    onError: (e: Error) => setLog(e.message),
  })
  return (
    <>
      <div className="flex items-center gap-2">
        <Input placeholder="https://github.com/you/dreamkit-your-plugin" value={url} onChange={(e) => setUrl(e.target.value)} />
        <Button onClick={() => url && install.mutate(url)} disabled={!url || install.isPending}>
          <GitBranch className="h-4 w-4" /> Install from GitHub
        </Button>
      </div>
      <p className="text-xs text-zinc-500">catalogue: {registry.data?.source}</p>
      <div className="grid grid-cols-1 gap-3 md:grid-cols-2 xl:grid-cols-3">
        {registry.data?.plugins.map((p) => (
          <Card key={p.name} className="flex items-start gap-3 p-4">
            <div className="rounded-lg bg-zinc-800 p-2 text-zinc-400"><Package className="h-4 w-4" /></div>
            <div className="min-w-0 flex-1">
              <div className="font-medium">{p.name}</div>
              <p className="text-sm text-zinc-400">{p.description}</p>
              {p.repo !== 'builtin' && <a href={p.repo} target="_blank" rel="noreferrer" className="text-xs text-sky-400 hover:underline">{p.repo.replace('https://github.com/', '')}</a>}
            </div>
            {p.installed ? (
              <Badge tone="green">installed</Badge>
            ) : (
              <Button variant="outline" className="!py-1 text-xs" disabled={install.isPending} onClick={() => install.mutate(p.name)}>
                {install.isPending && install.variables === p.name ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <Download className="h-3.5 w-3.5" />} Install
              </Button>
            )}
          </Card>
        ))}
      </div>
      {log && <pre className="max-h-60 overflow-auto rounded-lg bg-zinc-900 p-3 text-xs text-zinc-400">{log}</pre>}
    </>
  )
}
