import { useEffect, useRef, useState } from 'react'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import clsx from 'clsx'
import { Check, Download, Loader2, Send, Sparkles, X } from 'lucide-react'
import { api, createZipUrl, type ChatMessage, type CreateJob } from '../api'
import { Button, Panel, PanelTitle, PluginIcon, Tag } from '../ui'

/** Requests that reliably come back as a working plugin (checked live against the model). */
const TRY = [
  'Warn when a hand moves faster than 1.5 rad/s within 15 cm of the bowl',
  'Warn when a commanded joint and the measured joint differ by more than 0.3 rad for over half a second',
  'Count how often each gripper opens and closes in an attempt and report it at the end',
]

/** A plugin described in words: a model writes it, dreamkit checks it like `dreamkit install`, then it can be kept. */
export default function CreatePage({ onOpenPlugins }: { onOpenPlugins: () => void }) {
  const [messages, setMessages] = useState<ChatMessage[]>([])
  const [jobId, setJobId] = useState<string | null>(null)
  const [ready, setReady] = useState<CreateJob | null>(null) // the last plugin that passed every check
  const [text, setText] = useState('')
  const answered = useRef(new Set<string>())

  const job = useQuery({
    queryKey: ['create', jobId], queryFn: () => api.createJob(jobId!), enabled: jobId != null,
    refetchInterval: (q) => (q.state.data?.status === 'running' ? 1000 : false),
  })
  const start = useMutation({
    mutationFn: (chat: ChatMessage[]) => api.create(chat, ready?.files),
    onSuccess: (j) => setJobId(j.id),
    onError: (e: Error) => setMessages((m) => [...m, { role: 'assistant', text: `Could not start: ${e.message}` }]),
  })
  const running = start.isPending || job.data?.status === 'running'

  useEffect(() => {  // a finished job answers in the chat once
    const j = job.data
    if (!j || j.status === 'running' || answered.current.has(j.id)) return
    answered.current.add(j.id)
    if (j.status === 'ready') {
      setReady(j)
      setMessages((m) => [...m, { role: 'assistant', text: `Made ${j.name}: ${j.summary} Check, test and bench passed${j.tries > 1 ? ` on try ${j.tries}` : ''}. Ask for changes, or keep it.` }])
    } else {
      setMessages((m) => [...m, { role: 'assistant', text: `Could not make the plugin: ${j.error ?? 'unknown error'}.` }])
    }
  }, [job.data])

  const send = (value: string) => {
    const t = value.trim()
    if (!t || running) return
    const chat: ChatMessage[] = [...messages, { role: 'user', text: t }]
    setMessages(chat)
    setText('')
    start.mutate(chat)
  }

  return (
    <div className="space-y-4">
    <div className="flex flex-wrap items-center gap-3">
      <a href="#/plugins" className="inline-flex items-center gap-2 rounded-lg px-2.5 py-1.5 text-[13px] text-dim hover:bg-raised hover:text-ink">‹ Plugins</a>
      <h1 className="text-[22px] font-semibold">Create a plugin with AI</h1>
    </div>
    <div className="grid gap-6 xl:grid-cols-[minmax(0,5fr)_minmax(0,7fr)]">
      <Panel className="flex min-h-[560px] flex-col">
        <PanelTitle>Describe a plugin</PanelTitle>
        <div className="flex-1 space-y-3 overflow-y-auto px-5 pb-4">
          {messages.length === 0 && (
            <div className="space-y-2">
              <p className="text-[14px] text-dim">Say what it should watch. A model writes the plugin; dreamkit checks it the way <code className="text-ink">dreamkit install</code> does (standard, replay tests, 1 ms budget) before you can keep it. Generated plugins only watch and report.</p>
              {TRY.map((t) => (
                <button key={t} onClick={() => send(t)} className="block w-full rounded-lg border border-line bg-panel px-3 py-2 text-left text-[13.5px] hover:bg-raised">
                  <span className="text-faint">Try: </span>{t}
                </button>
              ))}
            </div>
          )}
          {messages.map((m, i) => (
            <div key={i} className={clsx('max-w-[90%] rounded-xl px-3.5 py-2 text-[14px] whitespace-pre-wrap',
              m.role === 'user' ? 'ml-auto bg-ink text-panel' : 'bg-raised text-ink')}>{m.text}</div>
          ))}
          {running && (
            <div className="rounded-xl border border-line px-3.5 py-2 text-[13px]">
              {(job.data?.status === 'running' ? job.data.steps : []).map((s, i, all) => (
                <p key={i} className={clsx('flex items-center gap-2', i < all.length - 1 ? 'text-faint' : 'text-ink')}>
                  {i < all.length - 1 ? <Check className="h-3.5 w-3.5 text-ok" /> : <Loader2 className="h-3.5 w-3.5 animate-spin text-signal" />}
                  {s.text}
                </p>
              ))}
              {!job.data || job.data.status !== 'running' ? <p className="flex items-center gap-2 text-dim"><Loader2 className="h-3.5 w-3.5 animate-spin" /> starting</p> : null}
            </div>
          )}
        </div>
        <form className="flex gap-2 border-t border-line p-3" onSubmit={(e) => { e.preventDefault(); send(text) }}>
          <textarea value={text} onChange={(e) => setText(e.target.value)} rows={2} disabled={running}
            onKeyDown={(e) => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); send(text) } }}
            placeholder={ready ? 'Change it, e.g. "make the threshold 0.5 m/s"' : 'e.g. warn when a hand moves too fast near the bowl'}
            aria-label="Describe the plugin"
            className="min-w-0 flex-1 resize-none rounded-lg border border-line bg-panel px-3 py-2 text-[14px] outline-none focus:border-signal" />
          <Button type="submit" variant="go" disabled={running || !text.trim()} aria-label="Send"><Send className="h-4 w-4" /></Button>
        </form>
      </Panel>
      <Preview job={ready} onOpenPlugins={onOpenPlugins} />
    </div>
    </div>
  )
}

function Preview({ job, onOpenPlugins }: { job: CreateJob | null; onOpenPlugins: () => void }) {
  const queryClient = useQueryClient()
  const [enable, setEnable] = useState(true)
  const [added, setAdded] = useState<string | null>(null)
  const add = useMutation({
    mutationFn: () => api.addToCatalog(job!.id, enable),
    onSuccess: () => { setAdded(job!.id); queryClient.invalidateQueries({ queryKey: ['plugins'] }) },
  })
  if (!job) {
    return (
      <Panel className="flex min-h-[560px] flex-col items-center justify-center gap-2 text-center">
        <Sparkles className="h-6 w-6 text-faint" />
        <p className="text-dim">The plugin appears here once it passes every check.</p>
      </Panel>
    )
  }
  const info = job.info
  const props = Object.entries(info.params?.properties ?? {})
  return (
    <div className="space-y-4">
      <Panel>
        <div className="flex items-start gap-3 px-5 pt-4">
          <span className="flex h-9 w-9 shrink-0 items-center justify-center rounded-lg bg-raised"><PluginIcon name={job.name!} icon={info.icon} className="!h-[18px] !w-[18px]" /></span>
          <div className="min-w-0 flex-1">
            <p className="flex flex-wrap items-center gap-2 text-[17px] font-semibold">{job.name} <Tag tone="steel">generated</Tag>
              {info.group && <Tag>{info.group}</Tag>}{info.needs?.includes('world') && <Tag>sim only</Tag>}</p>
            <p className="text-[14px] text-dim">{info.description || job.summary}</p>
            <p className="mt-1 text-[12.5px] text-faint">hooks: {(info.hooks ?? []).join(', ')} · {info.step_rate_hz} Hz · written by {job.model}</p>
          </div>
        </div>
        <div className="flex flex-wrap items-center gap-2 px-5 py-4">
          <Button variant="go" onClick={() => add.mutate()} disabled={add.isPending || added === job.id}>
            {added === job.id ? <><Check className="h-4 w-4" /> In the catalog</> : 'Add to catalog'}
          </Button>
          <label className="flex items-center gap-1.5 text-[13px] text-dim">
            <input type="checkbox" checked={enable} onChange={(e) => setEnable(e.target.checked)} disabled={added === job.id} /> switch it on
          </label>
          <a href={createZipUrl(job.id)} download className="ml-auto inline-flex items-center gap-2 rounded-lg border border-line bg-panel px-3.5 py-1.5 text-[14px] hover:bg-raised">
            <Download className="h-4 w-4" /> Download .zip
          </a>
          {added === job.id && <button onClick={onOpenPlugins} className="w-full text-left text-[13px] text-signal hover:underline">Open Plugins: it is listed as generated{enable ? ' and on' : ''}; it runs the next time your loop starts</button>}
          {add.isError && <p className="w-full text-[13px] text-bad">{(add.error as Error).message}</p>}
        </div>
      </Panel>

      <Panel>
        <PanelTitle>Checks</PanelTitle>
        <div className="space-y-1 px-5 pb-4">
          {(['check', 'test', 'bench'] as const).map((s) => (
            <details key={s} className="text-[13.5px]">
              <summary className="flex cursor-pointer items-center gap-2">
                {job.checks[s]?.ok ? <Check className="h-4 w-4 text-ok" /> : <X className="h-4 w-4 text-bad" />}
                <span className="font-medium">dreamkit {s}</span>
                <span className="text-faint">{{ check: 'the plugin standard', test: 'its replay tests', bench: 'ms/step on every fixture, 1 ms budget' }[s]}</span>
              </summary>
              <pre className="mt-1 max-h-64 overflow-auto rounded-lg bg-log p-3 font-mono text-[12px] whitespace-pre-wrap text-[#e8e6e1]">{job.checks[s]?.out}</pre>
            </details>
          ))}
        </div>
      </Panel>

      {props.length > 0 && (
        <Panel>
          <PanelTitle>Settings</PanelTitle>
          <table className="mx-5 mb-4 w-[calc(100%-40px)] text-[13.5px]">
            <tbody>
              {props.map(([name, p]) => (
                <tr key={name} className="border-t border-line align-top">
                  <td className="py-1.5 pr-3 font-mono text-[12.5px]">{name}</td>
                  <td className="py-1.5 pr-3 whitespace-nowrap">{String(info.defaults?.[name])}{p.unit ? ` ${p.unit}` : ''}</td>
                  <td className="py-1.5 text-dim">{p.description}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </Panel>
      )}

      <Panel>
        <PanelTitle>Code</PanelTitle>
        <div className="space-y-1 px-5 pb-4">
          {Object.entries(job.files).sort(([a], [b]) => Number(b.endsWith('plugin.py')) - Number(a.endsWith('plugin.py')) || a.localeCompare(b)).map(([path, code]) => (
            <details key={path} open={path.endsWith('plugin.py')}>
              <summary className="cursor-pointer font-mono text-[13px]">{path}</summary>
              <pre className="mt-1 max-h-[480px] overflow-auto rounded-lg bg-log p-3 font-mono text-[12px] text-[#e8e6e1]">{code}</pre>
            </details>
          ))}
        </div>
      </Panel>
    </div>
  )
}
