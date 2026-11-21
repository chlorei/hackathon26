import { useEffect, useState } from 'react'
import clsx from 'clsx'
import { History, Play, Plug } from 'lucide-react'
import RunPage from './pages/RunPage'
import PluginsPage from './pages/PluginsPage'
import HistoryPage from './pages/HistoryPage'

type Route = { page: 'run' | 'plugins' | 'history'; runId: string | null; episode: number | null }

// Hash routes: #/run, #/plugins, #/history, #/history/<run>/<episode>
function parse(hash: string): Route {
  const [page, runId, episode] = hash.replace(/^#\/?/, '').split('/')
  if (page === 'plugins') return { page: 'plugins', runId: null, episode: null }
  if (page === 'history') return { page: 'history', runId: runId || null, episode: episode ? Number(episode) : null }
  return { page: 'run', runId: null, episode: null }
}

export default function App() {
  const [route, setRoute] = useState<Route>(() => parse(window.location.hash))
  useEffect(() => {
    const onHash = () => setRoute(parse(window.location.hash))
    window.addEventListener('hashchange', onHash)
    return () => window.removeEventListener('hashchange', onHash)
  }, [])
  const go = (hash: string) => {
    window.location.hash = hash
  }
  const openRun = (id: string, episode?: number) => go(`/history/${id}${episode ? `/${episode}` : ''}`)

  const tabs = [
    { page: 'run', label: 'Run', icon: Play, hash: '/run' },
    { page: 'plugins', label: 'Plugins', icon: Plug, hash: '/plugins' },
    { page: 'history', label: 'History', icon: History, hash: '/history' },
  ] as const

  return (
    <div className="flex min-h-full flex-col">
      <header className="sticky top-0 z-10 border-b border-zinc-800 bg-zinc-950/80 backdrop-blur">
        <div className="mx-auto flex max-w-[1600px] items-center gap-6 px-5 py-3">
          <div className="flex items-center gap-2">
            <div className="flex h-7 w-7 items-center justify-center rounded-lg bg-gradient-to-br from-violet-500 to-sky-500 text-sm font-bold text-white">dk</div>
            <span className="font-semibold tracking-tight">dreamkit</span>
            <span className="text-xs text-zinc-500">plugins for Dream-1</span>
          </div>
          <nav className="flex gap-1">
            {tabs.map((t) => (
              <button key={t.page} onClick={() => go(t.hash)}
                className={clsx('flex items-center gap-1.5 rounded-lg px-3 py-1.5 text-sm transition',
                  route.page === t.page ? 'bg-zinc-800 text-white' : 'text-zinc-400 hover:text-zinc-100')}>
                <t.icon className="h-4 w-4" /> {t.label}
              </button>
            ))}
          </nav>
        </div>
      </header>
      <main className="mx-auto w-full max-w-[1600px] flex-1 px-5 py-4">
        {route.page === 'run' && <RunPage onOpenRun={openRun} />}
        {route.page === 'plugins' && <PluginsPage />}
        {route.page === 'history' && (
          <HistoryPage runId={route.runId} episode={route.episode} onOpen={(id, ep) => openRun(id, ep)} onBack={() => go('/history')} />
        )}
      </main>
    </div>
  )
}
