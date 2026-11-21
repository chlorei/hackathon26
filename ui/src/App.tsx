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
      <header className="sticky top-0 z-20 border-b border-line bg-base/90 backdrop-blur">
        <div className="mx-auto flex h-[52px] max-w-[1640px] items-center gap-8 px-5">
          <a href="#/run" className="flex items-baseline gap-2">
            <span className="text-[17px] font-semibold tracking-tight">dreamkit</span>
            <span className="text-[12px] text-faint">for Dream-1</span>
          </a>
          <nav className="flex h-full gap-1" aria-label="Main">
            {tabs.map((t) => (
              <button key={t.page} onClick={() => go(t.hash)} aria-current={route.page === t.page ? 'page' : undefined}
                className={clsx('relative flex items-center gap-1.5 px-3 text-[13px] transition-colors',
                  route.page === t.page ? 'text-ink' : 'text-dim hover:text-ink')}>
                <t.icon className="h-4 w-4" strokeWidth={1.75} /> {t.label}
                {route.page === t.page && <span className="absolute inset-x-2 -bottom-px h-[2px] rounded-full bg-signal" />}
              </button>
            ))}
          </nav>
        </div>
      </header>
      <main className="mx-auto w-full max-w-[1640px] flex-1 px-5 py-4">
        {route.page === 'run' && <RunPage onOpenRun={openRun} />}
        {route.page === 'plugins' && <PluginsPage />}
        {route.page === 'history' && (
          <HistoryPage runId={route.runId} episode={route.episode} onOpen={(id, ep) => openRun(id, ep)} onBack={() => go('/history')} />
        )}
      </main>
    </div>
  )
}
