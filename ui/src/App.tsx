import { useEffect, useState } from 'react'
import clsx from 'clsx'
import { History, Plug, Radio } from 'lucide-react'
import LivePage from './pages/LivePage'
import PluginsPage from './pages/PluginsPage'
import HistoryPage from './pages/HistoryPage'

type Route = { page: 'live' | 'plugins' | 'history'; runId: string | null; episode: number | null }

// Hash routes: #/live, #/plugins, #/history, #/history/<run>/<episode>
function parse(hash: string): Route {
  const [page, runId, episode] = hash.replace(/^#\/?/, '').split('/')
  if (page === 'plugins') return { page: 'plugins', runId: null, episode: null }
  if (page === 'history') return { page: 'history', runId: runId || null, episode: episode ? Number(episode) : null }
  return { page: 'live', runId: null, episode: null }
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
    { page: 'live', label: 'Live', icon: Radio, hash: '/live' },
    { page: 'plugins', label: 'Plugins', icon: Plug, hash: '/plugins' },
    { page: 'history', label: 'History', icon: History, hash: '/history' },
  ] as const

  return (
    <div className="flex min-h-full flex-col">
      <header className="sticky top-0 z-20 border-b border-line bg-panel">
        <div className="mx-auto flex h-[56px] max-w-[1680px] items-center gap-6 px-7">
          <a href="#/live" className="flex items-baseline gap-2">
            <span className="text-[17px] font-bold tracking-[-0.015em]">dreamkit</span>
            <span className="text-[12px] text-dim">for Dream-1</span>
          </a>
          <nav className="flex gap-1" aria-label="Main">
            {tabs.map((t) => (
              <button key={t.page} onClick={() => go(t.hash)} aria-current={route.page === t.page ? 'page' : undefined}
                className={clsx('flex items-center gap-1.5 rounded-lg px-3 py-1.5 text-[13.5px] transition-colors',
                  route.page === t.page ? 'bg-raised font-medium text-ink' : 'text-dim hover:text-ink')}>
                <t.icon className="h-4 w-4" strokeWidth={1.75} /> {t.label}
              </button>
            ))}
          </nav>
        </div>
      </header>
      <main className="mx-auto w-full max-w-[1680px] flex-1 px-7 py-6">
        {route.page === 'live' && <LivePage onOpenRun={openRun} />}
        {route.page === 'plugins' && <PluginsPage />}
        {route.page === 'history' && (
          <HistoryPage runId={route.runId} episode={route.episode} onOpen={(id, ep) => openRun(id, ep)} onBack={() => go('/history')} />
        )}
      </main>
    </div>
  )
}
