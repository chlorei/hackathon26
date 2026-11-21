import { useEffect, useState } from 'react'
import clsx from 'clsx'
import { useQuery } from '@tanstack/react-query'
import { History, Plug, Radio } from 'lucide-react'
import LivePage from './pages/LivePage'
import PluginsPage from './pages/PluginsPage'
import HistoryPage from './pages/HistoryPage'
import AppPage from './pages/AppPage'
import { api } from './api'
import { PluginIcon, Tag } from './ui'

type Route = { page: 'live' | 'plugins' | 'history' | 'app'; runId: string | null; episode: number | null }

// Hash routes: #/live, #/plugins, #/history, #/history/<run>/<episode>, #/app/<app plugin>
function parse(hash: string): Route {
  const [page, runId, episode] = hash.replace(/^#\/?/, '').split('/')
  if (page === 'app' && runId) return { page: 'app', runId, episode: null }
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

  const apps = useQuery({ queryKey: ['apps'], queryFn: api.apps, staleTime: 60_000 })
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
            <span className="text-[19px] font-bold tracking-[-0.02em]">dreamkit</span>
          </a>
          <nav className="flex gap-1" aria-label="Main">
            {tabs.map((t) => (
              <button key={t.page} onClick={() => go(t.hash)} aria-current={route.page === t.page ? 'page' : undefined}
                className={clsx('flex items-center gap-2 rounded-lg px-3.5 py-2 text-[15px] transition-colors',
                  route.page === t.page ? 'bg-raised font-medium text-ink' : 'text-dim hover:text-ink')}>
                <t.icon className="h-4 w-4" strokeWidth={1.75} /> {t.label}
              </button>
            ))}
            {(apps.data?.apps ?? []).filter((a) => a.page).map((a) => {
              const on = route.page === 'app' && route.runId === a.name
              return (
                <button key={a.name} onClick={() => go(`/app/${a.name}`)} aria-current={on ? 'page' : undefined}
                  title={a.inactive ?? a.description}
                  className={clsx('flex items-center gap-2 rounded-lg px-3.5 py-2 text-[15px] transition-colors',
                    on ? 'bg-raised font-medium text-ink' : 'text-dim hover:text-ink')}>
                  <PluginIcon name={a.name} icon={a.icon} className="!h-4 !w-4" /> {a.title}
                  {a.inactive && <Tag tone="warn">sim only</Tag>}
                </button>
              )
            })}
          </nav>
        </div>
      </header>
      <main className="mx-auto w-full max-w-[1680px] flex-1 px-7 py-6">
        {route.page === 'live' && <LivePage onOpenRun={openRun} />}
        {route.page === 'plugins' && <PluginsPage />}
        {route.page === 'app' && route.runId && <AppPage app={(apps.data?.apps ?? []).find((a) => a.name === route.runId)} name={route.runId} />}
        {route.page === 'history' && (
          <HistoryPage runId={route.runId} episode={route.episode} onOpen={(id, ep) => openRun(id, ep)} onBack={() => go('/history')} />
        )}
      </main>
    </div>
  )
}
