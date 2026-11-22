import { useEffect, useState } from 'react'
import clsx from 'clsx'
import { useQuery } from '@tanstack/react-query'
import { History, Plug, Radio } from 'lucide-react'
import LivePage from './pages/LivePage'
import PluginsPage from './pages/PluginsPage'
import CreatePage from './pages/CreatePage'
import HistoryPage from './pages/HistoryPage'
import AppPage from './pages/AppPage'
import ErrorBoundary from './ErrorBoundary'
import { api, type AppInfo } from './api'
import { PAGES } from './labels'
import { Toaster } from './toast'
import { PluginIcon, Tag } from './ui'

// Hash routes, one address per page:
//   #/live · #/history · #/history/<run>/<attempt> · #/plugins · #/plugins/<name> · #/plugins/create · #/<app>
// An app plugin's address is its one-word title (DreamGym: #/data). Older addresses redirect:
//   #/app/<name>, #/<app name>, #/gym -> #/data;  #/create -> #/plugins/create;  '' -> #/live.

type Route =
  | { page: 'live' }
  | { page: 'history'; run: string | null; episode: number | null }
  | { page: 'plugins'; focus: string | null }
  | { page: 'create' }
  | { page: 'app'; app: AppInfo }
  | { page: 'loading' }
  | { page: 'missing'; path: string }

/** The address of an app plugin's tab: its title when that is one plain word ("Data"), else its name. */
const appSlug = (a: AppInfo) => (/^[a-z]+$/i.test(a.title) ? a.title.toLowerCase() : a.name)

function resolve(hash: string, apps: AppInfo[] | undefined): Route | { redirect: string } {
  const parts = hash.replace(/^#\/?/, '').split('/').filter(Boolean).map(decodeURIComponent)
  const [page, a, b] = parts
  if (!page) return { redirect: '/live' }
  if (page === 'live') return { page: 'live' }
  if (page === 'history') return { page: 'history', run: a ?? null, episode: b ? Number(b) : null }
  if (page === 'plugins') return a === 'create' ? { page: 'create' } : { page: 'plugins', focus: a ?? null }
  if (page === 'create') return { redirect: '/plugins/create' }
  if (!apps) return { page: 'loading' }
  const byName = (n: string) => apps.find((x) => x.name === n || appSlug(x) === n || x.commands.includes(n))
  if (page === 'app' && a) {
    const app = byName(a)
    return app ? { redirect: `/${appSlug(app)}` } : { page: 'missing', path: hash }
  }
  const app = byName(page === 'gym' ? 'dreamgym' : page)
  if (app) return appSlug(app) === page ? { page: 'app', app } : { redirect: `/${appSlug(app)}` }
  return { page: 'missing', path: hash }
}

export default function App() {
  const [hash, setHash] = useState(() => window.location.hash)
  useEffect(() => {
    const onHash = () => setHash(window.location.hash)
    window.addEventListener('hashchange', onHash)
    return () => window.removeEventListener('hashchange', onHash)
  }, [])
  const apps = useQuery({ queryKey: ['apps'], queryFn: api.apps, staleTime: 60_000 })
  const appList = apps.data?.apps.filter((x) => x.page) ?? (apps.isError ? [] : undefined)
  const resolved = resolve(hash, appList)
  const redirect = 'redirect' in resolved ? resolved.redirect : null
  useEffect(() => {
    if (redirect) window.location.replace(`#${redirect}`)
  }, [redirect])
  const route: Route = 'redirect' in resolved ? { page: 'loading' } : resolved

  const go = (to: string) => {
    window.location.hash = to
  }
  const openRun = (id: string, episode?: number) => go(`/history/${id}${episode ? `/${episode}` : ''}`)
  const current = route.page === 'create' ? 'plugins' : route.page === 'app' ? appSlug(route.app) : route.page

  const tabs = [
    { key: 'live', label: PAGES.live, icon: <Radio className="h-4 w-4" strokeWidth={1.75} />, to: '/live', warn: null },
    { key: 'history', label: PAGES.history, icon: <History className="h-4 w-4" strokeWidth={1.75} />, to: '/history', warn: null },
    { key: 'plugins', label: PAGES.plugins, icon: <Plug className="h-4 w-4" strokeWidth={1.75} />, to: '/plugins', warn: null },
    ...(appList ?? []).map((x) => ({
      key: appSlug(x), label: x.title, icon: <PluginIcon name={x.name} icon={x.icon} className="!h-4 !w-4" />,
      to: `/${appSlug(x)}`, warn: x.inactive,
    })),
  ]

  return (
    <div className="flex min-h-full flex-col">
      <header className="sticky top-0 z-20 border-b border-line bg-panel">
        <div className="mx-auto flex h-[56px] max-w-[1680px] items-center gap-6 px-7">
          <a href="#/live" className="text-[19px] font-bold tracking-[-0.02em]">dreamkit</a>
          <nav className="flex gap-1" aria-label="Main">
            {tabs.map((t) => (
              <a key={t.key} href={`#${t.to}`} aria-current={current === t.key ? 'page' : undefined} title={t.warn ?? undefined}
                className={clsx('flex items-center gap-2 rounded-lg px-3.5 py-2 text-[15px] transition-colors',
                  current === t.key ? 'bg-raised font-medium text-ink' : 'text-dim hover:text-ink')}>
                {t.icon} {t.label}
                {t.warn && <Tag tone="warn">sim only</Tag>}
              </a>
            ))}
          </nav>
        </div>
      </header>
      <main className={clsx('mx-auto w-full max-w-[1680px] flex-1', route.page !== 'app' && 'px-7 py-6')}>
        <ErrorBoundary page={hash}>
          {route.page === 'live' && <LivePage onOpenRun={openRun} />}
          {route.page === 'history' && (
            <HistoryPage runId={route.run} episode={route.episode} onOpen={(id, ep) => openRun(id, ep)} onBack={() => go('/history')} />
          )}
          {route.page === 'plugins' && <PluginsPage focus={route.focus} />}
          {route.page === 'create' && <CreatePage onOpenPlugins={() => go('/plugins')} />}
          {route.page === 'app' && <AppPage app={route.app} />}
          {route.page === 'missing' && <NotFound path={route.path} />}
        </ErrorBoundary>
      </main>
      <Toaster />
    </div>
  )
}

function NotFound({ path }: { path: string }) {
  return (
    <div className="mx-auto max-w-[640px] pt-16">
      <h1 className="text-[22px] font-semibold">No page at this address</h1>
      <p className="mt-2 text-dim">
        <code className="text-ink">{path.replace(/^#/, '') || '/'}</code> is not a page of the panel.
      </p>
      <a href="#/live" className="mt-4 inline-block rounded-lg bg-ink px-3.5 py-1.5 text-[13px] font-medium text-white">Go to Live</a>
    </div>
  )
}
