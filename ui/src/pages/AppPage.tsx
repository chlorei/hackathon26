import type { AppInfo } from '../api'
import { Panel, PluginIcon, Tag } from '../ui'

/** An app plugin's own page (PLUGIN_STANDARD.md §13), in a frame that fills the panel's content area.
 *  The page loads /theme.css, so it wears the panel's colours; its API is /apps/<name>/api. */
export default function AppPage({ app, name }: { app: AppInfo | undefined; name: string }) {
  if (!app) {
    return <p className="text-dim">No app plugin named <code className="text-ink">{name}</code>. Install one with <code className="text-ink">dreamkit install</code>, then restart <code className="text-ink">dreamkit ui</code>.</p>
  }
  if (app.inactive) {
    return (
      <Panel className="max-w-[720px] px-6 py-5">
        <div className="flex items-center gap-3">
          <PluginIcon name={app.name} icon={app.icon} className="!h-5 !w-5" />
          <h1 className="text-[22px] font-semibold">{app.title}</h1>
          <Tag tone="warn">sim only</Tag>
        </div>
        <p className="mt-3 text-dim">{app.description}</p>
        <p className="mt-3">This app {app.inactive}.</p>
      </Panel>
    )
  }
  return (
    <iframe key={app.name} src={`/apps/${app.name}/`} title={app.title}
      className="-mx-7 -my-6 block h-[calc(100vh-56px)] w-[calc(100%+3.5rem)] border-0 bg-base" />
  )
}
