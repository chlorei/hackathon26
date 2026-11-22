import type { AppInfo } from '../api'
import { Panel, PluginIcon, Tag } from '../ui'

/** An app plugin's own page (PLUGIN_STANDARD.md §13), in a frame that fills the space under the top bar.
 *  The page loads the shared theme, so it wears the panel's colours; its API is /apps/<name>/api. */
export default function AppPage({ app }: { app: AppInfo }) {
  if (app.inactive) {
    return (
      <div className="px-7 py-6">
        <Panel className="max-w-[720px] px-6 py-5">
          <div className="flex items-center gap-3">
            <PluginIcon name={app.name} icon={app.icon} className="!h-5 !w-5" />
            <h1 className="text-[22px] font-semibold">{app.title}</h1>
            <Tag tone="warn">sim only</Tag>
          </div>
          <p className="mt-3 text-dim">{app.description}</p>
          <p className="mt-3">This page {app.inactive}.</p>
        </Panel>
      </div>
    )
  }
  return <iframe key={app.name} src={`/apps/${app.name}/`} title={app.title} className="block h-[calc(100vh-56px)] w-full border-0 bg-base" />
}
