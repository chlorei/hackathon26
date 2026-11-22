import { Component, type ReactNode } from 'react'

/** One page that breaks does not take the panel down: the message, a way back, and the details folded away. */
export default class ErrorBoundary extends Component<{ children: ReactNode; page: string }, { error: Error | null }> {
  state: { error: Error | null } = { error: null }

  static getDerivedStateFromError(error: Error) {
    return { error }
  }

  componentDidUpdate(prev: { page: string }) {
    if (prev.page !== this.props.page && this.state.error) this.setState({ error: null })
  }

  render() {
    const { error } = this.state
    if (!error) return this.props.children
    return (
      <div className="mx-auto max-w-[640px] pt-16">
        <h1 className="text-[22px] font-semibold">This page could not be shown</h1>
        <p className="mt-2 text-dim">The rest of the panel still works. Reload to try again.</p>
        <div className="mt-4 flex gap-2">
          <button onClick={() => window.location.reload()} className="rounded-lg bg-ink px-3.5 py-1.5 text-[13px] font-medium text-white">Reload</button>
          <a href="#/live" className="rounded-lg border border-line bg-panel px-3.5 py-1.5 text-[13px] font-medium">Go to Live</a>
        </div>
        <details className="mt-5 text-[13px] text-dim">
          <summary className="cursor-pointer select-none">Details</summary>
          <pre className="mt-2 overflow-auto rounded-lg bg-raised p-3 font-mono text-[12px] whitespace-pre-wrap">{error.stack ?? error.message}</pre>
        </details>
      </div>
    )
  }
}
