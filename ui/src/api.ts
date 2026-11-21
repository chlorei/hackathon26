// The dreamkit server's API (dreamkit/server.py) and the shapes it returns.

export type Level = 'info' | 'warn' | 'error' | 'success'

export interface EventRecord {
  type: 'event'
  ts: number
  plugin: string
  level: Level
  message: string
  episode: number | null
  t: number | null
  data: Record<string, unknown>
}

export interface EpisodeResult {
  episode: number
  scene: number
  attempt: number
  seed: number
  task: string
  behavior?: string | null
  success: boolean | null
  reason: string | null
  at?: number
  message?: string
  details?: Record<string, unknown>
  candidates?: { reason: string; at: number }[]
  timeline?: { t: number; kind: string; message: string }[]
  duration_s: number
  requests: number
  videos: string[]
  latency?: { mean_ms: number; p95_ms: number; max_ms: number; n: number }
  recap?: { headline: string; moments: { t: number; text: string; tone: 'bad' | 'good' | 'info'; verdict?: boolean }[]; video: string | null }
  gripper?: { t: number; side: string; kind: string }[]
}

export interface RunInfo {
  id: string
  task: string
  model: string
  mock: boolean
  status: string
  started: number
  finished?: number
  episodes_planned: number
  seconds: number
  plugins: string[]
  episodes: EpisodeResult[]
  summary: { attempts: number; successes: number; scenes: number; scene_successes: number; reasons: Record<string, number> }
  events?: StreamRecord[]
  report?: boolean
  live?: boolean
  path?: string
  light?: boolean
  recap?: { text: string; attempts: string[] } | null
}

export interface PluginInfo {
  name: string
  installed: boolean
  enabled: boolean
  in_config: boolean
  description: string
  source: string | null
  group: string
  order: number
  params: Record<string, unknown>
  defaults: Record<string, unknown>
}

export interface RegistryEntry {
  name: string
  repo: string
  description: string
  tags?: string[]
  installed: boolean
}

export interface Config {
  config_file: string
  sim_found: boolean
  light: boolean
  runs_dir: string
}

export type StreamRecord =
  | EventRecord
  | { type: 'frame'; cam: string; jpeg: string; episode: number; t: number }
  | { type: 'telemetry'; episode: number; t: number; v: Record<string, number> }
  | { type: 'heartbeat'; episode: number; scene: number; attempt: number; task: string; t: number; in_episode: boolean; status: string; updated: number; points: { t: number; v: Record<string, number> }[] }
  | { type: 'run_start'; run_id: string; mock: boolean; plugins: string[] }
  | { type: 'episode_start'; episode: number; scene: number; attempt: number; task: string; seed?: number; positions?: Record<string, number[]> }
  | { type: 'episode_end'; episode: number; scene: number; attempt: number; duration_s: number; result: EpisodeResult }
  | { type: 'run_end'; run_id: string; status: string; attempts?: number; successes?: number }

async function call<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(path, {
    ...init,
    headers: { 'content-type': 'application/json', ...(init?.headers || {}) },
  })
  if (!response.ok) {
    let detail = response.statusText
    try {
      const body = await response.json()
      detail = typeof body.detail === 'string' ? body.detail : JSON.stringify(body.detail)
    } catch {
      /* not json */
    }
    throw new Error(detail)
  }
  return response.json() as Promise<T>
}

export const api = {
  config: () => call<Config>('/api/config'),
  plugins: () => call<{ plugins: PluginInfo[]; problems: string[] }>('/api/plugins'),
  updatePlugin: (name: string, body: { enabled?: boolean; params?: Record<string, unknown> }) =>
    call<{ plugins: PluginInfo[] }>(`/api/plugins/${encodeURIComponent(name)}`, { method: 'POST', body: JSON.stringify(body) }),
  registry: () => call<{ source: string; plugins: RegistryEntry[] }>('/api/registry'),
  install: (name: string) => call<{ ok: boolean; output?: string }>('/api/install', { method: 'POST', body: JSON.stringify({ name }) }),
  runs: () => call<{ runs: RunInfo[] }>('/api/runs'),
  run: (id: string) => call<RunInfo>(`/api/runs/${id}`),
  trace: (id: string, episode: number) =>
    call<{ object: string; target: string; rows: TraceRow[]; milestones: { t: number; kind: string; message: string }[] }>(
      `/api/runs/${id}/trace/${episode}`,
    ),
  removePlugin: (name: string) => call<{ ok: boolean; output: string }>(`/api/plugins/${encodeURIComponent(name)}`, { method: 'DELETE' }),
  reveal: (id: string, episode?: number) =>
    call<{ path: string }>(`/api/runs/${id}/reveal${episode ? `?episode=${episode}` : ''}`, { method: 'POST' }),
  live: () => call<{ id: string | null; task?: string; model?: string; mock?: boolean }>('/api/live'),
  deleteRun: (id: string) => call<{ ok: boolean }>(`/api/runs/${id}`, { method: 'DELETE' }),
}

export interface TraceRow {
  t: number
  z?: number
  d?: number
  dt?: number
  cmd_left: number
  cmd_right: number
}

export function wsUrl(path: string): string {
  const proto = window.location.protocol === 'https:' ? 'wss' : 'ws'
  return `${proto}://${window.location.host}${path}`
}

export const REASON_LABEL: Record<string, string> = {
  fell_off_table: 'fell off table',
  dropped: 'dropped',
  missed_target: 'missed target',
  released_off_target: 'released off target',
  grasped_air: 'grasped air',
  collision: 'collision',
  stalled: 'stalled',
  not_reached: 'not reached',
  slow_model: 'slow model',
  timeout: 'timeout',
  stopped: 'stopped',
  stopped_by_plugin: 'stopped by plugin',
  model_error: 'model error',
  error: 'error',
  unknown: 'unknown',
}
