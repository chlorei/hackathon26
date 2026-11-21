// The dreamkit server's API (dreamkit/server.py) and the shapes it returns.

export type Level = 'info' | 'warn' | 'error' | 'success'

export interface EventRecord {
  type: 'event'
  ts: number
  plugin: string
  level: Level
  message: string
  episode: number | null
  t_s?: number | null
  t?: number | null  // runs before the standard
  data: Record<string, unknown>
}

/** An event's time, for runs written before and after the plugin standard. */
export const tOf = (e: { t_s?: number | null; t?: number | null }) => e.t_s ?? e.t ?? null
/** An attempt's failure time, before and after the standard. */
export const atOf = (e: { at_s?: number; at?: number }) => e.at_s ?? e.at

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
  at_s?: number
  stopped_by?: string
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
  recoveries?: number
  recovery?: { count: number; actions: string[]; worked: boolean }
  'latency-monitor'?: { mean_ms?: number; p95_ms?: number; max_ms?: number; calls?: number; mock?: boolean }
  harvester?: { split: string; path: string; frames: number }
  ui?: Record<string, { metrics: Record<string, { value: number; unit: string }>; marks: { t_s: number; label: string; level: string }[]; badges: { text: string; level: string }[] }>
}

export const recoveriesOf = (e: EpisodeResult) => e.recovery?.count ?? e.recoveries ?? 0
export const latencyOf = (e: EpisodeResult) => e['latency-monitor'] ?? e.latency

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
  report?: string | null
  live?: boolean
  path?: string
  light?: boolean
  recap?: { text: string; attempts: string[] } | null
}

export interface PluginStats {
  ms_per_step: number
  on_step_p95_ms?: number
  errors: number
  last_error: string | null
  disabled: string | null
  suppressed_events: number
  hooks: Record<string, { calls: number; avg_ms: number; max_ms: number }>
}

export interface PluginInfo {
  name: string
  installed: boolean
  enabled: boolean
  in_config: boolean
  order: number
  description: string
  source: string | null
  group: 'core' | 'diagnosis' | 'safety' | 'output' | string
  icon: string
  capability: 'observe' | 'modify_actions' | 'control'
  manifest: { hooks?: string[]; step_rate_hz?: number; needs_frames?: boolean; requires?: string[]; conflicts?: string[] }
  schema: { properties?: Record<string, SchemaProp>; required?: string[] }
  params: Record<string, unknown>
  defaults: Record<string, unknown>
  stats: PluginStats | null
  refused: string | null
  conflicts: string[]
}

export interface SchemaProp {
  type?: string
  title?: string
  description?: string
  unit?: string
  choices?: string[]
  minimum?: number
  maximum?: number
  exclusiveMinimum?: number
  items?: { type?: string }
  anyOf?: { type?: string }[]
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
  install: (name: string) => call<{ ok: boolean; log: string }>('/api/install', { method: 'POST', body: JSON.stringify({ name }) }),
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
  harvest: (id: string) => call<{ success: number; failed: number; validation: Validation | null }>(`/api/runs/${id}/harvest`),
  validateHarvest: (id: string) => call<Validation>(`/api/runs/${id}/harvest/validate`, { method: 'POST' }),
  deleteRun: (id: string) => call<{ ok: boolean }>(`/api/runs/${id}`, { method: 'DELETE' }),
}

export interface Validation {
  ok: boolean
  error?: string | null
  episodes?: number
  frames?: number
  fps?: number
  state_dim?: number
  action_dim?: number
  video_keys?: string[]
  checked?: number
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
  missed_target: 'released off target',  // runs before the merge
  released_off_target: 'released off target',
  camera_failure: 'camera failure',
  stopped_by_plugin: 'camera failure',
  grasped_air: 'grasped air',
  collision: 'collision',
  stalled: 'stalled',
  not_reached: 'not reached',
  slow_model: 'slow model',
  timeout: 'timeout',
  stopped: 'stopped',
  model_error: 'model error',
  error: 'error',
  unknown: 'unknown',
}
