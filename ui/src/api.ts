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
  labeled_by?: 'person' | 'loop'
  message?: string
  details?: Record<string, unknown>
  candidates?: { reason: string; at: number }[]
  timeline?: { t: number; kind: string; message: string }[]
  duration_s: number
  requests: number
  task_readable?: boolean  // dreamkit could read an object and a target from the task (false: not judged, invalid)
  cost_usd?: number  // what this attempt's model calls cost (dreamkit.cost)
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

export interface AppInfo {
  name: string
  title: string
  icon: string
  needs: ('robot' | 'world')[]
  description: string
  page: boolean
  api: boolean
  commands: string[]
  inactive: string | null
}

export interface RunInfo {
  id: string
  source: 'gym' | 'loop'  // filled by the server for every run
  kind: 'dream' | 'scripted'  // Dream-1 drove it, or a mock / scripted / test policy
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
  cost_usd?: number | null  // what the model's time cost (dreamkit.cost); null for a mock run
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
  needs: ('robot' | 'world')[]
  manifest: { hooks?: string[]; step_rate_hz?: number; needs_frames?: boolean; requires?: string[]; conflicts?: string[]; ui?: string | null }
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

/** A failed request: `message` is for people; `details` (status, server text) goes in a Details fold. */
export class ApiError extends Error {
  details: string
  status: number
  constructor(message: string, details: string, status: number) {
    super(message)
    this.details = details
    this.status = status
  }
}

/** The one fetch every page uses. Errors come back as ApiError with a short message and the rest as details. */
async function call<T>(path: string, init?: RequestInit): Promise<T> {
  let response: Response
  try {
    response = await fetch(path, { ...init, headers: { 'content-type': 'application/json', ...(init?.headers || {}) } })
  } catch (error) {
    throw new ApiError('The dreamkit server does not answer. Is dreamkit ui still running?', String(error), 0)
  }
  if (!response.ok) {
    let detail = response.statusText
    try {
      const body = await response.json()
      detail = typeof body.detail === 'string' ? body.detail : JSON.stringify(body.detail ?? body)
    } catch {
      /* not json */
    }
    const short = detail.split('\n')[0].replace(/\/(Users|home|private|var)\/[^\s:]+/g, '…')  // no local paths up front
    throw new ApiError(short.length > 160 ? `${short.slice(0, 157)}…` : short, `${response.status} ${path}\n${detail}`, response.status)
  }
  return response.json() as Promise<T>
}

export const api = {
  config: () => call<Config>('/api/config'),
  plugins: () => call<{ plugins: PluginInfo[]; problems: string[] }>('/api/plugins'),
  updatePlugin: (name: string, body: { enabled?: boolean; params?: Record<string, unknown> }) =>
    call<{ plugins: PluginInfo[] }>(`/api/plugins/${encodeURIComponent(name)}`, { method: 'POST', body: JSON.stringify(body) }),
  registry: () => call<{ source: string; plugins: RegistryEntry[] }>('/api/registry'),
  install: (name: string) => call<{ ok: boolean; log: string; app?: string | null }>('/api/install', { method: 'POST', body: JSON.stringify({ name }) }),
  runs: () => call<{ runs: RunInfo[] }>('/api/runs'),
  run: (id: string) => call<RunInfo>(`/api/runs/${id}`),
  trace: (id: string, episode: number) =>
    call<{ object: string; target: string; rows: TraceRow[]; milestones: { t: number; kind: string; message: string }[] }>(
      `/api/runs/${id}/trace/${episode}`,
    ),
  label: (id: string, episode: number, success: boolean) =>
    call<{ ok: boolean }>(`/api/runs/${id}/episodes/${episode}/label`, { method: 'POST', body: JSON.stringify({ success }) }),
  removePlugin: (name: string) => call<{ ok: boolean; output: string }>(`/api/plugins/${encodeURIComponent(name)}`, { method: 'DELETE' }),
  reveal: (id: string, episode?: number) =>
    call<{ path: string }>(`/api/runs/${id}/reveal${episode ? `?episode=${episode}` : ''}`, { method: 'POST' }),
  apps: () => call<{ apps: AppInfo[]; problems: string[]; world: boolean }>('/api/apps'),
  live: () => call<{ id: string | null; task?: string; model?: string; mock?: boolean }>('/api/live'),
  harvest: (id: string) => call<{ success: number; failed: number; validation: Validation | null }>(`/api/runs/${id}/harvest`),
  validateHarvest: (id: string) => call<Validation>(`/api/runs/${id}/harvest/validate`, { method: 'POST' }),
  deleteRun: (id: string) => call<{ ok: boolean }>(`/api/runs/${id}`, { method: 'DELETE' }),
  // Create: a plugin described in words (dreamkit/create.py)
  create: (messages: ChatMessage[], files?: Record<string, string> | null) =>
    call<CreateJob>('/api/create', { method: 'POST', body: JSON.stringify({ messages, files: files ?? null }) }),
  createJob: (id: string) => call<CreateJob>(`/api/create/${id}`),
  llmUsage: () => call<{ today: Record<string, { requests: number; ok?: number; '429'?: number }> }>('/api/llm/usage'),
  addToCatalog: (id: string, enable: boolean) =>
    call<{ plugins: PluginInfo[] }>(`/api/create/${id}/catalog`, { method: 'POST', body: JSON.stringify({ enable }) }),
}

export const createZipUrl = (id: string) => `/api/create/${id}/zip`

export interface ChatMessage { role: 'user' | 'assistant'; text: string }

export interface CreateJob {
  id: string
  status: 'running' | 'ready' | 'failed'
  steps: { t: number; text: string }[]
  name: string | null
  summary: string
  files: Record<string, string>
  checks: Record<string, { ok: boolean; out: string }>
  info: {
    name?: string; description?: string; hooks?: string[]; group?: string; icon?: string; needs?: string[]
    step_rate_hz?: number; params?: { properties?: Record<string, SchemaProp> }; defaults?: Record<string, unknown>
  }
  model: string | null
  tries: number
  error: string | null
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

export { REASON_LABEL } from './labels'
