// Every word the panel shows for a dreamkit concept, in one place: the same thing has the same name on every page.

/** The panel's pages, in the order of the top bar. App plugins add their own tab after these (Data). */
export const PAGES = { live: 'Live', history: 'History', plugins: 'Plugins' } as const

/** Nouns. A run is one start of a loop (dreamkit.wrap ... close); an attempt is one try at the task inside it. */
export const TERM = {
  run: 'run', runs: 'runs', attempt: 'attempt', attempts: 'attempts', plugin: 'plugin', plugins: 'plugins',
  gym: 'Gym',
} as const

/** Where a run came from. */
export const SOURCE_LABEL: Record<string, string> = { gym: 'Gym', loop: 'Loop' }

/** What drove the robot: only Dream-1 is a real result; mock and scripted policies test the machinery. */
export type PolicyKind = 'dream' | 'scripted'
export const POLICY_FILTER: Record<PolicyKind | 'all', string> = { dream: 'Dream-1', scripted: 'Scripted', all: 'All' }

/** Failure codes (dreamkit.reasons) and the old names runs may still carry, as one short phrase each. */
export const REASON_LABEL: Record<string, string> = {
  fell_off_table: 'fell off table',
  dropped: 'dropped',
  released_off_target: 'released off target',
  camera_failure: 'camera failure',
  grasped_air: 'grasped air',
  released_early: 'let go too early',
  collision: 'collision',
  stalled: 'stalled',
  still_held: 'still held at the end',
  not_reached: 'not reached',
  slow_model: 'slow model',
  timeout: 'timeout',
  stopped: 'stopped',
  model_error: 'model error',
  error: 'error',
  marked_failed: 'marked failed',
  budget_exhausted: 'credit budget used up',
  unknown: 'unknown',
  // older names of the same reasons (runs written before the shared codes)
  missed_target: 'released off target',
  stopped_by_plugin: 'camera failure',
  untouched: 'not reached',
  misplaced: 'released off target',
  not_settled: 'timeout',
  api_error: 'model error',
  ik_error: 'error',
  invalid_action: 'error',
}

/** A reason code as people read it. */
export const reasonLabel = (code: string | null | undefined) => REASON_LABEL[code ?? 'unknown'] ?? (code ?? 'unknown').replace(/_/g, ' ')

/** "1 attempt", "3 attempts". */
export const count = (n: number, one: string, many = `${one}s`) => `${n} ${n === 1 ? one : many}`
