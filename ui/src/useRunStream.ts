import { useEffect, useMemo, useState } from 'react'
import { type EpisodeResult, type EventRecord, type StreamRecord, wsUrl } from './api'

/** Frames and telemetry change many times a second: they live outside React state, and the few components that
 *  draw them subscribe directly (an <img> src swap, a canvas redraw). Only events and outcomes re-render the page. */
export class LiveStore {
  frames: Record<string, string> = {}
  points: { t: number; v: Record<string, number> }[] = []
  t = 0
  private listeners = new Set<() => void>()

  subscribe(fn: () => void) {
    this.listeners.add(fn)
    return () => {
      this.listeners.delete(fn)
    }
  }
  notify() {
    for (const fn of this.listeners) fn()
  }
  resetEpisode() {
    this.points = []
    this.t = 0
  }
}

export interface RunStream {
  events: EventRecord[]
  episodes: EpisodeResult[]
  current: { episode: number; scene: number; attempt: number } | null
  status: 'idle' | 'connecting' | 'running' | 'finished' | 'stopped' | 'error' | 'aborted'
  plugins: string[]
  task: string
  mode?: string
  live: LiveStore
}

const MAX_EVENTS = 300

export function useRunStream(runId: string | null): RunStream {
  const live = useMemo(() => new LiveStore(), [runId]) // eslint-disable-line react-hooks/exhaustive-deps
  const [state, setState] = useState<Omit<RunStream, 'live'>>({ events: [], episodes: [], current: null, status: 'idle', plugins: [], task: '' })

  useEffect(() => {
    if (!runId) {
      setState({ events: [], episodes: [], current: null, status: 'idle', plugins: [], task: '' })
      return
    }
    setState({ events: [], episodes: [], current: null, status: 'connecting', plugins: [], task: '' })
    const socket = new WebSocket(wsUrl(`/ws/live/${runId}`))
    let scheduled = 0
    const schedule = () => {
      if (!scheduled) scheduled = requestAnimationFrame(() => { scheduled = 0; live.notify() })
    }
    socket.onmessage = (message) => {
      const record = JSON.parse(message.data) as StreamRecord
      if (record.type === 'frame') {
        live.frames[record.cam] = `data:image/jpeg;base64,${record.jpeg}`
        if (typeof record.t === 'number') live.t = Math.max(live.t, record.t)
        schedule()
        return
      }
      if (record.type === 'telemetry') {
        live.points.push({ t: record.t, v: record.v })
        if (live.points.length > 1200) live.points.splice(0, live.points.length - 1200)
        live.t = Math.max(live.t, record.t)
        schedule()
        return
      }
      if (record.type === 'heartbeat') {
        for (const point of record.points ?? []) live.points.push(point)
        if (live.points.length > 1200) live.points.splice(0, live.points.length - 1200)
        if (record.in_episode && typeof record.t === 'number') live.t = Math.max(live.t, record.t)
        schedule()
        return
      }
      if (record.type === 'episode_start') live.resetEpisode()
      setState((s) => {
        switch (record.type) {
          case 'run_start':
            return { ...s, status: 'running', plugins: record.plugins }
          case 'episode_start':
            return { ...s, status: 'running', task: record.task, current: { episode: record.episode, scene: record.scene, attempt: record.attempt } }
          case 'event': {
            if (record.plugin === 'dreamkit' && record.message.startsWith('mode: ')) {
              return { ...s, mode: record.message.slice(6).split(';')[0], events: [...s.events, record] }
            }
            const events = s.events.length >= MAX_EVENTS ? [...s.events.slice(-MAX_EVENTS + 1), record] : [...s.events, record]
            return { ...s, events }
          }
          case 'episode_end':
            return { ...s, current: null, episodes: [...s.episodes.filter((e) => e.episode !== record.episode), record.result] }
          case 'run_end':
            return { ...s, status: (record.status as RunStream['status']) || 'finished', current: null }
          default:
            return s
        }
      })
    }
    socket.onclose = () => setState((s) => (s.status === 'running' || s.status === 'connecting' ? { ...s, status: 'finished' } : s))
    return () => {
      cancelAnimationFrame(scheduled)
      socket.close()
    }
  }, [runId, live])

  return { ...state, live }
}

/** Re-renders the caller whenever the live store changes (use sparingly: once per animation frame at most). */
export function useLive(live: LiveStore) {
  const [, tick] = useState(0)
  useEffect(() => live.subscribe(() => tick((n) => n + 1)), [live])
  return live
}
