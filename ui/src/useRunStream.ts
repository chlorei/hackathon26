import { useEffect, useRef, useState } from 'react'
import { type EpisodeResult, type EventRecord, type StreamRecord, wsUrl } from './api'

export interface RunStream {
  events: EventRecord[]
  frames: Record<string, string>
  episodes: EpisodeResult[]
  current: { episode: number; scene: number; attempt: number; t: number } | null
  status: 'idle' | 'connecting' | 'running' | 'finished' | 'stopped' | 'error' | 'aborted'
  plugins: string[]
}

const EMPTY: RunStream = { events: [], frames: {}, episodes: [], current: null, status: 'idle', plugins: [] }

/** Follows one run over /ws/runs/<id>: events, the latest frame per camera, finished attempts. */
export function useRunStream(runId: string | null): RunStream {
  const [state, setState] = useState<RunStream>(EMPTY)
  const frames = useRef<Record<string, string>>({})

  useEffect(() => {
    if (!runId) {
      setState(EMPTY)
      return
    }
    frames.current = {}
    setState({ ...EMPTY, status: 'connecting' })
    const socket = new WebSocket(wsUrl(`/ws/runs/${runId}`))
    let pendingFrames = false
    let raf = 0
    const flushFrames = () => {
      pendingFrames = false
      setState((s) => ({ ...s, frames: { ...frames.current } }))
    }
    socket.onmessage = (message) => {
      const record = JSON.parse(message.data) as StreamRecord
      if (record.type === 'frame') {
        frames.current[record.cam] = `data:image/jpeg;base64,${record.jpeg}`
        if (!pendingFrames) {
          pendingFrames = true
          raf = requestAnimationFrame(flushFrames)
        }
        setState((s) => (s.current && record.t > s.current.t ? { ...s, current: { ...s.current, t: record.t } } : s))
        return
      }
      setState((s) => {
        switch (record.type) {
          case 'run_start':
            return { ...s, status: 'running', plugins: record.plugins }
          case 'episode_start':
            return { ...s, status: 'running', current: { episode: record.episode, scene: record.scene, attempt: record.attempt, t: 0 } }
          case 'event':
            return { ...s, events: [...s.events, record] }
          case 'episode_end':
            return { ...s, episodes: [...s.episodes.filter((e) => e.episode !== record.episode), record.result] }
          case 'run_end':
            return { ...s, status: (record.status as RunStream['status']) || 'finished', current: null }
          default:
            return s
        }
      })
    }
    socket.onclose = () => setState((s) => (s.status === 'running' || s.status === 'connecting' ? { ...s, status: 'finished' } : s))
    return () => {
      cancelAnimationFrame(raf)
      socket.close()
    }
  }, [runId])

  return state
}
