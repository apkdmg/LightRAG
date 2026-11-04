import { LogCaptureState, opsApi } from './api'
import { useOpsShell } from './context'
import { OpsPoll, useOpsPoll } from './poll'

/** The capture state is rechecked every 30 s; the countdown itself runs locally between polls. */
export const CAPTURE_POLL_SECONDS = 30

export type LogCaptureInfo = {
  poll: OpsPoll<LogCaptureState>
  /** The server answers GET /logs/capture (a server one release behind does not). */
  supported: boolean
  /** This viewer may start or stop a capture (operator, D8, and the log sink is on). */
  canCapture: boolean
  /** This server's log sink is off (OPS_LOG_SINK), so a capture would store nothing. */
  sinkOff: boolean
}

/**
 * The INFO capture in force (FR-26, KV `log_capture`, read by the log sink on
 * every worker). Polled only while the Logs tab is shown and stored logs can
 * be read.
 */
export function useLogCapture(): LogCaptureInfo {
  const { status } = useOpsShell()
  const enabled = !!status?.available && status?.features?.logs !== false
  const poll = useOpsPoll((signal) => opsApi.logCapture(signal), [], CAPTURE_POLL_SECONDS, { enabled })
  const supported = enabled && !(poll.error && !poll.data && (poll.error.kind === 'not_found' || poll.error.status === 405))
  // A server whose log sink is off would capture nothing: the start is not offered there.
  const sinkOn = poll.data?.sink_enabled !== false && status?.features?.log_capture !== false
  return { poll, supported, canCapture: supported && sinkOn && status?.operator === true, sinkOff: !sinkOn }
}
