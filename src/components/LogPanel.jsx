import { useEffect, useRef } from 'react'
import './LogPanel.css'

function LogPanel({ logs = [] }) {
  const logsEndRef = useRef(null)

  // Auto-scroll to bottom when new logs arrive
  useEffect(() => {
    logsEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [logs])

  const getLogIcon = (type) => {
    switch (type) {
      case 'upload':
        return '📤'
      case 'analysis':
        return '🔍'
      case 'success':
        return '✓'
      case 'error':
        return '✕'
      case 'warning':
        return '⚠'
      case 'denoise':
        return '🔊'
      default:
        return '•'
    }
  }

  const getLogClass = (type) => {
    return `log-entry log-entry--${type}`
  }

  const formatTime = (timestamp) => {
    if (!timestamp) return ''
    const date = new Date(timestamp)
    return date.toLocaleTimeString('en-US', {
      hour: '2-digit',
      minute: '2-digit',
      second: '2-digit',
      hour12: false,
    })
  }

  return (
    <div className="log-panel">
      <div className="log-header">
        <h3 className="log-title">Event Log</h3>
        <span className="log-count">{logs.length}</span>
      </div>

      <div className="log-list">
        {logs.length === 0 ? (
          <div className="log-empty">
            <p>Awaiting events...</p>
          </div>
        ) : (
          logs.map((log, index) => (
            <div
              key={index}
              className={getLogClass(log.type)}
            >
              <span className="log-icon" aria-hidden="true">
                {getLogIcon(log.type)}
              </span>
              <div className="log-content">
                <p className="log-message">{log.message}</p>
                {log.details && (
                  <p className="log-details">{log.details}</p>
                )}
              </div>
              <span className="log-time">{formatTime(log.timestamp)}</span>
            </div>
          ))
        )}
        <div ref={logsEndRef} />
      </div>

      {logs.length > 0 && (
        <div className="log-footer">
          <span className="log-count-text">
            {logs.length} {logs.length === 1 ? 'event' : 'events'}
          </span>
        </div>
      )}
    </div>
  )
}

export default LogPanel
