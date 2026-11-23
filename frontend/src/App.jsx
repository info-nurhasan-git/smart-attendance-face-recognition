import { useCallback, useEffect, useRef, useState } from 'react'

const API_BASE = import.meta.env.VITE_API_URL || ''

async function apiRequest(path, options = {}) {
  const response = await fetch(`${API_BASE}${path}`, options)
  if (!response.ok) throw new Error((await response.json().catch(() => ({}))).error || 'Request failed')
  return response.json()
}

const apiGet = (path) => apiRequest(path)
const apiPost = (path, body) => apiRequest(path, {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify(body),
})
const apiDelete = (path) => apiRequest(path, { method: 'DELETE' })

const timeOf = (value) => value ? new Date(value).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' }) : '—'
const dayOf = (value) => value ? new Date(value).toLocaleDateString([], { day: 'numeric', month: 'short' }) : '—'
const initials = (name = '') =>
  name.split(/[\s_]+/).filter(Boolean).slice(0, 2).map((word) => word[0]).join('').toUpperCase() || '?'

function StatusPill({ health }) {
  if (!health) return <span className="status-pill">Connecting…</span>
  const ok = health.status === 'ok'
  return (
    <span className={`status-pill ${ok ? 'is-live' : ''}`}>
      <span className="status-dot" />
      {ok ? `MongoDB connected · ${health.people_count} faces` : 'Database offline'}
    </span>
  )
}

function ResultCard({ result }) {
  if (!result) {
    return (
      <div className="result-card empty">
        <div className="result-icon neutral">◎</div>
        <h3>No scan yet</h3>
        <p>Capture a face or upload a photo to see the match result.</p>
      </div>
    )
  }

  const ok = ['checked_in', 'checked_out', 'already_in', 'no_open_checkin'].includes(result.status)
  const match = result.match || {}
  const best = match.best
  const summary = result.summary

  return (
    <div className={`result-card ${ok ? 'success' : 'warning'}`}>
      <div className="result-icon">{ok ? '✓' : '!'}</div>
      <span className="eyebrow">{result.mode === 'out' ? 'CHECK-OUT' : 'CHECK-IN'} · {result.status.replaceAll('_', ' ')}</span>
      <h3>{result.person ? result.person.name : 'No match'}</h3>
      <p className="result-message">{result.message}</p>

      {result.person && (
        <div className="result-meta">
          <span><strong>{best ? Math.round(best.score * 100) : 0}%</strong> face match</span>
          <span>ID <strong>{result.person.id}</strong></span>
        </div>
      )}

      {result.snapshot_url && (
        <img className="scan-preview" src={`${API_BASE}${result.snapshot_url}`} alt="Scanned face" />
      )}

      {summary && (
        <div className="summary-grid">
          <div><small>In</small><strong>{timeOf(summary.check_in)}</strong></div>
          <div><small>Out</small><strong>{timeOf(summary.check_out)}</strong></div>
          <div><small>Duration</small><strong>{summary.duration_minutes ?? 0} min</strong></div>
        </div>
      )}

      {match.candidates?.length > 0 && (
        <div className="scores">
          {match.candidates.map((candidate) => (
            <div className="score-row" key={candidate.id}>
              <span>{candidate.name}</span>
              <div className="score-track"><div style={{ width: `${Math.round(candidate.score * 100)}%` }} /></div>
              <b>{Math.round(candidate.score * 100)}%</b>
            </div>
          ))}
        </div>
      )}

      {result.model_guess && (
        <p className="model-note">SVC model guess: {result.model_guess.guess} ({Math.round(result.model_guess.confidence * 100)}%)</p>
      )}
    </div>
  )
}

function RegisterForm({ regId, setRegId, regName, setRegName, registerResult }) {
  return (
    <div className="panel">
      <div className="panel-head">
        <h2>Register a face</h2>
        <span className="tag">Step 02</span>
      </div>
      <div className="form-grid">
        <label>Student / Employee ID
          <input value={regId} onChange={(event) => setRegId(event.target.value)} placeholder="e.g. STU-101" maxLength={64} />
        </label>
        <label>Name
          <input value={regName} onChange={(event) => setRegName(event.target.value)} placeholder="e.g. Rahim Ahmed" maxLength={80} />
        </label>
      </div>
      <p className="hint">Capture the face from the camera, then add 2–3 photos with the same ID for better matching.</p>
      {registerResult ? (
        <div className="register-result">
          <strong>{registerResult.status === 'registered' ? '✓ Registered' : '✓ Photo added'}</strong>
          <span>{registerResult.person.name} · {registerResult.person.id} · {registerResult.person.photos} photo(s)</span>
        </div>
      ) : (
        <div className="register-result muted">Not registered yet in this session.</div>
      )}
    </div>
  )
}

function PeopleList({ people, onDelete }) {
  if (!people.length) return <p className="empty-list">No registered faces yet.</p>
  return (
    <ul className="attendance-list">
      {people.map((person) => (
        <li key={person.id}>
          {person.photo_urls?.length
            ? <img className="thumb" src={`${API_BASE}${person.photo_urls[0]}`} alt={person.name} />
            : <span className="avatar">{initials(person.name)}</span>}
          <div className="who">
            <strong>{person.name}</strong>
            <small>ID {person.id} · {person.photos} photo(s)</small>
          </div>
          <div className="when">
            <strong>{person.attendance_count}×</strong>
            <small>attendance</small>
          </div>
          <button type="button" className="icon-btn" title="Delete" onClick={() => onDelete(person)}>×</button>
        </li>
      ))}
    </ul>
  )
}

function EventList({ events }) {
  if (!events.length) return <p className="empty-list">No IN / OUT events recorded yet.</p>
  return (
    <ul className="attendance-list event-list">
      {events.map((event) => (
        <li key={event.id}>
          {event.snapshot_url
            ? <img className="thumb small" src={`${API_BASE}${event.snapshot_url}`} alt="scan" />
            : <span className={`type-badge ${event.type}`}>{event.type.toUpperCase()}</span>}
          <div className="who">
            <strong>{event.name}</strong>
            <small>ID {event.person_id} · {dayOf(event.timestamp)} · {Math.round(event.match_score * 100)}% match</small>
          </div>
          <div className="when">
            <strong>{timeOf(event.timestamp)}</strong>
            {event.duration_minutes != null && <small>{event.duration_minutes} min inside</small>}
          </div>
        </li>
      ))}
    </ul>
  )
}

export default function App() {
  const [health, setHealth] = useState(null)
  const [people, setPeople] = useState([])
  const [events, setEvents] = useState([])
  const [result, setResult] = useState(null)
  const [registerResult, setRegisterResult] = useState(null)
  const [error, setError] = useState('')
  const [busy, setBusy] = useState(false)
  const [cameraOn, setCameraOn] = useState(false)
  const [tab, setTab] = useState('attendance')
  const [mode, setMode] = useState('in')
  const [regId, setRegId] = useState('')
  const [regName, setRegName] = useState('')
  const videoRef = useRef(null)
  const streamRef = useRef(null)
  const fileRef = useRef(null)

  const loadHealth = useCallback(async () => {
    try { setHealth(await apiGet('/api/health')) } catch { setHealth({ status: 'error' }) }
  }, [])
  const loadPeople = useCallback(async () => {
    try { setPeople((await apiGet('/api/people')).people || []) } catch { /* keep old data */ }
  }, [])
  const loadEvents = useCallback(async () => {
    try { setEvents((await apiGet('/api/attendance?limit=30')).events || []) } catch { /* keep old data */ }
  }, [])

  useEffect(() => {
    loadHealth(); loadPeople(); loadEvents()
    const timer = setInterval(() => { loadPeople(); loadEvents(); loadHealth() }, 15000)
    return () => clearInterval(timer)
  }, [loadHealth, loadPeople, loadEvents])

  const stopCamera = useCallback(() => {
    if (streamRef.current) streamRef.current.getTracks().forEach((track) => track.stop())
    streamRef.current = null
    if (videoRef.current) videoRef.current.srcObject = null
    setCameraOn(false)
  }, [])
  useEffect(() => () => stopCamera(), [stopCamera])

  const startCamera = useCallback(async () => {
    setError('')
    if (!navigator.mediaDevices?.getUserMedia) {
      setError('Camera is not available here. Upload a photo instead.')
      return
    }
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: 'user', width: { ideal: 640 }, height: { ideal: 480 } }, audio: false,
      })
      streamRef.current = stream
      if (videoRef.current) {
        videoRef.current.srcObject = stream
        await videoRef.current.play().catch(() => {})
      }
      setCameraOn(true)
    } catch {
      setError('Camera access was blocked. Allow it in your browser, or upload a photo.')
    }
  }, [])

  const handleImage = useCallback(async (image) => {
    setBusy(true); setError('')
    try {
      if (tab === 'register') {
        if (!regId.trim() || !regName.trim()) throw new Error('Enter an ID and a name before registering.')
        const data = await apiPost('/api/people/register', { id: regId.trim(), name: regName.trim(), image })
        setRegisterResult(data)
        setResult(null)
        await loadPeople()
        await loadHealth()
      } else {
        const data = await apiPost('/api/attendance', { image, mode })
        setResult(data)
        if (data.event) { await loadPeople(); await loadEvents() }
      }
    } catch (err) {
      setError(err.message)
    } finally {
      setBusy(false)
    }
  }, [tab, regId, regName, mode, loadPeople, loadEvents, loadHealth])

  const capture = () => {
    const video = videoRef.current
    if (!video?.videoWidth) { setError('Camera is still starting. Try again in a moment.'); return }
    const size = Math.min(video.videoWidth, video.videoHeight)
    const canvas = document.createElement('canvas')
    canvas.width = size; canvas.height = size
    canvas.getContext('2d').drawImage(
      video,
      (video.videoWidth - size) / 2, (video.videoHeight - size) / 2, size, size,
      0, 0, size, size,
    )
    handleImage(canvas.toDataURL('image/jpeg', 0.92))
  }

  const onFile = (event) => {
    const file = event.target.files?.[0]
    if (!file) return
    const reader = new FileReader()
    reader.onload = () => handleImage(String(reader.result))
    reader.onerror = () => setError('Could not read that file.')
    reader.readAsDataURL(file)
    event.target.value = ''
  }

  const onDelete = async (person) => {
    if (!window.confirm(`Delete ${person.name} (${person.id}) and their events?`)) return
    try {
      await apiDelete(`/api/people/${encodeURIComponent(person.id)}`)
      await loadPeople(); await loadEvents(); await loadHealth()
    } catch (err) { setError(err.message) }
  }

  const actionLabel = tab === 'register' ? 'Capture & register' : mode === 'in' ? 'Scan face · CHECK IN' : 'Scan face · CHECK OUT'

  return (
    <div className="app">
      <header className="site-header">
        <div className="brand">
          <span className="brand-mark">FA</span>
          <span>Face<span>Attend</span></span>
        </div>
        <StatusPill health={health} />
      </header>

      <main>
        <section className="hero">
          <span className="eyebrow">MONGODB FACE ATTENDANCE</span>
          <h1>Register once. Scan in / out in seconds.</h1>
          <p>Faces and IDs live in MongoDB. A scan finds the closest registered face; a match records the time, toggles IN / OUT, and updates the attendance count.</p>
        </section>

        <div className="tabs" role="tablist">
          <button type="button" className={tab === 'attendance' ? 'active' : ''} onClick={() => { setTab('attendance'); setError('') }}>Mark attendance</button>
          <button type="button" className={tab === 'register' ? 'active' : ''} onClick={() => { setTab('register'); setError('') }}>Register face</button>
        </div>

        {error && <div className="alert" role="alert">{error}</div>}

        <section className="grid">
          <div className="panel camera-panel">
            <div className="panel-head">
              <h2>Face camera</h2>
              <span className="tag">{tab === 'register' ? 'Registration photo' : `${mode.toUpperCase()} scan`}</span>
            </div>
            <div className={`camera-view ${cameraOn ? 'is-live' : ''}`}>
              <video ref={videoRef} playsInline muted className={cameraOn ? 'visible' : ''} />
              {!cameraOn && (
                <div className="camera-placeholder">
                  <span>◉</span><p>Camera is off</p><small>Start it below or upload a photo</small>
                </div>
              )}
              {cameraOn && <div className="face-frame"><i /><i /><i /><i /></div>}
              {busy && <div className="scanning"><span /> Processing…</div>}
            </div>
            <div className="camera-actions">
              {cameraOn ? (
                <button type="button" className="btn primary" onClick={capture} disabled={busy}>{actionLabel}</button>
              ) : (
                <button type="button" className="btn primary" onClick={startCamera}>Start camera</button>
              )}
              <button type="button" className="btn ghost" onClick={() => fileRef.current?.click()} disabled={busy}>Upload photo</button>
              {cameraOn && <button type="button" className="btn text" onClick={stopCamera}>Stop</button>}
              <input ref={fileRef} type="file" accept="image/*" onChange={onFile} hidden />
            </div>
          </div>

          <div className="side-column">
            {tab === 'attendance' ? (
              <>
                <div className="panel">
                  <div className="panel-head"><h2>Scan mode</h2><span className="tag">Step 02</span></div>
                  <div className="mode-switch">
                    <button type="button" className={mode === 'in' ? 'active in' : ''} onClick={() => setMode('in')}>⇥ Check IN</button>
                    <button type="button" className={mode === 'out' ? 'active out' : ''} onClick={() => setMode('out')}>⇤ Check OUT</button>
                  </div>
                </div>
                <div className="panel">
                  <div className="panel-head"><h2>Recognition result</h2><span className="tag">Step 03</span></div>
                  <ResultCard result={result} />
                </div>
              </>
            ) : (
              <RegisterForm
                regId={regId} setRegId={setRegId}
                regName={regName} setRegName={setRegName}
                registerResult={registerResult}
              />
            )}
          </div>
        </section>

        <section className="grid bottom-grid">
          <div className="panel">
            <div className="panel-head"><h2>Registered faces</h2><span className="count">{people.length}</span></div>
            <PeopleList people={people} onDelete={onDelete} />
          </div>
          <div className="panel">
            <div className="panel-head"><h2>IN / OUT log</h2><span className="count">{health?.event_count ?? events.length}</span></div>
            <EventList events={events} />
          </div>
        </section>
      </main>

      <footer>MongoDB + React + Flask · model: model/attendance_model.pkl</footer>
    </div>
  )
}
