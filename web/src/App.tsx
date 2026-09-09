import { useEffect, useRef, useState } from 'react'
import './App.css'

const SERVER = 'http://localhost:8000'
const FREE_LIMIT = 20

type Message = { role: 'you' | 'flow'; text: string }
type Health =
  | { ready: true; val_loss: number; iter: number; vocab_size: number }
  | { ready: false; reason: string }

/** Usage is per calendar day. Storing the date alongside the count means the
 *  quota resets on its own — no timer, no server, nothing to reset by hand. */
function readUsage(): { date: string; count: number } {
  const today = new Date().toISOString().slice(0, 10)
  try {
    const saved = JSON.parse(localStorage.getItem('flow_usage') ?? '{}')
    if (saved.date === today) return saved
  } catch {
    // corrupted or blocked storage — fall through to a fresh count
  }
  return { date: today, count: 0 }
}

function readHistory(): Message[] {
  try {
    return JSON.parse(localStorage.getItem('flow_history') ?? '[]')
  } catch {
    return []
  }
}

export default function App() {
  const [messages, setMessages] = useState<Message[]>(readHistory)
  const [input, setInput] = useState('')
  const [busy, setBusy] = useState(false)
  const [health, setHealth] = useState<Health | null>(null)
  const [usage, setUsage] = useState(readUsage)
  const [plan, setPlan] = useState(() => localStorage.getItem('flow_plan') ?? 'free')
  const [showUpgrade, setShowUpgrade] = useState(false)
  const bottom = useRef<HTMLDivElement>(null)

  useEffect(() => {
    fetch(`${SERVER}/health`)
      .then((r) => r.json())
      .then(setHealth)
      .catch(() => setHealth({ ready: false, reason: 'server not running' }))
  }, [])

  useEffect(() => {
    localStorage.setItem('flow_history', JSON.stringify(messages))
    bottom.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])

  const atLimit = plan === 'free' && usage.count >= FREE_LIMIT
  const left = Math.max(FREE_LIMIT - usage.count, 0)

  async function send() {
    const prompt = input.trim()
    if (!prompt || busy) return
    if (atLimit) {
      setShowUpgrade(true)
      return
    }

    setInput('')
    setBusy(true)
    setMessages((m) => [...m, { role: 'you', text: prompt }, { role: 'flow', text: '' }])

    const next = { date: usage.date, count: usage.count + 1 }
    setUsage(next)
    localStorage.setItem('flow_usage', JSON.stringify(next))

    try {
      const res = await fetch(`${SERVER}/generate`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ prompt, max_tokens: 120 }),
      })
      if (!res.ok || !res.body) throw new Error(`server said ${res.status}`)

      // Server-Sent Events over fetch. EventSource can't do POST, so the
      // frames are parsed by hand: each one is "data: {...}\n\n".
      const reader = res.body.getReader()
      const decoder = new TextDecoder()
      let buffer = ''

      for (;;) {
        const { done, value } = await reader.read()
        if (done) break
        buffer += decoder.decode(value, { stream: true })
        const frames = buffer.split('\n\n')
        buffer = frames.pop() ?? ''          // keep the incomplete tail
        for (const frame of frames) {
          if (!frame.startsWith('data: ')) continue
          const data = frame.slice(6)
          if (data === '[DONE]') continue
          const { t } = JSON.parse(data) as { t: string }
          setMessages((m) => {
            const copy = [...m]
            copy[copy.length - 1] = { role: 'flow', text: copy[copy.length - 1].text + t }
            return copy
          })
        }
      }
    } catch (e) {
      setMessages((m) => {
        const copy = [...m]
        copy[copy.length - 1] = {
          role: 'flow',
          text: `(couldn't reach the model: ${(e as Error).message}. Is the server running?)`,
        }
        return copy
      })
    } finally {
      setBusy(false)
    }
  }

  return (
    <div className="app">
      <header>
        <div className="brand">
          <span className="logo">✦</span>
          <div>
            <h1>Flow</h1>
            <p className="tagline">Your own AI. Runs on your Mac. Never leaves it.</p>
          </div>
        </div>
        <div className="status">
          {health === null && <span className="pill">checking…</span>}
          {health?.ready === false && <span className="pill bad">{health.reason}</span>}
          {health?.ready && (
            <span className="pill good" title={`trained to iteration ${health.iter}`}>
              model ready · loss {health.val_loss}
            </span>
          )}
          <span className={`pill ${plan === 'paid' ? 'paid' : ''}`}>
            {plan === 'paid' ? 'Unlimited' : `${left} left today`}
          </span>
        </div>
      </header>

      <main>
        {messages.length === 0 && (
          <div className="empty">
            <p className="big">Say something and Flow will carry it on.</p>
            <p className="small">
              Flow's model was trained from zero on this machine. It writes like a storyteller
              rather than answering questions — it has no facts, only language.
            </p>
          </div>
        )}
        {messages.map((m, i) => (
          <div key={i} className={`msg ${m.role}`}>
            <span className="who">{m.role === 'you' ? 'You' : 'Flow'}</span>
            <p>{m.text || (busy && i === messages.length - 1 ? '…' : '')}</p>
          </div>
        ))}
        <div ref={bottom} />
      </main>

      <footer>
        <textarea
          value={input}
          placeholder={atLimit ? 'Daily free messages used up' : 'Write something…'}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
              e.preventDefault()
              send()
            }
          }}
          rows={1}
        />
        <button onClick={send} disabled={busy || !input.trim()}>
          {busy ? '…' : 'Send'}
        </button>
      </footer>

      {showUpgrade && (
        <div className="modal" onClick={() => setShowUpgrade(false)}>
          <div className="card" onClick={(e) => e.stopPropagation()}>
            <h2>You've used today's {FREE_LIMIT} free messages</h2>
            <p>
              Flow is free forever on the small model. Upgrading gives you unlimited messages
              and bigger models you train yourself.
            </p>
            <div className="row">
              <button
                className="primary"
                onClick={() => {
                  // v1 has no real payments on purpose — this flag is flipped by
                  // hand. Real money is v2.
                  localStorage.setItem('flow_plan', 'paid')
                  setPlan('paid')
                  setShowUpgrade(false)
                }}
              >
                Upgrade
              </button>
              <button onClick={() => setShowUpgrade(false)}>Not now</button>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
