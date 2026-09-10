import { useEffect, useRef, useState } from 'react'
import './App.css'

const SERVER = 'http://localhost:8000'
const FREE_LIMIT = 20

type Message = { role: 'you' | 'flow'; text: string; src?: 'memory' | 'model' }
type Health =
  | { ready: true; val_loss: number; iter: number; vocab_size: number; facts: number }
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
  const [showTeach, setShowTeach] = useState(false)
  const [teachQ, setTeachQ] = useState('')
  const [teachA, setTeachA] = useState('')
  const [taught, setTaught] = useState('')
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
          const { t, src } = JSON.parse(data) as { t: string; src?: 'memory' }
          setMessages((m) => {
            const copy = [...m]
            const prev = copy[copy.length - 1]
            copy[copy.length - 1] = {
              role: 'flow',
              text: prev.text + t,
              src: src ?? prev.src ?? 'model',
            }
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
          {health?.ready && <span className="pill">{health.facts} facts</span>}
          <span className={`pill ${plan === 'paid' ? 'paid' : ''}`}>
            {plan === 'paid' ? 'Unlimited' : `${left} left today`}
          </span>
          <button className="chip" onClick={() => setShowTeach(true)}>
            Teach Flow
          </button>
        </div>
      </header>

      <main>
        {messages.length === 0 && (
          <div className="empty">
            <p className="big">Say hello to Flow.</p>
            <p className="small">
              Flow's model was built and trained from zero on this machine. It talks, but it
              knows no facts — it learned language, not the world. Ask it about itself, or
              just chat.
            </p>
            <div className="suggestions">
              {['hello', 'what is your name?', 'tell me a story', 'how are you?'].map((s) => (
                <button key={s} className="chip" onClick={() => setInput(s)}>
                  {s}
                </button>
              ))}
            </div>
          </div>
        )}
        {messages.map((m, i) => (
          <div key={i} className={`msg ${m.role}`}>
            <span className="who">
              {m.role === 'you' ? 'You' : 'Flow'}
              {m.src === 'memory' && <em className="tag" title="a fact Flow was taught">remembered</em>}
            </span>
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

      {showTeach && (
        <div className="modal" onClick={() => setShowTeach(false)}>
          <div className="card" onClick={(e) => e.stopPropagation()}>
            <h2>Teach Flow something</h2>
            <p>
              Flow's model is far too small to hold facts, so it keeps them in a database
              instead. Anything you teach it, it knows straight away — no retraining.
            </p>
            <input
              className="field"
              placeholder="When someone asks…  e.g. what is my sister's name?"
              value={teachQ}
              onChange={(e) => setTeachQ(e.target.value)}
            />
            <input
              className="field"
              placeholder="Flow should say…  e.g. Your sister is Ariel."
              value={teachA}
              onChange={(e) => setTeachA(e.target.value)}
            />
            {taught && <p className="ok">{taught}</p>}
            <div className="row">
              <button
                className="primary"
                disabled={!teachQ.trim() || !teachA.trim()}
                onClick={async () => {
                  try {
                    const res = await fetch(`${SERVER}/teach`, {
                      method: 'POST',
                      headers: { 'Content-Type': 'application/json' },
                      body: JSON.stringify({ question: teachQ, answer: teachA }),
                    })
                    const { facts } = (await res.json()) as { facts: number }
                    setTaught(`Learned. Flow now knows ${facts} facts.`)
                    setTeachQ('')
                    setTeachA('')
                    setHealth((h) => (h?.ready ? { ...h, facts } : h))
                  } catch {
                    setTaught("Couldn't reach the server. Is it running?")
                  }
                }}
              >
                Teach
              </button>
              <button
                onClick={() => {
                  setShowTeach(false)
                  setTaught('')
                }}
              >
                Done
              </button>
            </div>
          </div>
        </div>
      )}

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
