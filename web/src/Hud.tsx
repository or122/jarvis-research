import { useEffect, useRef, useState } from 'react'
import Reactor from './Reactor'
import './hud.css'

const SERVER = `http://${location.hostname}:8000`

type Msg = { who: 'you' | 'jarvis'; text: string; src?: string }
type Health =
  | { ready: true; val_loss: number; iter: number; facts: number }
  | { ready: false; reason: string }

const STARTERS = ['who are you', 'what time is it', 'what is 89 times 47', 'make me a game']

export default function Hud({ onExit }: { onExit: () => void }) {
  const [messages, setMessages] = useState<Msg[]>([])
  const [input, setInput] = useState('')
  const [busy, setBusy] = useState(false)
  const [health, setHealth] = useState<Health | null>(null)
  const bottom = useRef<HTMLDivElement>(null)

  useEffect(() => {
    fetch(`${SERVER}/health`)
      .then((r) => r.json())
      .then(setHealth)
      .catch(() => setHealth({ ready: false, reason: 'offline' }))
  }, [])

  useEffect(() => bottom.current?.scrollIntoView({ behavior: 'smooth' }), [messages])

  async function send(text?: string) {
    const prompt = (text ?? input).trim()
    if (!prompt || busy) return
    setInput('')
    setBusy(true)
    setMessages((m) => [...m, { who: 'you', text: prompt }, { who: 'jarvis', text: '' }])

    try {
      const res = await fetch(`${SERVER}/generate`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ prompt, max_tokens: 90 }),
      })
      if (!res.ok || !res.body) throw new Error(`server said ${res.status}`)

      const reader = res.body.getReader()
      const decoder = new TextDecoder()
      let buffer = ''

      for (;;) {
        const { done, value } = await reader.read()
        if (done) break
        buffer += decoder.decode(value, { stream: true })
        const frames = buffer.split('\n\n')
        buffer = frames.pop() ?? ''
        for (const frame of frames) {
          if (!frame.startsWith('data: ')) continue
          const data = frame.slice(6)
          if (data === '[DONE]') continue
          const { t, src } = JSON.parse(data) as { t: string; src?: string }
          setMessages((m) => {
            const copy = [...m]
            const prev = copy[copy.length - 1]
            copy[copy.length - 1] = {
              who: 'jarvis',
              text: prev.text + t,
              src: src ?? prev.src,
            }
            return copy
          })
        }
      }
    } catch (e) {
      setMessages((m) => {
        const copy = [...m]
        copy[copy.length - 1] = {
          who: 'jarvis',
          text: `Connection failed: ${(e as Error).message}`,
        }
        return copy
      })
    } finally {
      setBusy(false)
    }
  }

  // The reactor doubles as the status light, so its state has to mirror what
  // is really happening rather than being decorative.
  const last = messages[messages.length - 1]
  const state = busy ? (last?.text ? 'speaking' : 'thinking') : 'idle'

  return (
    <div className="hud">
      <div className="hud-top">
        <span className="hud-name">J A R V I S</span>
        <span className="hud-stat">
          {health?.ready ? (
            <>
              <span>
                LOSS <b>{health.val_loss}</b>
              </span>
              <span>
                FACTS <b>{health.facts}</b>
              </span>
              <span>
                ITER <b>{health.iter}</b>
              </span>
            </>
          ) : (
            <span>{health ? 'OFFLINE' : 'LINKING…'}</span>
          )}
          <button className="hud-chip" onClick={onExit}>
            Exit
          </button>
        </span>
      </div>

      <Reactor state={state} />
      <div className="hud-state">
        {state === 'thinking' ? 'processing' : state === 'speaking' ? 'responding' : 'standing by'}
      </div>

      <div className="hud-log">
        {messages.length === 0 && (
          <div className="hud-empty">
            <p>
              <b>JARVIS ONLINE</b>
            </p>
            <p>
              Running on a language model Or built and trained himself.
              <br />
              No internet. Nothing leaves this machine.
            </p>
            <div className="hud-chips">
              {STARTERS.map((s) => (
                <button key={s} className="hud-chip" onClick={() => send(s)}>
                  {s}
                </button>
              ))}
            </div>
          </div>
        )}

        {messages.map((m, i) => (
          <div key={i} className={`hud-msg from-${m.who}`}>
            <span className="hud-who">
              {m.who === 'you' ? 'You' : 'Jarvis'}
              {m.src && <em className="hud-src">{m.src}</em>}
            </span>
            {m.text || (busy && i === messages.length - 1 ? '▍' : '')}
          </div>
        ))}
        <div ref={bottom} />
      </div>

      <div className="hud-input">
        <input
          value={input}
          placeholder="Speak to Jarvis…"
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={(e) => e.key === 'Enter' && send()}
        />
        <button className="hud-btn" onClick={() => send()} disabled={busy || !input.trim()}>
          {busy ? '···' : 'Send'}
        </button>
      </div>
    </div>
  )
}
