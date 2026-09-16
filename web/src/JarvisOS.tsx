import { useEffect, useRef, useState } from 'react'
import BrainGraph, { type Link, type Node } from './BrainGraph'
import './os.css'

const SERVER = `http://${location.hostname}:8000`

type Group = { group: string; colour: string; count: number }
type Hub = { label: string; degree: number; colour: string }
type Brain = { nodes: Node[]; links: Link[]; groups: Group[]; hubs: Hub[] }
type Health = { ready: boolean; val_loss?: number; iter?: number; facts?: number }

export default function JarvisOS({ onExit }: { onExit: () => void }) {
  const [brain, setBrain] = useState<Brain | null>(null)
  const [health, setHealth] = useState<Health | null>(null)
  const [hidden, setHidden] = useState<Set<string>>(new Set())
  const [picked, setPicked] = useState<Node | null>(null)
  const [repel, setRepel] = useState(900)
  const [linkLength, setLinkLength] = useState(70)
  const [input, setInput] = useState('')
  const [reply, setReply] = useState('')
  const [busy, setBusy] = useState(false)
  const pulse = useRef(0)

  const load = () =>
    fetch(`${SERVER}/brain`)
      .then((r) => r.json())
      .then(setBrain)
      .catch(() => setBrain(null))

  useEffect(() => {
    load()
    fetch(`${SERVER}/health`).then((r) => r.json()).then(setHealth).catch(() => {})
  }, [])

  function toggle(group: string) {
    setHidden((h) => {
      const next = new Set(h)
      next.has(group) ? next.delete(group) : next.add(group)
      return next
    })
  }

  async function ask() {
    const prompt = input.trim()
    if (!prompt || busy) return
    setInput('')
    setBusy(true)
    setReply('')
    try {
      const res = await fetch(`${SERVER}/generate`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ prompt, max_tokens: 90 }),
      })
      const reader = res.body!.getReader()
      const decoder = new TextDecoder()
      let buffer = ''
      for (;;) {
        const { done, value } = await reader.read()
        if (done) break
        buffer += decoder.decode(value, { stream: true })
        const frames = buffer.split('\n\n')
        buffer = frames.pop() ?? ''
        for (const f of frames) {
          if (!f.startsWith('data: ')) continue
          const d = f.slice(6)
          if (d === '[DONE]') continue
          const { t } = JSON.parse(d) as { t: string }
          setReply((r) => r + t)
          pulse.current++
        }
      }
      // Teaching Jarvis changes the graph, so refresh it after every exchange.
      load()
    } catch {
      setReply('Connection lost.')
    } finally {
      setBusy(false)
    }
  }

  const shown = brain
    ? brain.nodes.filter((n) => !hidden.has(n.group)).length
    : 0

  return (
    <div className="os">
      {/* ---------------------------------------------------------- left */}
      <aside className="os-panel os-left">
        <div className="os-title">
          JARVIS OS
          <button className="os-x" onClick={onExit}>
            ✕
          </button>
        </div>
        <div className="os-sub">
          {brain ? `${brain.nodes.length} facts · ${brain.links.length} connections` : 'linking…'}
        </div>

        <div className="os-block">
          <h4>Inspector</h4>
          {picked ? (
            <div className="os-inspect">
              <span className="os-dot" style={{ background: picked.colour }} />
              <div className="os-q">{picked.label}</div>
              <div className="os-a">{picked.answer}</div>
              <div className="os-meta">
                {picked.group} · {picked.degree} connections
              </div>
              <button className="os-btn-sm" onClick={() => setPicked(null)}>
                clear
              </button>
            </div>
          ) : (
            <p className="os-hint">
              Click a node to focus it — only that node and its connections stay
              lit, so you can read the path.
            </p>
          )}
        </div>

        <div className="os-block">
          <h4>Top hubs</h4>
          {brain?.hubs.map((h) => (
            <div key={h.label} className="os-hub">
              <span className="os-dot sm" style={{ background: h.colour }} />
              <span className="os-hub-label">{h.label}</span>
              <span className="os-hub-n">{h.degree}</span>
            </div>
          ))}
        </div>

        <div className="os-block">
          <h4>Forces</h4>
          <label className="os-slider">
            <span>Repel</span>
            <input
              type="range"
              min={200}
              max={2600}
              value={repel}
              onChange={(e) => setRepel(+e.target.value)}
            />
          </label>
          <label className="os-slider">
            <span>Link length</span>
            <input
              type="range"
              min={30}
              max={190}
              value={linkLength}
              onChange={(e) => setLinkLength(+e.target.value)}
            />
          </label>
        </div>
      </aside>

      {/* -------------------------------------------------------- centre */}
      <main className="os-stage">
        {brain ? (
          <BrainGraph
            nodes={brain.nodes}
            links={brain.links}
            hidden={hidden}
            repel={repel}
            linkLength={linkLength}
            onPick={setPicked}
            picked={picked}
            pulse={pulse.current}
          />
        ) : (
          <div className="os-offline">
            <p>NO LINK TO THE BRAIN</p>
            <p className="os-hint">Start the model server on port 8000.</p>
          </div>
        )}

        {/* the reply, as a caption under the graph */}
        {(reply || busy) && (
          <div className="os-caption">
            <span className="os-caption-who">JARVIS</span>
            {reply || '…'}
          </div>
        )}

        <div className="os-bar">
          <input
            value={input}
            placeholder="Ask Jarvis…"
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => e.key === 'Enter' && ask()}
          />
          <button onClick={ask} disabled={busy || !input.trim()}>
            {busy ? '···' : 'Send'}
          </button>
        </div>
      </main>

      {/* --------------------------------------------------------- right */}
      <aside className="os-panel os-right">
        <div className="os-block">
          <h4>Filter</h4>
          {brain?.groups.map((g) => (
            <button
              key={g.group}
              className={`os-filter ${hidden.has(g.group) ? 'off' : ''}`}
              onClick={() => toggle(g.group)}
            >
              <span className="os-dot sm" style={{ background: g.colour }} />
              <span className="os-filter-name">{g.group}</span>
              <span className="os-filter-n">{g.count}</span>
            </button>
          ))}
          <div className="os-showing">{shown} showing</div>
        </div>

        {/* the J.A.R.V.I.S. ring */}
        <div className="os-ring-wrap">
          <div className={`os-ring ${busy ? 'busy' : ''}`}>
            <svg viewBox="0 0 120 120">
              <circle cx="60" cy="60" r="54" className="r-line" />
              <circle cx="60" cy="60" r="46" className="r-line r-dash" />
              <circle cx="60" cy="60" r="34" className="r-line r-dash2" />
              {Array.from({ length: 36 }, (_, i) => {
                const a = (i * 10 * Math.PI) / 180
                return (
                  <line
                    key={i}
                    x1={60 + 48 * Math.cos(a)}
                    y1={60 + 48 * Math.sin(a)}
                    x2={60 + 53 * Math.cos(a)}
                    y2={60 + 53 * Math.sin(a)}
                    className="r-tick"
                  />
                )
              })}
            </svg>
            <div className="os-ring-core" />
            <div className="os-ring-label">J.A.R.V.I.S.</div>
          </div>
          <div className="os-badges">
            <span className={`os-badge ${health?.ready ? 'on' : 'off'}`}>
              ● {health?.ready ? 'ONLINE' : 'OFFLINE'}
            </span>
            <span className="os-badge">FLOW 1.0</span>
          </div>
          {health?.ready && (
            <div className="os-stats">
              <span>
                LOSS <b>{health.val_loss}</b>
              </span>
              <span>
                ITER <b>{health.iter}</b>
              </span>
            </div>
          )}
        </div>
      </aside>
    </div>
  )
}
