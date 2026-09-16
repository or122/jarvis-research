import { useEffect, useRef, useState } from 'react'
import BrainGraph, { type Link, type Node } from './BrainGraph'
import './os.css'

const SERVER = `http://${location.hostname}:8000`

type Group = { group: string; colour: string; count: number }
type Hub = { label: string; degree: number; colour: string }
type Brain = { nodes: Node[]; links: Link[]; groups: Group[]; hubs: Hub[] }
type Health = {
  ready: boolean
  val_loss?: number
  iter?: number
  facts?: number
  petrol?: boolean
  petrol_model?: string
}

/** How much to trust each source, and what to call it.
 *  certain = real code or a stored fact, and cannot be wrong
 *  strong  = a frontier model, usually right
 *  guess   = the 11M-parameter model, frequently wrong */
const CONFIDENCE: Record<string, string> = {
  memory: 'certain',
  command: 'certain',
  opus: 'strong',
  fable: 'strong',
  // The agent wrote a real file, but it cannot run one, so it has not checked
  // that the file works. Real code, unverified - that is 'strong', not
  // 'certain'.
  agent: 'strong',
  flow: 'guess',
}

const LABEL: Record<string, string> = {
  memory: '🔋 remembered · certain',
  command: '⚙ ran it · certain',
  opus: '⛽ Claude Opus',
  fable: '⛽ Claude Fable',
  agent: '🛠 built it · in the workspace',
  flow: '🔋 my small model · a guess',
}

export default function JarvisOS({ onExit }: { onExit: () => void }) {
  const [brain, setBrain] = useState<Brain | null>(null)
  const [health, setHealth] = useState<Health | null>(null)
  const [hidden, setHidden] = useState<Set<string>>(new Set())
  const [picked, setPicked] = useState<Node | null>(null)
  const [repel, setRepel] = useState(900)
  const [linkLength, setLinkLength] = useState(70)
  const [input, setInput] = useState('')
  const [reply, setReply] = useState('')
  // Which engine answered, so the dashboard can show it.
  const [engine, setEngine] = useState<string>('')
  // The question that produced the current reply. A correction has to teach
  // Jarvis the answer to THAT question, so it must survive the input clearing.
  const [asked, setAsked] = useState('')
  const [fixing, setFixing] = useState(false)
  const [fix, setFix] = useState('')
  const [fixed, setFixed] = useState('')
  // What the agent is doing right now, and what it left behind when it is done.
  const [status, setStatus] = useState('')
  const [built, setBuilt] = useState<{ files: string[]; cost: number } | null>(null)
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
    setEngine('')
    setAsked(prompt)
    setFixing(false)
    setFixed('')
    setStatus('')
    setBuilt(null)
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
          const { t, src, status, files, cost } = JSON.parse(d) as {
            t: string; src?: string; status?: string
            files?: string[]; cost?: number
          }
          if (src) setEngine(src)
          // A build takes half a minute. These frames carry no text, only news
          // of what is happening, so the screen is never silently frozen.
          if (status) setStatus(status)
          if (files) setBuilt({ files, cost: cost ?? 0 })
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

  /** Correcting Jarvis teaches Jarvis - the correction becomes a stored fact,
   *  so the same question is answered from memory next time. */
  async function teachCorrection() {
    const answer = fix.trim()
    if (!answer || !asked) return
    try {
      const res = await fetch(`${SERVER}/teach`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question: asked, answer }),
      })
      const { facts } = (await res.json()) as { facts: number }
      setFixed(`Learned. Jarvis now knows ${facts} facts.`)
      setReply(answer)
      setEngine('memory')
      setFix('')
      setFixing(false)
      load()                       // the new fact is a new node on the graph
      setHealth((h) => (h ? { ...h, facts } : h))
    } catch {
      setFixed('Could not save that.')
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

        {/* The reply. How sure Jarvis is changes how it LOOKS - a stored fact
            and an 11M-parameter guess must not appear equally authoritative,
            which is the "false confidence" trust research warns about. */}
        {(reply || busy) && (
          <div className={`os-caption conf-${CONFIDENCE[engine] ?? 'guess'}`}>
            <span className="os-caption-who">
              JARVIS
              {engine && <em className={`os-engine ${engine}`}>{LABEL[engine] ?? LABEL.flow}</em>}
            </span>
            {/* A build runs for half a minute with nothing to stream, so it
                says what it is doing instead of showing an empty box. */}
            {busy && status && !reply && (
              <span className="os-status">⟳ {status}</span>
            )}

            {reply || (busy && status ? '' : '…')}

            {/* What the agent actually left on the disk. The claim "I built
                it" is only checkable if the files are named. */}
            {built && built.files.length > 0 && (
              <div className="os-built">
                <span className="os-built-head">
                  files in workspace/
                  {built.cost > 0 && <em> · ${built.cost.toFixed(2)}</em>}
                </span>
                <ul>
                  {built.files.map((f) => (
                    <li key={f}>{f}</li>
                  ))}
                </ul>
              </div>
            )}

            {/* Every answer can be corrected. That is the strongest trust
                finding, and here the correction also teaches Jarvis. */}
            {reply && !busy && !fixing && !fixed && (
              <button className="os-wrong" onClick={() => setFixing(true)}>
                ✎ that's wrong
              </button>
            )}

            {fixing && (
              <div className="os-fix">
                <label>What should Jarvis have said?</label>
                <input
                  autoFocus
                  value={fix}
                  placeholder="The right answer…"
                  onChange={(e) => setFix(e.target.value)}
                  onKeyDown={(e) => e.key === 'Enter' && teachCorrection()}
                />
                <div className="os-fix-row">
                  <button className="os-fix-go" disabled={!fix.trim()} onClick={teachCorrection}>
                    Teach it
                  </button>
                  <button className="os-fix-no" onClick={() => setFixing(false)}>
                    cancel
                  </button>
                </div>
              </div>
            )}

            {fixed && <div className="os-fixed">✓ {fixed}</div>}
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
            <span className="os-badge elec">🔋 FLOW</span>
            <span className={`os-badge ${health?.petrol ? 'petrol' : 'off'}`}>
              ⛽ {health?.petrol ? 'FABLE' : 'NO KEY'}
            </span>
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
