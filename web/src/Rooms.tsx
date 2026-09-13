import { useEffect, useRef, useState } from 'react'

const SERVER = `http://${location.hostname}:8000`

type Msg = { seq: number; author: string; text: string }
type Room = {
  name: string
  code: string
  paid: boolean
  seats: number
  members: string[]
  messages: Msg[]
}

/** Remembers which room you were in, so a refresh does not throw you out. */
function saved() {
  try {
    return JSON.parse(localStorage.getItem('flow_room') ?? 'null')
  } catch {
    return null
  }
}

export default function Rooms({ onLeave }: { onLeave: () => void }) {
  const [me, setMe] = useState(() => localStorage.getItem('flow_name') ?? '')
  const [session, setSession] = useState<{ id: number; name: string } | null>(saved)
  const [room, setRoom] = useState<Room | null>(null)
  const [messages, setMessages] = useState<Msg[]>([])
  const [input, setInput] = useState('')
  const [code, setCode] = useState('')
  const [roomName, setRoomName] = useState('')
  const [full, setFull] = useState<{ seats: number; members: string[] } | null>(null)
  const bottom = useRef<HTMLDivElement>(null)
  const seen = useRef(0)

  // Poll for anything said since the last message we have. Polling rather
  // than a socket because a family room is a handful of people, and this
  // needs no extra moving parts.
  useEffect(() => {
    if (!session) return
    let alive = true

    const tick = async () => {
      try {
        const res = await fetch(`${SERVER}/rooms/${session.id}?since=${seen.current}`)
        if (!res.ok) return
        const data = (await res.json()) as Room
        if (!alive) return
        setRoom(data)
        if (data.messages.length) {
          seen.current = data.messages[data.messages.length - 1].seq
          setMessages((m) => [...m, ...data.messages])
        }
      } catch {
        // server restarting or wifi blipped; the next tick retries
      }
    }

    tick()
    const timer = setInterval(tick, 1500)
    return () => {
      alive = false
      clearInterval(timer)
    }
  }, [session])

  useEffect(() => bottom.current?.scrollIntoView({ behavior: 'smooth' }), [messages])

  async function createRoom() {
    const res = await fetch(`${SERVER}/rooms`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ name: roomName || 'My Room', owner: me || 'You' }),
    })
    const r = await res.json()
    enter(r.id, r.name)
  }

  async function joinRoom() {
    setFull(null)
    const res = await fetch(`${SERVER}/rooms/join`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ code: code.toUpperCase(), name: me || 'Guest' }),
    })
    const info = await res.json()
    // 402 Payment Required: the room is full and more seats cost money.
    if (res.status === 402) {
      setFull(info.error === 'full' ? info : null)
      return
    }
    if (!res.ok) return
    enter(info.room_id, info.name)
  }

  function enter(id: number, name: string) {
    localStorage.setItem('flow_name', me)
    localStorage.setItem('flow_room', JSON.stringify({ id, name }))
    seen.current = 0
    setMessages([])
    setSession({ id, name })
  }

  function leave() {
    localStorage.removeItem('flow_room')
    setSession(null)
    setRoom(null)
    setMessages([])
  }

  async function send() {
    const text = input.trim()
    if (!text || !session) return
    setInput('')
    await fetch(`${SERVER}/rooms/${session.id}/say`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ name: me || 'Guest', text }),
    })
  }

  async function upgrade() {
    if (!room || !session) return
    await fetch(`${SERVER}/rooms/${session.id}/paid`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ paid: true }),
    })
    setFull(null)
  }

  // ---------------------------------------------------------------- lobby
  if (!session) {
    return (
      <div className="app">
        <header>
          <div className="brand">
            <span className="logo">✦</span>
            <div>
              <h1>Flow Rooms</h1>
              <p className="tagline">Talk to Flow together, on your own wifi.</p>
            </div>
          </div>
          <button className="chip" onClick={onLeave}>
            Back to chat
          </button>
        </header>

        <main>
          <div className="lobby">
            <label className="lab">Your name</label>
            <input
              className="field"
              value={me}
              placeholder="Or"
              onChange={(e) => setMe(e.target.value)}
            />

            <div className="split">
              <div className="half">
                <h3>Start a room</h3>
                <input
                  className="field"
                  value={roomName}
                  placeholder="Science Project"
                  onChange={(e) => setRoomName(e.target.value)}
                />
                <button className="primary" disabled={!me.trim()} onClick={createRoom}>
                  Create
                </button>
              </div>

              <div className="half">
                <h3>Join a room</h3>
                <input
                  className="field code-in"
                  value={code}
                  placeholder="5-letter code"
                  maxLength={5}
                  onChange={(e) => setCode(e.target.value.toUpperCase())}
                />
                <button disabled={!me.trim() || code.length < 5} onClick={joinRoom}>
                  Join
                </button>
              </div>
            </div>

            {full && (
              <div className="notice">
                <strong>That room is full.</strong> It has {full.seats} seats, taken by{' '}
                {full.members.join(' and ')}. The owner can add more seats by upgrading.
              </div>
            )}

            <p className="small">
              Everyone must be on the same wifi as the Mac running Flow.
            </p>
          </div>
        </main>
      </div>
    )
  }

  // ----------------------------------------------------------------- room
  const atLimit = room ? room.members.length >= room.seats : false

  return (
    <div className="app">
      <header>
        <div className="brand">
          <span className="logo">✦</span>
          <div>
            <h1>{room?.name ?? session.name}</h1>
            <p className="tagline">
              code <strong>{room?.code ?? '…'}</strong> · {room?.members.join(', ')}
            </p>
          </div>
        </div>
        <div className="status">
          <span className={`pill ${atLimit ? 'bad' : ''}`}>
            {room ? `${room.members.length}/${room.seats} seats` : '…'}
          </span>
          {room?.paid && <span className="pill paid">Unlimited</span>}
          <button className="chip" onClick={leave}>
            Leave
          </button>
        </div>
      </header>

      <main>
        {messages.length === 0 && (
          <div className="empty">
            <p className="big">Nobody has said anything yet.</p>
            <p className="small">
              Tell someone the code <strong>{room?.code}</strong> so they can join from
              their own computer.
            </p>
          </div>
        )}
        {messages.map((m) => (
          <div key={m.seq} className={`msg ${m.author === me ? 'you' : ''}`}>
            <span className="who">
              {m.author}
              {m.author === 'Flow' && <em className="tag">AI</em>}
            </span>
            <p>{m.text}</p>
          </div>
        ))}
        <div ref={bottom} />
      </main>

      {atLimit && !room?.paid && (
        <div className="notice">
          This room is full at {room?.seats} seats.{' '}
          <button className="linkish" onClick={upgrade}>
            Add more seats
          </button>
        </div>
      )}

      <footer>
        <textarea
          rows={1}
          value={input}
          placeholder="Say something to the room…"
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
              e.preventDefault()
              send()
            }
          }}
        />
        <button onClick={send} disabled={!input.trim()}>
          Send
        </button>
      </footer>
    </div>
  )
}
