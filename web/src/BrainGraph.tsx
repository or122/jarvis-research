import { useEffect, useRef } from 'react'

/**
 * The knowledge graph — Jarvis's memory drawn as a brain.
 *
 * A force simulation written by hand rather than pulled from a library: three
 * forces, about forty lines, and it keeps the app's promise of loading
 * instantly with the wifi off.
 *
 *   repulsion  every node pushes every other apart      (spreads the cloud)
 *   springs    linked nodes pull together               (forms the clusters)
 *   centring   everything drifts gently to the middle   (stops it escaping)
 *
 * Repulsion is the expensive one at O(n²), which is fine for 62 nodes and
 * would need a quadtree past a few hundred.
 */
export type Node = {
  id: number
  label: string
  answer: string
  group: string
  colour: string
  size: number
  degree: number
  x?: number
  y?: number
  vx?: number
  vy?: number
}
export type Link = { source: number; target: number; weight: number }

type Props = {
  nodes: Node[]
  links: Link[]
  hidden: Set<string>
  repel: number
  linkLength: number
  onPick: (n: Node | null) => void
  picked: Node | null
  pulse: number
}

export default function BrainGraph({
  nodes, links, hidden, repel, linkLength, onPick, picked, pulse,
}: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const stateRef = useRef({ nodes, links, hidden, repel, linkLength, picked, pulse })
  stateRef.current = { nodes, links, hidden, repel, linkLength, picked, pulse }

  // Positions live outside React. Re-rendering 62 nodes at 60fps through the
  // virtual DOM would be pointless work; the canvas owns them.
  const posRef = useRef<Map<number, Node>>(new Map())

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    const ctx = canvas.getContext('2d')
    if (!ctx) return

    const pos = posRef.current
    for (const n of nodes) {
      if (!pos.has(n.id)) {
        // Start on a ring rather than dead centre — from a single point the
        // repulsion explodes outward in one violent frame.
        const a = (n.id * 137.5 * Math.PI) / 180     // golden angle, spreads evenly
        const r = 120 + (n.id % 7) * 26
        pos.set(n.id, { ...n, x: Math.cos(a) * r, y: Math.sin(a) * r, vx: 0, vy: 0 })
      } else {
        Object.assign(pos.get(n.id)!, {
          colour: n.colour, group: n.group, size: n.size,
          degree: n.degree, label: n.label, answer: n.answer,
        })
      }
    }
    for (const id of [...pos.keys()]) {
      if (!nodes.some((n) => n.id === id)) pos.delete(id)
    }

    let frame = 0
    let raf = 0

    const resize = () => {
      const dpr = Math.min(window.devicePixelRatio || 1, 2)
      const { clientWidth: w, clientHeight: h } = canvas
      canvas.width = w * dpr
      canvas.height = h * dpr
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0)
    }
    resize()
    window.addEventListener('resize', resize)

    const tick = () => {
      const s = stateRef.current
      const w = canvas.clientWidth
      const h = canvas.clientHeight
      const cx = w / 2
      const cy = h / 2
      const visible = [...pos.values()].filter((n) => !s.hidden.has(n.group))

      // --- forces -------------------------------------------------------
      for (let i = 0; i < visible.length; i++) {
        const a = visible[i]
        for (let j = i + 1; j < visible.length; j++) {
          const b = visible[j]
          let dx = (b.x ?? 0) - (a.x ?? 0)
          let dy = (b.y ?? 0) - (a.y ?? 0)
          let d2 = dx * dx + dy * dy
          if (d2 < 1) {
            // Exactly overlapping nodes give a divide-by-zero and fling one to
            // infinity, so nudge them apart deterministically.
            dx = (a.id - b.id) * 0.01 || 0.01
            dy = 0.01
            d2 = dx * dx + dy * dy
          }
          const force = s.repel / d2
          const d = Math.sqrt(d2)
          const fx = (dx / d) * force
          const fy = (dy / d) * force
          a.vx = (a.vx ?? 0) - fx
          a.vy = (a.vy ?? 0) - fy
          b.vx = (b.vx ?? 0) + fx
          b.vy = (b.vy ?? 0) + fy
        }
      }

      for (const link of s.links) {
        const a = pos.get(link.source)
        const b = pos.get(link.target)
        if (!a || !b || s.hidden.has(a.group) || s.hidden.has(b.group)) continue
        const dx = (b.x ?? 0) - (a.x ?? 0)
        const dy = (b.y ?? 0) - (a.y ?? 0)
        const d = Math.hypot(dx, dy) || 1
        // Heavier links (more shared words) pull harder, so strongly related
        // facts sit closer together.
        const k = 0.0016 * Math.min(link.weight, 4)
        const pull = (d - s.linkLength) * k
        a.vx = (a.vx ?? 0) + (dx / d) * pull
        a.vy = (a.vy ?? 0) + (dy / d) * pull
        b.vx = (b.vx ?? 0) - (dx / d) * pull
        b.vy = (b.vy ?? 0) - (dy / d) * pull
      }

      for (const n of visible) {
        n.vx = ((n.vx ?? 0) - (n.x ?? 0) * 0.0016) * 0.86   // centring + damping
        n.vy = ((n.vy ?? 0) - (n.y ?? 0) * 0.0016) * 0.86
        n.x = (n.x ?? 0) + (n.vx ?? 0)
        n.y = (n.y ?? 0) + (n.vy ?? 0)
      }

      // --- draw ---------------------------------------------------------
      ctx.clearRect(0, 0, w, h)
      frame++

      const neighbours = new Set<number>()
      if (s.picked) {
        for (const l of s.links) {
          if (l.source === s.picked.id) neighbours.add(l.target)
          if (l.target === s.picked.id) neighbours.add(l.source)
        }
      }

      ctx.lineWidth = 1
      for (const link of s.links) {
        const a = pos.get(link.source)
        const b = pos.get(link.target)
        if (!a || !b || s.hidden.has(a.group) || s.hidden.has(b.group)) continue
        const lit =
          s.picked && (link.source === s.picked.id || link.target === s.picked.id)
        ctx.strokeStyle = lit ? 'rgba(199,125,255,0.85)' : 'rgba(120,170,210,0.13)'
        ctx.lineWidth = lit ? 1.6 : 1
        ctx.beginPath()
        ctx.moveTo(cx + (a.x ?? 0), cy + (a.y ?? 0))
        ctx.lineTo(cx + (b.x ?? 0), cy + (b.y ?? 0))
        ctx.stroke()
      }

      for (const n of visible) {
        const x = cx + (n.x ?? 0)
        const y = cy + (n.y ?? 0)
        const dim = s.picked && n.id !== s.picked.id && !neighbours.has(n.id)
        // A slow shimmer keeps the graph alive once the physics settles.
        const shimmer = 1 + Math.sin(frame * 0.02 + n.id) * 0.06
        const r = n.size * shimmer * (s.picked?.id === n.id ? 1.7 : 1)

        ctx.globalAlpha = dim ? 0.25 : 1
        ctx.beginPath()
        ctx.arc(x, y, r + 7, 0, Math.PI * 2)
        ctx.fillStyle = n.colour + '22'
        ctx.fill()

        ctx.beginPath()
        ctx.arc(x, y, r, 0, Math.PI * 2)
        ctx.fillStyle = n.colour
        ctx.fill()

        if (s.picked?.id === n.id || (!s.picked && n.degree >= 10)) {
          ctx.globalAlpha = 1
          ctx.font = '10px ui-monospace, Menlo, monospace'
          ctx.fillStyle = '#d8ecff'
          ctx.textAlign = 'center'
          ctx.fillText(n.label.slice(0, 34), x, y - r - 9)
        }
        ctx.globalAlpha = 1
      }

      raf = requestAnimationFrame(tick)
    }

    raf = requestAnimationFrame(tick)
    return () => {
      cancelAnimationFrame(raf)
      window.removeEventListener('resize', resize)
    }
  }, [nodes, links])

  function handleClick(e: React.MouseEvent<HTMLCanvasElement>) {
    const canvas = canvasRef.current
    if (!canvas) return
    const rect = canvas.getBoundingClientRect()
    const mx = e.clientX - rect.left - rect.width / 2
    const my = e.clientY - rect.top - rect.height / 2

    let best: Node | null = null
    let bestD = 22            // generous: these are small targets
    for (const n of posRef.current.values()) {
      if (hidden.has(n.group)) continue
      const d = Math.hypot((n.x ?? 0) - mx, (n.y ?? 0) - my)
      if (d < bestD) {
        bestD = d
        best = n
      }
    }
    onPick(best)
  }

  return <canvas ref={canvasRef} className="brain-canvas" onClick={handleClick} />
}
