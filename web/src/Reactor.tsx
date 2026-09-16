/**
 * The arc reactor — Jarvis's face.
 *
 * Three rings on different planes, tilted in 3D so they read as a physical
 * object rather than flat circles. Each ring spins at its own speed and in
 * alternating directions; that difference is what stops it looking like one
 * spinning disc.
 *
 * Built from CSS transforms and SVG, no 3D library. It has to load instantly
 * and work with the wifi off, which is the whole promise of this app.
 */
type State = 'idle' | 'listening' | 'thinking' | 'speaking'

export default function Reactor({ state }: { state: State }) {
  return (
    <div className={`reactor reactor-${state}`}>
      <div className="reactor-stage">
        {/* Outer ring: tick marks, like a dial */}
        <div className="ring ring-outer">
          <svg viewBox="0 0 200 200">
            <circle cx="100" cy="100" r="92" className="ring-line" />
            {Array.from({ length: 60 }, (_, i) => {
              const a = (i * 6 * Math.PI) / 180
              const long = i % 5 === 0
              const r1 = long ? 78 : 84
              return (
                <line
                  key={i}
                  x1={100 + r1 * Math.cos(a)}
                  y1={100 + r1 * Math.sin(a)}
                  x2={100 + 90 * Math.cos(a)}
                  y2={100 + 90 * Math.sin(a)}
                  className={long ? 'tick tick-long' : 'tick'}
                />
              )
            })}
          </svg>
        </div>

        {/* Middle ring: broken arcs, counter-rotating */}
        <div className="ring ring-mid">
          <svg viewBox="0 0 200 200">
            <circle cx="100" cy="100" r="66" className="ring-line arc-a" />
            <circle cx="100" cy="100" r="58" className="ring-line arc-b" />
          </svg>
        </div>

        {/* Inner ring: the segmented core */}
        <div className="ring ring-inner">
          <svg viewBox="0 0 200 200">
            {Array.from({ length: 8 }, (_, i) => {
              const a0 = (i * 45 + 4) * (Math.PI / 180)
              const a1 = (i * 45 + 41) * (Math.PI / 180)
              const r = 40
              return (
                <path
                  key={i}
                  d={`M ${100 + r * Math.cos(a0)} ${100 + r * Math.sin(a0)}
                      A ${r} ${r} 0 0 1 ${100 + r * Math.cos(a1)} ${100 + r * Math.sin(a1)}`}
                  className="segment"
                />
              )
            })}
          </svg>
        </div>

        {/* The core itself */}
        <div className="core">
          <div className="core-glow" />
          <div className="core-dot" />
        </div>
      </div>
    </div>
  )
}
