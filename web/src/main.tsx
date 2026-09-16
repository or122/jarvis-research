import { StrictMode, useState } from 'react'
import { createRoot } from 'react-dom/client'
import './index.css'
import App from './App.tsx'
import Rooms from './Rooms.tsx'
import Hud from './Hud.tsx'
import JarvisOS from './JarvisOS.tsx'

type View = 'os' | 'hud' | 'chat' | 'rooms'

/**
 * Four ways in:
 *   os     the knowledge graph - the default, and what Jarvis really is
 *   hud    the arc-reactor chat, for talking without the graph
 *   chat   the plain chat, when you just want the text
 *   rooms  several people at once
 */
function Flow() {
  const [view, setView] = useState<View>(() =>
    localStorage.getItem('flow_room') ? 'rooms' : 'os',
  )

  if (view === 'os') return <JarvisOS onExit={() => setView('hud')} />
  if (view === 'hud') return <Hud onExit={() => setView('chat')} />
  if (view === 'rooms') return <Rooms onLeave={() => setView('os')} />
  return <App onRooms={() => setView('rooms')} onHud={() => setView('os')} />
}

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <Flow />
  </StrictMode>,
)
