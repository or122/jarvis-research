import { StrictMode, useState } from 'react'
import { createRoot } from 'react-dom/client'
import './index.css'
import App from './App.tsx'
import Rooms from './Rooms.tsx'
import Hud from './Hud.tsx'

type View = 'hud' | 'chat' | 'rooms'

/** Three ways in: the Jarvis HUD (default), the plain chat, and rooms. */
function Flow() {
  const [view, setView] = useState<View>(() =>
    localStorage.getItem('flow_room') ? 'rooms' : 'hud',
  )

  if (view === 'hud') return <Hud onExit={() => setView('chat')} />
  if (view === 'rooms') return <Rooms onLeave={() => setView('hud')} />
  return <App onRooms={() => setView('rooms')} onHud={() => setView('hud')} />
}

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <Flow />
  </StrictMode>,
)
