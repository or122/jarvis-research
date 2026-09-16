import { StrictMode, useState } from 'react'
import { createRoot } from 'react-dom/client'
import './index.css'
import App from './App.tsx'
import Rooms from './Rooms.tsx'
import JarvisOS from './JarvisOS.tsx'

type View = 'os' | 'chat' | 'rooms'

/** Three ways in: the OS (default), the plain chat, and rooms. */
function Flow() {
  const [view, setView] = useState<View>(() =>
    localStorage.getItem('flow_room') ? 'rooms' : 'os',
  )

  if (view === 'os') return <JarvisOS onExit={() => setView('chat')} />
  if (view === 'rooms') return <Rooms onLeave={() => setView('os')} />
  return <App onRooms={() => setView('rooms')} onHud={() => setView('os')} />
}

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <Flow />
  </StrictMode>,
)
