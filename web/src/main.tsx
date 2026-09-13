import { StrictMode, useState } from 'react'
import { createRoot } from 'react-dom/client'
import './index.css'
import App from './App.tsx'
import Rooms from './Rooms.tsx'

/** Two ways to use Flow: alone, or in a room with other people. */
function Flow() {
  const [inRooms, setInRooms] = useState(() => Boolean(localStorage.getItem('flow_room')))
  return inRooms ? (
    <Rooms onLeave={() => setInRooms(false)} />
  ) : (
    <App onRooms={() => setInRooms(true)} />
  )
}

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <Flow />
  </StrictMode>,
)
