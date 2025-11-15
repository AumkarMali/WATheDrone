import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import './index.css'
import App from './App.jsx'

// Error boundary for catching render errors
console.log('Starting app initialization...')
console.log('Root element:', document.getElementById('root'))

try {
  const rootElement = document.getElementById('root')
  if (!rootElement) {
    throw new Error('Root element not found')
  }
  
  console.log('Creating React root...')
  const root = createRoot(rootElement)
  
  console.log('Rendering App component...')
  root.render(
    <StrictMode>
      <App />
    </StrictMode>,
  )
  
  console.log('App rendered successfully!')
} catch (error) {
  console.error('Failed to render app:', error)
  const errorDiv = document.createElement('div')
  errorDiv.style.cssText = 'padding: 20px; font-family: sans-serif; background: #fff; color: #000;'
  errorDiv.innerHTML = `
    <h1>Error Loading App</h1>
    <p><strong>Error:</strong> ${error.message}</p>
    <pre style="background: #f0f0f0; padding: 10px; overflow: auto;">${error.stack}</pre>
    <p>Check the browser console (F12) for more details.</p>
  `
  document.body.appendChild(errorDiv)
}
