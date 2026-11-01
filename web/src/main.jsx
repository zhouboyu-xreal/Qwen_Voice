import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import App from './App.jsx'
import MemoryCenter from './memory/MemoryCenter.jsx'
import { setRuntimeLanguage, t } from './i18n.js'
import './styles.css'

const parameters = new URLSearchParams(window.location.search)
const memoryView = parameters.get('view') === 'memory'
if (memoryView) {
  document.documentElement.dataset.view = 'memory'
  setRuntimeLanguage(parameters.get('lang') || navigator.language)
  document.title = t('我的记忆')
}

if (parameters.get('desktop') === 'orb') {
  document.documentElement.dataset.desktop = 'orb'
}

createRoot(document.getElementById('root')).render(
  <StrictMode>
    {memoryView ? <MemoryCenter /> : <App />}
  </StrictMode>,
)
