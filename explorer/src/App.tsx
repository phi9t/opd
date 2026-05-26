import { Activity } from 'lucide-react'
import OpdExplorer from './opd/OpdExplorer'

export default function App() {
  return (
    <div className="relative min-h-screen">
      <div className="observatory-bg" aria-hidden="true" />

      <div className="explorer-container">
        <header className="explorer-header family-header">
          <div className="header-title-section">
            <a href="#main-content" className="skip-link">
              Skip to main content
            </a>
            <div className="header-title-row">
              <Activity
                className="header-logo-icon"
                size={32}
                aria-hidden="true"
              />
              <h1>OPD Training Explorer</h1>
            </div>
            <p>
              On-Policy Distillation lab metrics — step timings, KL curves, and
              run configuration.
            </p>
          </div>
        </header>

        <div id="main-content">
          <OpdExplorer />
        </div>
      </div>
    </div>
  )
}
