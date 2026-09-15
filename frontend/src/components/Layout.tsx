import {
  Activity,
  ArrowUpRight,
  Boxes,
  ChartNoAxesCombined,
  FlaskConical,
  GitCompareArrows,
  Terminal,
} from 'lucide-react';
import { NavLink, Outlet } from 'react-router-dom';

export function Layout() {
  return (
    <div className="app-shell">
      <aside className="sidebar">
        <NavLink to="/" className="brand">
          <span className="brand-icon">
            <FlaskConical size={22} />
          </span>
          <span>
            Model Eval<span className="brand-sub">CONSOLE</span>
          </span>
        </NavLink>
        <div className="workspace">
          <span className="workspace-avatar">ML</span>
          <div>
            Research workspace<span>Local environment</span>
          </div>
          <span className="live-dot" />
        </div>
        <div className="nav-label">PLATFORM</div>
        <nav aria-label="Main navigation">
          <NavLink to="/compare">
            <GitCompareArrows size={18} />
            Compare models
          </NavLink>
          <NavLink to="/candidates">
            <Boxes size={18} />
            Candidates
          </NavLink>
          <NavLink to="/runs">
            <Activity size={18} />
            Evaluation runs
          </NavLink>
        </nav>
        <div className="sidebar-bottom">
          <div className="sidebar-note">
            <ChartNoAxesCombined size={18} />
            <strong>Evidence before release.</strong>
            <p>Reproducible runs. Paired comparisons. Honest uncertainty.</p>
          </div>
          <a href="http://localhost:8000/docs" target="_blank" rel="noreferrer">
            <Terminal size={15} />
            API reference
            <ArrowUpRight size={14} />
          </a>
          <span className="sidebar-version">v0.1 · CPU inference</span>
        </div>
      </aside>
      <div className="main-shell">
        <header className="topbar">
          <span>
            AI Platform <span className="breadcrumb-slash">/</span> Model evaluation
          </span>
          <span className="environment">
            <span className="live-dot" />
            Local workspace
          </span>
        </header>
        <main id="main-content">
          <Outlet />
        </main>
        <footer>
          Model Eval Console
          <span>Quality gates describe a benchmark, not a guarantee of deployment safety.</span>
        </footer>
      </div>
    </div>
  );
}
