export function ReportHeader({ filename, verdict }) {
  const isReal = verdict === 'REAL'
  return (
    <div className="report-header">
      <div>
        <div className="report-label">Analysis of</div>
        <div className="report-filename">{filename}</div>
      </div>
      <div className={`verdict-stamp ${isReal ? 'real' : 'fake'}`}>
        {isReal ? 'Authentic' : 'Manipulated'}
      </div>
    </div>
  )
}

export function RiskBadge({ level }) {
  const cls = `risk-badge risk-${level.toLowerCase()}`
  return <span className={cls}>{level} risk</span>
}

export function Gauge({ label, value, color }) {
  return (
    <div style={{ marginBottom: 14 }}>
      <div className="metric-row" style={{ border: 'none', padding: '0 0 4px' }}>
        <span>{label}</span>
        <span className="metric-value">{(value * 100).toFixed(1)}%</span>
      </div>
      <div className="gauge-track">
        <div
          className="gauge-fill"
          style={{ width: `${value * 100}%`, background: color }}
        />
      </div>
    </div>
  )
}
