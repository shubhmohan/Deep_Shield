import { ReportHeader, RiskBadge, Gauge } from './ReportHeader'

export default function ImageResult({ result, filename }) {
  const isReal = result.verdict === 'REAL'
  const accentColor = isReal ? '#1f8f78' : '#c94a30'

  return (
    <div className="report">
      <ReportHeader filename={filename} verdict={result.verdict} />

      <div className="report-body">
        <div>
          <div className="section-title">Confidence &amp; risk</div>

          <Gauge label="Fake probability" value={result.fake_probability} color={accentColor} />

          <div className="metric-row">
            <span>Confidence</span>
            <span className="metric-value">{result.confidence.toFixed(1)}%</span>
          </div>
          <div className="metric-row">
            <span>Risk level</span>
            <RiskBadge level={result.risk_level} />
          </div>
          <div className="metric-row">
            <span>ViT branch score</span>
            <span className="metric-value">{(result.model_breakdown.vit_score * 100).toFixed(1)}%</span>
          </div>
          <div className="metric-row">
            <span>EfficientNet branch score</span>
            <span className="metric-value">{(result.model_breakdown.efficientnet_score * 100).toFixed(1)}%</span>
          </div>
          <div className="metric-row">
            <span>Processing time</span>
            <span className="metric-value">{result.processing_time_sec.toFixed(2)}s</span>
          </div>
        </div>

        {result.heatmap_overlay && (
          <div>
            <div className="section-title">Evidence — Grad-CAM attention</div>
            <img className="heatmap-img" src={result.heatmap_overlay} alt="Grad-CAM heatmap showing which regions influenced the verdict" />
            <div className="frame-note">
              Highlighted regions show where the model concentrated evidence when forming this verdict.
            </div>
          </div>
        )}
      </div>
    </div>
  )
}
