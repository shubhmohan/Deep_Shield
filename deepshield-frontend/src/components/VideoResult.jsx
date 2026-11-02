import { ReportHeader, RiskBadge, Gauge } from './ReportHeader'

export default function VideoResult({ result, filename }) {
  const isReal = result.verdict === 'REAL'
  const accentColor = isReal ? '#1f8f78' : '#c94a30'
  const fs = result.frame_stats

  return (
    <div className="report">
      <ReportHeader filename={filename} verdict={result.verdict} />

      <div className="report-body">
        <div>
          <div className="section-title">Confidence &amp; risk</div>

          <Gauge label="Mean fake probability across frames" value={result.fake_probability} color={accentColor} />

          <div className="metric-row">
            <span>Confidence</span>
            <span className="metric-value">{result.confidence.toFixed(1)}%</span>
          </div>
          <div className="metric-row">
            <span>Risk level</span>
            <RiskBadge level={result.risk_level} />
          </div>
          <div className="metric-row">
            <span>Frames sampled</span>
            <span className="metric-value">{fs.frames_sampled}</span>
          </div>
          <div className="metric-row">
            <span>Frames with a detected face</span>
            <span className="metric-value">{fs.frames_with_face_detected}</span>
          </div>
          <div className="metric-row">
            <span>Frame-to-frame variance</span>
            <span className="metric-value">{fs.std_fake_probability.toFixed(3)}</span>
          </div>
          <div className="metric-row">
            <span>Processing time</span>
            <span className="metric-value">{result.processing_time_sec.toFixed(2)}s</span>
          </div>

          <div className="section-title" style={{ marginTop: 20 }}>Blink pattern signal</div>
          <div className="frame-note">
            <strong style={{ color: '#14181f' }}>
              {result.blink_analysis.signal === 'suspicious' ? 'Flagged as unusual — ' : ''}
            </strong>
            {result.blink_analysis.note}
          </div>
        </div>

        <div>
          <div className="section-title">Evidence — most suspicious frame</div>
          {result.most_suspicious_frame.heatmap_overlay && (
            <img
              className="heatmap-img"
              src={result.most_suspicious_frame.heatmap_overlay}
              alt="Grad-CAM heatmap of the most suspicious sampled frame"
            />
          )}
          <div className="frame-note">
            Frame #{result.most_suspicious_frame.frame_index} scored highest for manipulation
            evidence ({(result.most_suspicious_frame.fake_probability * 100).toFixed(1)}% fake probability)
            among all sampled frames.
          </div>
        </div>
      </div>
    </div>
  )
}
