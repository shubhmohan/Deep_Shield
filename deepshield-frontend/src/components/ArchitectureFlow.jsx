const NODES = [
  { label: '01 INPUT', title: 'Face extraction', desc: 'MediaPipe locates and crops the subject\u2019s face from the image or sampled video frames.' },
  { label: '02 DETECTOR', title: 'CLIP ViT-L/14', desc: 'A Vision Transformer trained on the DF40 deepfake benchmark scores how likely the face is manipulated.' },
  { label: '03 VERDICT', title: 'Score to decision', desc: 'The fake probability becomes a verdict, a confidence value, and a risk level.' },
  { label: '04 EVIDENCE', title: 'Grad-CAM overlay', desc: 'Highlights which regions drove the decision, so the result is not a black box.' },
]

export default function ArchitectureFlow() {
  return (
    <div className="arch-section">
      <h2 className="arch-eyebrow-title">How a verdict gets made</h2>
      <div className="arch-flow">
        {NODES.map((node, i) => (
          <div key={node.title} style={{ display: 'flex', alignItems: 'center' }}>
            <div className="arch-node">
              <div className="arch-node-label mono">{node.label}</div>
              <div className="arch-node-title">{node.title}</div>
              <div className="arch-node-desc">{node.desc}</div>
            </div>
            {i < NODES.length - 1 && <div className="arch-connector" />}
          </div>
        ))}
      </div>
    </div>
  )
}