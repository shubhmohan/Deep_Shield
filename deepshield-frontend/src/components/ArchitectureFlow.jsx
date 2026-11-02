const NODES = [
  { label: '01 INPUT', title: 'Face extraction', desc: 'MediaPipe locates and crops the subject\u2019s face from the image or sampled video frames.' },
  { label: '02 BRANCH A', title: 'Vision Transformer', desc: 'ViT-B/16 evaluates long-range patch relationships across the face.' },
  { label: '03 BRANCH B', title: 'EfficientNet-B4', desc: 'A CNN branch tuned to catch pixel-level GAN and blending artifacts.' },
  { label: '04 FUSION', title: 'Weighted ensemble', desc: 'Both branch scores combine into one verdict, confidence, and risk level.' },
  { label: '05 EVIDENCE', title: 'Grad-CAM overlay', desc: 'Highlights exactly which regions drove the decision, so nothing is a black box.' },
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
