import { useState } from 'react'
import UploadZone from './components/UploadZone'
import ImageResult from './components/ImageResult'
import VideoResult from './components/VideoResult'
import ArchitectureFlow from './components/ArchitectureFlow'
import { analyzeImage, analyzeVideo } from './api'

const STATUS_MESSAGES = {
  image: 'Extracting face and running the detector\u2026',
  video: 'Sampling frames and checking temporal consistency\u2026',
}

export default function App() {
  const [file, setFile] = useState(null)
  const [mediaType, setMediaType] = useState(null)
  const [previewUrl, setPreviewUrl] = useState(null)
  const [status, setStatus] = useState('idle') // idle | scanning | done | error
  const [result, setResult] = useState(null)
  const [error, setError] = useState(null)

  async function handleFileSelected(selectedFile, type) {
    setFile(selectedFile)
    setMediaType(type)
    setPreviewUrl(URL.createObjectURL(selectedFile))
    setResult(null)
    setError(null)
    setStatus('scanning')

    try {
      const data = type === 'image'
        ? await analyzeImage(selectedFile)
        : await analyzeVideo(selectedFile)
      setResult(data)
      setStatus('done')
    } catch (err) {
      setError(err.message)
      setStatus('error')
    }
  }

  function reset() {
    setFile(null)
    setMediaType(null)
    setPreviewUrl(null)
    setResult(null)
    setError(null)
    setStatus('idle')
  }

  return (
    <div className="page">
      <nav className="nav">
        <div className="nav-mark">
          <span className="nav-mark-dot" />
          DeepShield
        </div>
        <div className="nav-links">
          <a href="#how-it-works">How it works</a>
          <a href="https://github.com/" target="_blank" rel="noreferrer">GitHub</a>
        </div>
      </nav>

      <div className="hero">
        <div>
          <h1 className="hero-headline">Tell whether a face was made by a person or a model.</h1>
          <p className="hero-sub">
            DeepShield runs a deep-learning deepfake detector on any image or video
            you upload, then shows which pixels influenced the verdict — evidence,
            not just a score.
          </p>
          <div className="hero-facts">
            <div>
              <div className="hero-fact-num mono">Face-first</div>
              <div className="hero-fact-label">Crops the face before scoring</div>
            </div>
            <div>
              <div className="hero-fact-num mono">Heatmap</div>
              <div className="hero-fact-label">Shows where the model looked</div>
            </div>
            <div>
              <div className="hero-fact-num mono">100%</div>
              <div className="hero-fact-label">Runs on upload, nothing stored</div>
            </div>
          </div>
        </div>

        <div>
          {status === 'idle' && (
            <>
              <UploadZone onFileSelected={handleFileSelected} />
              <div className="trust-line">
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                  <path d="M12 2l8 4v6c0 5-3.5 8.5-8 10-4.5-1.5-8-5-8-10V6l8-4z" strokeLinejoin="round" />
                </svg>
                Your file is analyzed for this session only and is not saved afterward.
              </div>
            </>
          )}

          {(status === 'scanning' || status === 'error') && (
            <div className="preview-card">
              {mediaType === 'image' ? (
                <img className="preview-media" src={previewUrl} alt="Uploaded preview" />
              ) : (
                <video className="preview-media" src={previewUrl} muted playsInline />
              )}
              <div className="preview-filename">{file?.name}</div>

              {status === 'scanning' && (
                <div className="scan-status">
                  <span className="scan-dot" />
                  {STATUS_MESSAGES[mediaType]}
                </div>
              )}

              {status === 'error' && (
                <div className="error-box">
                  Couldn't complete analysis: {error}
                </div>
              )}

              <button className="reset-link" onClick={reset}>Try a different file</button>
            </div>
          )}

          {status === 'done' && (
            <div className="preview-card">
              {mediaType === 'image' ? (
                <img className="preview-media" src={previewUrl} alt="Uploaded preview" />
              ) : (
                <video className="preview-media" src={previewUrl} muted playsInline controls />
              )}
              <button className="reset-link" onClick={reset}>Analyze a different file</button>
            </div>
          )}
        </div>
      </div>

      {status === 'done' && result && mediaType === 'image' && (
        <ImageResult result={result} filename={file.name} />
      )}
      {status === 'done' && result && mediaType === 'video' && (
        <VideoResult result={result} filename={file.name} />
      )}

      <div id="how-it-works">
        <ArchitectureFlow />
      </div>

      <div className="footer">
        <span>DeepShield — built for detecting synthetic and manipulated faces.</span>
        <span>Not a substitute for professional forensic analysis.</span>
      </div>
    </div>
  )
}