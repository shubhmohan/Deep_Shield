import { useState, useRef } from 'react'
import { getMediaType } from '../api'

export default function UploadZone({ onFileSelected }) {
  const [dragging, setDragging] = useState(false)
  const inputRef = useRef(null)

  function handleFiles(files) {
    const file = files[0]
    if (!file) return
    const mediaType = getMediaType(file.name)
    if (!mediaType) {
      alert('Unsupported file type. Upload a JPG, PNG, WEBP image or an MP4, MOV, AVI, MKV video.')
      return
    }
    onFileSelected(file, mediaType)
  }

  return (
    <div
      className={`dropzone ${dragging ? 'dragging' : ''}`}
      onClick={() => inputRef.current?.click()}
      onDragOver={(e) => { e.preventDefault(); setDragging(true) }}
      onDragLeave={() => setDragging(false)}
      onDrop={(e) => {
        e.preventDefault()
        setDragging(false)
        handleFiles(e.dataTransfer.files)
      }}
      role="button"
      tabIndex={0}
      onKeyDown={(e) => { if (e.key === 'Enter' || e.key === ' ') inputRef.current?.click() }}
    >
      <svg className="dropzone-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
        <path d="M12 16V4M12 4l-4 4M12 4l4 4" strokeLinecap="round" strokeLinejoin="round" />
        <path d="M4 16v2a2 2 0 002 2h12a2 2 0 002-2v-2" strokeLinecap="round" strokeLinejoin="round" />
      </svg>
      <div className="dropzone-title">Drop an image or video, or click to browse</div>
      <div className="dropzone-sub">JPG, PNG, WEBP, MP4, MOV — analyzed in your session, not stored</div>
      <input
        ref={inputRef}
        type="file"
        accept=".jpg,.jpeg,.png,.webp,.mp4,.mov,.avi,.mkv"
        onChange={(e) => handleFiles(e.target.files)}
      />
    </div>
  )
}
