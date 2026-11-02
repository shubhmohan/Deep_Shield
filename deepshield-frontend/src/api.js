// DeepShield - API client
// Points at your deployed FastAPI backend. Set VITE_API_URL in a .env file
// (see .env.example) — falls back to localhost for local development.

const API_BASE = import.meta.env.VITE_API_URL || 'http://localhost:8000'

const IMAGE_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.webp']
const VIDEO_EXTENSIONS = ['.mp4', '.mov', '.avi', '.mkv']

export function getMediaType(filename) {
  const lower = filename.toLowerCase()
  if (IMAGE_EXTENSIONS.some((ext) => lower.endsWith(ext))) return 'image'
  if (VIDEO_EXTENSIONS.some((ext) => lower.endsWith(ext))) return 'video'
  return null
}

async function postFile(endpoint, file) {
  const formData = new FormData()
  formData.append('file', file)

  const response = await fetch(`${API_BASE}${endpoint}`, {
    method: 'POST',
    body: formData,
  })

  if (!response.ok) {
    let detail = `Request failed (${response.status})`
    try {
      const body = await response.json()
      if (body.detail) detail = body.detail
    } catch {
      // response wasn't JSON — keep the generic message
    }
    throw new Error(detail)
  }

  return response.json()
}

export function analyzeImage(file) {
  return postFile('/detect/image', file)
}

export function analyzeVideo(file) {
  return postFile('/detect/video', file)
}

export async function checkHealth() {
  try {
    const res = await fetch(`${API_BASE}/health`)
    return res.ok
  } catch {
    return false
  }
}
