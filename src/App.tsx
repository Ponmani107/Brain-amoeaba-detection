import React, { useCallback, useMemo, useState } from 'react'
import axios from 'axios'

const API_BASE = import.meta.env.VITE_API_BASE || 'http://localhost:8000'

type PredictResponse = {
  prediction: 'Positive' | 'Negative'
  confidence: number
  gradcam: string // base64 PNG
  summary: string
}

export default function App() {
  const [file, setFile] = useState<File | null>(null)
  const [previewUrl, setPreviewUrl] = useState<string | null>(null)
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState<PredictResponse | null>(null)
  const [error, setError] = useState<string | null>(null)

  const onDrop = useCallback((ev: React.DragEvent<HTMLDivElement>) => {
    ev.preventDefault()
    const f = ev.dataTransfer.files?.[0]
    if (f) handleFile(f)
  }, [])

  const onBrowse = useCallback((ev: React.ChangeEvent<HTMLInputElement>) => {
    const f = ev.target.files?.[0]
    if (f) handleFile(f)
  }, [])

  const handleFile = (f: File) => {
    setFile(f)
    setResult(null)
    setError(null)
    const url = URL.createObjectURL(f)
    setPreviewUrl(url)
  }

  const onPredict = async () => {
    if (!file) return
    setLoading(true)
    setError(null)
    setResult(null)
    try {
      const form = new FormData()
      form.append('file', file)
      const { data } = await axios.post<PredictResponse>(`${API_BASE}/predict`, form, {
        headers: { 'Content-Type': 'multipart/form-data' },
      })
      setResult(data)
    } catch (e: any) {
      setError(e?.response?.data?.detail || e?.message || 'Prediction failed')
    } finally {
      setLoading(false)
    }
  }

  const onDownloadReport = async () => {
    if (!result) return
    const ts = new Date().toISOString()
    const originalB64 = previewUrl ? await blobUrlToBase64(previewUrl) : undefined
    try {
      const { data } = await axios.post(`${API_BASE}/report`, {
        prediction: result.prediction,
        confidence: result.confidence,
        timestamp: ts,
        original: originalB64,
        gradcam: result.gradcam,
        summary: result.summary,
      }, { responseType: 'blob' })

      const url = window.URL.createObjectURL(new Blob([data], { type: 'application/pdf' }))
      const a = document.createElement('a')
      a.href = url
      a.download = 'brain_amoebic_report.pdf'
      a.click()
      window.URL.revokeObjectURL(url)
    } catch (e: any) {
      setError(e?.message || 'Report generation failed')
    }
  }

  const confidencePct = useMemo(() => result ? (result.confidence * 100).toFixed(1) + '%' : '-', [result])

  return (
    <div className="min-h-screen flex flex-col">
      <header className="bg-white shadow">
        <div className="max-w-5xl mx-auto px-4 py-4 flex items-center justify-between">
          <h1 className="text-xl font-semibold">Brain Amoebic Infection Detection</h1>
          <a className="text-sm text-blue-600 hover:underline" href="https://fastapi.tiangolo.com" target="_blank" rel="noreferrer">FastAPI</a>
        </div>
      </header>

      <main className="flex-1">
        <div className="max-w-5xl mx-auto p-4 grid grid-cols-1 lg:grid-cols-2 gap-6">
          <section>
            <div
              onDragOver={(e) => e.preventDefault()}
              onDrop={onDrop}
              className="border-2 border-dashed border-gray-300 rounded-lg p-6 bg-white flex flex-col items-center justify-center text-center h-64"
            >
              <div className="text-gray-500">Drag & drop MRI/CT image here</div>
              <div className="text-gray-400 text-sm mt-1">or</div>
              <label className="mt-2 cursor-pointer inline-flex items-center justify-center px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700">
                Browse
                <input type="file" accept="image/*" className="hidden" onChange={onBrowse} />
              </label>
            </div>

            {previewUrl && (
              <div className="mt-4">
                <div className="text-sm font-medium mb-2">Uploaded Image Preview</div>
                <img src={previewUrl} alt="preview" className="w-full max-h-96 object-contain rounded-md border" />
              </div>
            )}

            <div className="mt-4 flex gap-3">
              <button
                onClick={onPredict}
                disabled={!file || loading}
                className="px-4 py-2 rounded-md bg-green-600 text-white disabled:opacity-50 hover:bg-green-700"
              >
                {loading ? 'Analyzing…' : 'Analyze Image'}
              </button>
              {result && (
                <button
                  onClick={onDownloadReport}
                  className="px-4 py-2 rounded-md bg-indigo-600 text-white hover:bg-indigo-700"
                >
                  Download Report
                </button>
              )}
            </div>

            {error && (
              <div className="mt-3 text-sm text-red-600">{error}</div>
            )}
          </section>

          <section>
            <div className="bg-white rounded-lg shadow p-4 min-h-[16rem]">
              <div className="text-lg font-semibold mb-3">Results</div>
              {!result && (
                <div className="text-gray-500">No results yet. Upload an image and click Analyze.</div>
              )}
              {result && (
                <div className="space-y-3">
                  <div className="flex items-center gap-3">
                    <span className={`inline-flex px-2 py-1 rounded text-white text-sm ${result.prediction === 'Positive' ? 'bg-red-600' : 'bg-emerald-600'}`}>
                      {result.prediction}
                    </span>
                    <span className="text-sm text-gray-600">Confidence: {confidencePct}</span>
                  </div>
                  <div>
                    <div className="text-sm font-medium mb-2">Grad-CAM Heatmap Overlay</div>
                    <img
                      src={`data:image/png;base64,${result.gradcam}`}
                      alt="gradcam"
                      className="w-full max-h-96 object-contain rounded-md border"
                    />
                  </div>
                  <div className="text-sm text-gray-700">
                    {result.summary}
                  </div>
                </div>
              )}
            </div>
          </section>
        </div>
      </main>

      <footer className="py-6 text-center text-xs text-gray-500">© {new Date().getFullYear()} Brain Amoebic Detection</footer>
    </div>
  )
}

async function blobUrlToBase64(url: string): Promise<string> {
  const res = await fetch(url)
  const blob = await res.blob()
  return await blobToBase64(blob)
}

function blobToBase64(blob: Blob): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader()
    reader.onloadend = () => resolve((reader.result as string).split(',')[1])
    reader.onerror = reject
    reader.readAsDataURL(blob)
  })
}
