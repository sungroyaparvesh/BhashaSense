import { useState, useRef, useEffect } from 'react'
import axios from 'axios'
import {
  UploadCloud,
  FileAudio,
  Loader2,
  CheckCircle2,
  AlertCircle,
  Tag,
  TrendingDown,
  TrendingUp,
  MinusCircle,
  Layers,
  Sparkles,
  RefreshCw,
  Clock,
  Cpu,
  Volume2,
  Mic,
  Square,
  Radio,
  Play,
  Pause,
  Download,
  Search,
  Filter,
  FileText
} from 'lucide-react'

export default function App() {
  const [activeTab, setActiveTab] = useState('record') // 'record' | 'upload'
  const [file, setFile] = useState(null)
  const [audioUrl, setAudioUrl] = useState(null)
  const [isLoading, setIsLoading] = useState(false)
  const [loadingStage, setLoadingStage] = useState('')
  const [result, setResult] = useState(null)
  const [error, setError] = useState(null)
  const [isDragOver, setIsDragOver] = useState(false)

  // Live microphone recording states
  const [isRecording, setIsRecording] = useState(false)
  const [recordingDuration, setRecordingDuration] = useState(0)
  const [permissionError, setPermissionError] = useState(null)

  // Audio playback & synchronization states
  const [isPlaying, setIsPlaying] = useState(false)
  const [currentTime, setCurrentTime] = useState(0)
  const [audioDuration, setAudioDuration] = useState(0)

  // Filter & Search states
  const [searchQuery, setSearchQuery] = useState('')
  const [selectedEntityCategory, setSelectedEntityCategory] = useState('ALL')

  const fileInputRef = useRef(null)
  const mediaRecorderRef = useRef(null)
  const audioChunksRef = useRef([])
  const mediaStreamRef = useRef(null)
  const timerIntervalRef = useRef(null)
  const audioRef = useRef(null)

  // Keep audioUrl in sync with the current file
  useEffect(() => {
    if (file) {
      const url = URL.createObjectURL(file)
      setAudioUrl(url)
      setIsPlaying(false)
      setCurrentTime(0)
      return () => URL.revokeObjectURL(url)
    } else {
      setAudioUrl(null)
    }
  }, [file])

  // Cleanup media stream and timer on unmount
  useEffect(() => {
    return () => {
      if (timerIntervalRef.current) clearInterval(timerIntervalRef.current)
      if (mediaStreamRef.current) {
        mediaStreamRef.current.getTracks().forEach((track) => track.stop())
      }
    }
  }, [])

  // Format seconds to mm:ss
  const formatDuration = (seconds) => {
    const mins = Math.floor(seconds / 60)
    const secs = Math.floor(seconds % 60)
    return `${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`
  }

  const ALLOWED_MIME_TYPES = [
    'audio/wav',
    'audio/mpeg',
    'audio/aac',
    'audio/m4a',
    'audio/x-m4a',
    'audio/mp4',
    'audio/ogg',
    'audio/webm',
    'audio/flac',
  ]
  const ALLOWED_EXTENSIONS = ['.wav', '.mp3', '.aac', '.m4a', '.ogg', '.webm', '.flac']

  const validateAudioFile = (selectedFile) => {
    if (!selectedFile) return false
    const name = (selectedFile.name || '').toLowerCase()
    const hasValidExt = ALLOWED_EXTENSIONS.some((ext) => name.endsWith(ext))
    const hasValidMime =
      !selectedFile.type ||
      ALLOWED_MIME_TYPES.includes(selectedFile.type.toLowerCase()) ||
      selectedFile.type.startsWith('audio/')
    return hasValidExt || hasValidMime
  }

  // Handle file selection via browse
  const handleFileChange = (e) => {
    const selected = e.target.files?.[0]
    if (selected) {
      if (!validateAudioFile(selected)) {
        setError('Unsupported audio file format. Please upload a .wav, .mp3, .aac, .m4a, .ogg, or .webm file.')
        return
      }
      setFile(selected)
      setError(null)
      setResult(null)
      setPermissionError(null)
    }
  }

  // Handle drag and drop
  const handleDrop = (e) => {
    e.preventDefault()
    setIsDragOver(false)
    const dropped = e.dataTransfer.files?.[0]
    if (dropped) {
      if (!validateAudioFile(dropped)) {
        setError('Unsupported audio file format. Please upload a .wav, .mp3, .aac, .m4a, .ogg, or .webm file.')
        return
      }
      setFile(dropped)
      setError(null)
      setResult(null)
      setPermissionError(null)
    }
  }

  // Generate synthetic test audio tone (WAV) directly in browser for instant 1-click test
  const handleCreateSampleAudio = () => {
    const sampleRate = 16000
    const duration = 2.0
    const numSamples = Math.floor(sampleRate * duration)
    const buffer = new ArrayBuffer(44 + numSamples * 2)
    const view = new DataView(buffer)

    const writeString = (offset, string) => {
      for (let i = 0; i < string.length; i++) {
        view.setUint8(offset + i, string.charCodeAt(i))
      }
    }

    writeString(0, 'RIFF')
    view.setUint32(4, 36 + numSamples * 2, true)
    writeString(8, 'WAVE')
    writeString(12, 'fmt ')
    view.setUint32(16, 16, true)
    view.setUint16(20, 1, true)
    view.setUint16(22, 1, true)
    view.setUint32(24, sampleRate, true)
    view.setUint32(28, sampleRate * 2, true)
    view.setUint16(32, 2, true)
    view.setUint16(34, 16, true)
    writeString(36, 'data')
    view.setUint32(40, numSamples * 2, true)

    for (let i = 0; i < numSamples; i++) {
      const sample = Math.sin((2 * Math.PI * 440 * i) / sampleRate) * 0.4
      const intSample = Math.max(-32768, Math.min(32767, Math.floor(sample * 32767)))
      view.setInt16(44 + i * 2, intSample, true)
    }

    const sampleBlob = new Blob([buffer], { type: 'audio/wav' })
    const sampleFile = new File([sampleBlob], 'sample_call_audio.wav', { type: 'audio/wav' })
    setFile(sampleFile)
    setError(null)
    setResult(null)
    setPermissionError(null)
  }

  // Execute pipeline analysis against http://localhost:8000/analyze
  const triggerAnalyze = async (fileToAnalyze) => {
    const targetFile = fileToAnalyze || file
    if (!targetFile) {
      setError('Please select or record an audio file first.')
      return
    }

    setIsLoading(true)
    setError(null)
    setResult(null)
    setLoadingStage('Transcribing audio via OpenAI Whisper...')

    const timer1 = setTimeout(() => {
      setLoadingStage('Concurrently evaluating RoBERTa Sentiment & BERT NER...')
    }, 450)

    try {
      const formData = new FormData()
      formData.append('file', targetFile)

      const response = await axios.post('http://localhost:8000/analyze', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      })

      setResult(response.data)
    } catch (err) {
      console.error('API Error:', err)
      const message =
        err.response?.data?.detail ||
        err.response?.data?.message ||
        err.message ||
        'Failed to connect to http://localhost:8000/analyze. Is the backend server running?'
      setError(message)
    } finally {
      clearTimeout(timer1)
      setIsLoading(false)
      setLoadingStage('')
    }
  }

  // --- Live Microphone Recording Handlers ---
  const handleStartRecording = async () => {
    setPermissionError(null)
    setError(null)
    setResult(null)

    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      setPermissionError('Microphone recording is not supported in this browser environment.')
      return
    }

    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true })
      mediaStreamRef.current = stream

      let mimeType = ''
      if (typeof MediaRecorder !== 'undefined') {
        if (MediaRecorder.isTypeSupported('audio/webm;codecs=opus')) {
          mimeType = 'audio/webm;codecs=opus'
        } else if (MediaRecorder.isTypeSupported('audio/webm')) {
          mimeType = 'audio/webm'
        } else if (MediaRecorder.isTypeSupported('audio/ogg;codecs=opus')) {
          mimeType = 'audio/ogg;codecs=opus'
        }
      }

      const options = mimeType ? { mimeType } : undefined
      const mediaRecorder = new MediaRecorder(stream, options)
      mediaRecorderRef.current = mediaRecorder
      audioChunksRef.current = []

      mediaRecorder.ondataavailable = (event) => {
        if (event.data && event.data.size > 0) {
          audioChunksRef.current.push(event.data)
        }
      }

      mediaRecorder.onstop = async () => {
        if (mediaStreamRef.current) {
          mediaStreamRef.current.getTracks().forEach((track) => track.stop())
          mediaStreamRef.current = null
        }

        const blobType = mediaRecorder.mimeType || 'audio/wav'
        const audioBlob = new Blob(audioChunksRef.current, { type: blobType })
        const liveFile = new File([audioBlob], 'live_audio.wav', { type: 'audio/wav' })
        setFile(liveFile)

        await triggerAnalyze(liveFile)
      }

      mediaRecorder.start(250)
      setIsRecording(true)
      setRecordingDuration(0)

      timerIntervalRef.current = setInterval(() => {
        setRecordingDuration((prev) => prev + 1)
      }, 1000)
    } catch (err) {
      console.error('Microphone access error:', err)
      let friendlyMessage = 'Unable to access microphone.'

      if (err.name === 'NotAllowedError' || err.name === 'PermissionDeniedError') {
        friendlyMessage = 'Microphone permission was denied. Please allow microphone access in your browser address bar/settings.'
      } else if (err.name === 'NotFoundError' || err.name === 'DevicesNotFoundError') {
        friendlyMessage = 'No microphone device was detected on your system. Please connect an audio input device.'
      } else if (err.name === 'NotReadableError' || err.name === 'TrackStartError') {
        friendlyMessage = 'Your microphone is already in use by another application or tab.'
      } else if (err.message) {
        friendlyMessage = err.message
      }

      setPermissionError(friendlyMessage)
      setIsRecording(false)
    }
  }

  const handleStopRecording = () => {
    if (timerIntervalRef.current) {
      clearInterval(timerIntervalRef.current)
      timerIntervalRef.current = null
    }

    if (mediaRecorderRef.current && mediaRecorderRef.current.state !== 'inactive') {
      mediaRecorderRef.current.stop()
    }
    setIsRecording(false)
  }

  // --- Audio Player Controls ---
  const togglePlayAudio = () => {
    if (!audioRef.current) return
    if (isPlaying) {
      audioRef.current.pause()
      setIsPlaying(false)
    } else {
      audioRef.current.play()
      setIsPlaying(true)
    }
  }

  const handleSeek = (timeInSeconds) => {
    if (audioRef.current) {
      audioRef.current.currentTime = timeInSeconds
      setCurrentTime(timeInSeconds)
      if (!isPlaying) {
        audioRef.current.play()
        setIsPlaying(true)
      }
    }
  }

  // --- Export Report Handlers ---
  const handleExportJSON = () => {
    if (!result) return
    const jsonStr = JSON.stringify(result, null, 2)
    const blob = new Blob([jsonStr], { type: 'application/json' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `bhashasense_${result.audio?.filename || 'report'}.json`
    document.body.appendChild(a)
    a.click()
    a.remove()
    URL.revokeObjectURL(url)
  }

  const handleExportCSV = () => {
    if (!result?.transcription?.segments) return
    const rows = [
      ['segment_id', 'start_time_sec', 'end_time_sec', 'confidence', 'text']
    ]
    result.transcription.segments.forEach((seg) => {
      rows.push([
        seg.id,
        seg.start_time,
        seg.end_time,
        seg.confidence,
        `"${(seg.text || '').replace(/"/g, '""')}"`
      ])
    })
    const csvContent = rows.map((r) => r.join(',')).join('\n')
    const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `bhashasense_segments_${result.audio?.filename || 'segments'}.csv`
    document.body.appendChild(a)
    a.click()
    a.remove()
    URL.revokeObjectURL(url)
  }

  // Sentiment styling helper
  const getSentimentTheme = (sentimentStr = '') => {
    const s = sentimentStr.toLowerCase()
    if (s.includes('neg')) {
      return {
        bg: 'bg-rose-500/10',
        border: 'border-rose-500/30',
        badgeBg: 'bg-rose-500/20',
        text: 'text-rose-400',
        bar: 'bg-rose-500',
        icon: TrendingDown,
        label: 'Negative',
      }
    }
    if (s.includes('pos')) {
      return {
        bg: 'bg-emerald-500/10',
        border: 'border-emerald-500/30',
        badgeBg: 'bg-emerald-500/20',
        text: 'text-emerald-400',
        bar: 'bg-emerald-500',
        icon: TrendingUp,
        label: 'Positive',
      }
    }
    return {
      bg: 'bg-amber-500/10',
      border: 'border-amber-500/30',
      badgeBg: 'bg-amber-500/20',
      text: 'text-amber-400',
      bar: 'bg-amber-500',
      icon: MinusCircle,
      label: 'Neutral',
    }
  }

  // Entity label color badge helper
  const getEntityBadgeStyle = (label = '') => {
    switch (label.toUpperCase()) {
      case 'ORGANIZATION':
        return 'bg-blue-500/20 text-blue-300 border-blue-500/30'
      case 'MONEY':
        return 'bg-emerald-500/20 text-emerald-300 border-emerald-500/30'
      case 'APPLICATION_ID':
        return 'bg-purple-500/20 text-purple-300 border-purple-500/30'
      case 'LOCATION':
        return 'bg-cyan-500/20 text-cyan-300 border-cyan-500/30'
      case 'PERSON':
        return 'bg-violet-500/20 text-violet-300 border-violet-500/30'
      case 'DATE_TIME':
        return 'bg-amber-500/20 text-amber-300 border-amber-500/30'
      default:
        return 'bg-slate-700/50 text-slate-300 border-slate-600/40'
    }
  }

  // Render transcript with highlighted entities
  const renderHighlightedTranscript = (transcriptText, entitiesList = []) => {
    if (!entitiesList || entitiesList.length === 0) {
      return <span>{transcriptText}</span>
    }

    const sorted = [...entitiesList].sort((a, b) => a.start_char - b.start_char)
    const elements = []
    let cursor = 0

    sorted.forEach((ent, index) => {
      if (ent.start_char > cursor) {
        elements.push(
          <span key={`text-${cursor}`}>
            {transcriptText.slice(cursor, ent.start_char)}
          </span>
        )
      }

      const badgeStyle = getEntityBadgeStyle(ent.label)
      elements.push(
        <mark
          key={`ent-${index}`}
          className={`inline-flex items-center gap-1 mx-1 px-2 py-0.5 rounded-md font-semibold text-xs border ${badgeStyle} bg-opacity-70`}
          title={`${ent.label} (${Math.round(ent.confidence * 100)}% confidence)`}
        >
          <span>{ent.entity}</span>
          <span className="text-[10px] uppercase tracking-wider opacity-75 font-mono">
            {ent.label}
          </span>
        </mark>
      )

      cursor = Math.max(cursor, ent.end_char)
    })

    if (cursor < transcriptText.length) {
      elements.push(
        <span key="text-tail">
          {transcriptText.slice(cursor)}
        </span>
      )
    }

    return elements
  }

  // Filter entities
  const filteredEntities = (result?.entity_extraction?.entities || []).filter((ent) => {
    if (selectedEntityCategory === 'ALL') return true
    return ent.label.toUpperCase() === selectedEntityCategory.toUpperCase()
  })

  // Filter segments
  const filteredSegments = (result?.transcription?.segments || []).filter((seg) => {
    if (!searchQuery.trim()) return true
    return seg.text.toLowerCase().includes(searchQuery.toLowerCase())
  })

  const sentimentTheme = result ? getSentimentTheme(result.sentiment_analysis?.overall_sentiment) : null
  const SentimentIcon = sentimentTheme ? sentimentTheme.icon : null

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 flex flex-col items-center py-10 px-4 sm:px-6 lg:px-8">
      {/* Hidden HTML5 Audio Element for Synchronized Playback */}
      {audioUrl && (
        <audio
          ref={audioRef}
          src={audioUrl}
          onTimeUpdate={() => {
            if (audioRef.current) setCurrentTime(audioRef.current.currentTime)
          }}
          onLoadedMetadata={() => {
            if (audioRef.current) setAudioDuration(audioRef.current.duration || 0)
          }}
          onEnded={() => setIsPlaying(false)}
        />
      )}

      {/* Top Header */}
      <header className="max-w-5xl w-full text-center mb-8">
        <div className="inline-flex items-center gap-2 px-3.5 py-1.5 rounded-full bg-indigo-500/10 border border-indigo-500/20 text-indigo-400 text-xs font-medium mb-4">
          <Sparkles className="w-3.5 h-3.5" />
          <span>Multilingual Code-Mixed Audio AI</span>
        </div>
        <h1 className="text-4xl sm:text-5xl font-extrabold tracking-tight bg-gradient-to-r from-white via-slate-200 to-indigo-300 bg-clip-text text-transparent">
          BhashaSense Dashboard
        </h1>
        <p className="mt-3 text-slate-400 text-base max-w-2xl mx-auto">
          End-to-end speech transcription (Whisper), sentiment intelligence (RoBERTa), and named entity extraction (BERT NER).
        </p>
      </header>

      {/* Main Container */}
      <main className="max-w-5xl w-full space-y-8">
        {/* Audio Input Card */}
        <section className="bg-slate-900/90 border border-slate-800 rounded-3xl p-6 sm:p-8 shadow-2xl backdrop-blur-xl">
          {/* Tabs: Live Record vs Upload */}
          <div className="flex items-center justify-between pb-6 border-b border-slate-800 mb-6 flex-wrap gap-4">
            <h2 className="text-lg font-semibold text-white flex items-center gap-2">
              <Volume2 className="w-5 h-5 text-indigo-400" />
              Audio Input Mode
            </h2>

            <div className="flex items-center p-1 bg-slate-950/70 border border-slate-800 rounded-xl">
              <button
                type="button"
                onClick={() => setActiveTab('record')}
                className={`flex items-center gap-2 px-4 py-2 rounded-lg text-xs font-semibold transition-all ${
                  activeTab === 'record'
                    ? 'bg-rose-500 text-white shadow-md shadow-rose-500/30'
                    : 'text-slate-400 hover:text-white'
                }`}
              >
                <Mic className="w-3.5 h-3.5" />
                <span>Live Microphone</span>
                {isRecording && (
                  <span className="w-2 h-2 rounded-full bg-white animate-ping" />
                )}
              </button>

              <button
                type="button"
                onClick={() => setActiveTab('upload')}
                className={`flex items-center gap-2 px-4 py-2 rounded-lg text-xs font-semibold transition-all ${
                  activeTab === 'upload'
                    ? 'bg-indigo-600 text-white shadow-md shadow-indigo-600/30'
                    : 'text-slate-400 hover:text-white'
                }`}
              >
                <UploadCloud className="w-3.5 h-3.5" />
                <span>Upload File</span>
              </button>
            </div>
          </div>

          {/* TAB 1: LIVE MICROPHONE RECORDING */}
          {activeTab === 'record' && (
            <div className="flex flex-col items-center justify-center p-8 sm:p-12 border border-slate-800/80 rounded-2xl bg-slate-950/40 text-center relative overflow-hidden">
              {isRecording && (
                <div className="absolute inset-0 bg-rose-500/5 animate-pulse pointer-events-none" />
              )}

              {/* Pulsing Red Indicator */}
              {isRecording && (
                <div className="mb-6 inline-flex items-center gap-2 px-4 py-2 rounded-full bg-rose-500/15 border border-rose-500/30 text-rose-400 font-semibold text-sm shadow-lg shadow-rose-500/10">
                  <span className="relative flex h-3 w-3">
                    <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-rose-400 opacity-75" />
                    <span className="relative inline-flex rounded-full h-3 w-3 bg-rose-500" />
                  </span>
                  <span className="animate-pulse">Recording...</span>
                  <span className="font-mono bg-rose-950/60 px-2 py-0.5 rounded text-xs text-rose-300 border border-rose-500/30">
                    {formatDuration(recordingDuration)}
                  </span>
                </div>
              )}

              {/* Orb */}
              <div
                className={`w-24 h-24 rounded-full flex items-center justify-center mb-6 transition-all duration-300 ${
                  isRecording
                    ? 'bg-rose-500 text-white shadow-2xl shadow-rose-500/40 scale-110 ring-4 ring-rose-500/30 animate-pulse'
                    : 'bg-slate-800 text-slate-300 border border-slate-700 shadow-inner'
                }`}
              >
                <Mic className="w-10 h-10" />
              </div>

              <p className="text-base font-medium text-white mb-1">
                {isRecording
                  ? 'Capturing live speech stream via browser MediaRecorder'
                  : 'Record speech directly from your microphone'}
              </p>
              <p className="text-xs text-slate-400 max-w-md mb-8">
                {isRecording
                  ? 'Speak clearly in English, Hindi, or Hinglish. Press Stop when finished to automatically run extraction.'
                  : 'Click Start Recording to begin. When stopped, audio will be packaged as live_audio.wav and analyzed.'}
              </p>

              {isRecording && (
                <div className="flex items-center gap-1.5 h-8 mb-8">
                  {[40, 75, 55, 90, 60, 30, 85, 45, 95, 65, 35, 80, 50].map((h, i) => (
                    <span
                      key={i}
                      style={{ height: `${h}%` }}
                      className="w-1 bg-rose-500 rounded-full animate-pulse"
                    />
                  ))}
                </div>
              )}

              <div className="flex items-center gap-4">
                {!isRecording ? (
                  <button
                    type="button"
                    onClick={handleStartRecording}
                    disabled={isLoading}
                    className="px-8 py-3.5 rounded-xl font-semibold text-sm bg-rose-600 hover:bg-rose-500 text-white shadow-lg shadow-rose-600/30 hover:shadow-rose-600/50 transition-all hover:scale-105 active:scale-95 disabled:opacity-50 flex items-center gap-2.5"
                  >
                    <Radio className="w-4 h-4 animate-pulse" />
                    <span>Start Recording</span>
                  </button>
                ) : (
                  <button
                    type="button"
                    onClick={handleStopRecording}
                    className="px-8 py-3.5 rounded-xl font-semibold text-sm bg-rose-600 hover:bg-rose-700 text-white shadow-xl shadow-rose-600/40 hover:scale-105 active:scale-95 transition-all flex items-center gap-2.5"
                  >
                    <Square className="w-4 h-4 fill-white" />
                    <span>Stop Recording</span>
                  </button>
                )}
              </div>

              {permissionError && (
                <div className="mt-6 p-4 rounded-2xl bg-rose-500/10 border border-rose-500/20 text-rose-300 text-xs flex items-center gap-3 text-left max-w-lg">
                  <AlertCircle className="w-5 h-5 text-rose-400 shrink-0" />
                  <div>
                    <p className="font-semibold">Microphone Permission Error</p>
                    <p className="opacity-90 mt-0.5">{permissionError}</p>
                  </div>
                </div>
              )}
            </div>
          )}

          {/* TAB 2: FILE UPLOAD */}
          {activeTab === 'upload' && (
            <div>
              <div
                onDragOver={(e) => {
                  e.preventDefault()
                  setIsDragOver(true)
                }}
                onDragLeave={() => setIsDragOver(false)}
                onDrop={handleDrop}
                onClick={() => fileInputRef.current?.click()}
                className={`border-2 border-dashed rounded-2xl p-8 sm:p-10 text-center cursor-pointer transition-all duration-200 flex flex-col items-center justify-center ${
                  isDragOver
                    ? 'border-indigo-400 bg-indigo-500/10 scale-[1.01]'
                    : file
                    ? 'border-indigo-500/50 bg-slate-800/40'
                    : 'border-slate-700/80 bg-slate-950/40 hover:border-slate-600 hover:bg-slate-900/50'
                }`}
              >
                <input
                  ref={fileInputRef}
                  type="file"
                  accept="audio/*,.wav,.mp3,.aac,.m4a,.ogg,.webm"
                  onChange={handleFileChange}
                  className="hidden"
                />

                {file ? (
                  <div className="flex flex-col items-center">
                    <div className="p-4 rounded-2xl bg-indigo-500/20 text-indigo-400 mb-3 border border-indigo-500/30">
                      <FileAudio className="w-10 h-10" />
                    </div>
                    <p className="text-sm font-semibold text-white truncate max-w-xs sm:max-w-md">
                      {file.name}
                    </p>
                    <p className="text-xs text-slate-400 mt-1">
                      {(file.size / (1024 * 1024)).toFixed(2)} MB • {file.type || 'audio/wav'}
                    </p>
                    <span className="text-xs text-indigo-400 underline mt-2">
                      Click or drop to replace
                    </span>
                  </div>
                ) : (
                  <div className="flex flex-col items-center">
                    <div className="p-4 rounded-2xl bg-slate-800 text-slate-400 mb-3 border border-slate-700">
                      <UploadCloud className="w-10 h-10" />
                    </div>
                    <p className="text-sm font-medium text-slate-200">
                      Drag and drop your audio file here, or{' '}
                      <span className="text-indigo-400 font-semibold underline">browse</span>
                    </p>
                    <p className="text-xs text-slate-500 mt-1">
                      Supports WAV, MP3, AAC, M4A, OGG, WEBM (Max 50MB)
                    </p>
                  </div>
                )}
              </div>

              {/* Upload Actions Row */}
              <div className="mt-6 flex flex-col sm:flex-row items-center justify-between gap-4">
                <button
                  type="button"
                  onClick={handleCreateSampleAudio}
                  className="text-xs text-slate-400 hover:text-indigo-300 flex items-center gap-1.5 transition-colors"
                >
                  <Sparkles className="w-3.5 h-3.5 text-indigo-400" />
                  <span>Use sample test audio (Auto-generate WAV)</span>
                </button>

                <button
                  type="button"
                  onClick={() => triggerAnalyze(file)}
                  disabled={!file || isLoading}
                  className={`w-full sm:w-auto px-8 py-3.5 rounded-xl font-semibold text-sm flex items-center justify-center gap-2.5 shadow-lg transition-all duration-200 ${
                    isLoading
                      ? 'bg-indigo-600/60 text-indigo-200 cursor-not-allowed'
                      : !file
                      ? 'bg-slate-800 text-slate-500 cursor-not-allowed border border-slate-700/50'
                      : 'bg-indigo-600 hover:bg-indigo-500 text-white shadow-indigo-600/30 hover:shadow-indigo-600/50 hover:scale-[1.02] active:scale-[0.98]'
                  }`}
                >
                  {isLoading ? (
                    <>
                      <Loader2 className="w-4 h-4 animate-spin text-white" />
                      <span>Analyzing Audio...</span>
                    </>
                  ) : (
                    <>
                      <Layers className="w-4 h-4" />
                      <span>Analyze Audio</span>
                    </>
                  )}
                </button>
              </div>
            </div>
          )}

          {/* Synchronized Audio Player Card (Active when file is loaded) */}
          {file && (
            <div className="mt-6 p-4 bg-slate-950/70 border border-slate-800 rounded-2xl flex flex-col gap-3">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2.5 truncate">
                  <button
                    type="button"
                    onClick={togglePlayAudio}
                    className="p-2.5 rounded-xl bg-indigo-600 hover:bg-indigo-500 text-white shadow-md shadow-indigo-600/30 transition-all hover:scale-105 active:scale-95 shrink-0"
                    title={isPlaying ? 'Pause' : 'Play audio'}
                  >
                    {isPlaying ? (
                      <Pause className="w-4 h-4 fill-white" />
                    ) : (
                      <Play className="w-4 h-4 fill-white ml-0.5" />
                    )}
                  </button>
                  <div className="truncate">
                    <p className="text-xs font-semibold text-white truncate">{file.name}</p>
                    <p className="text-[11px] text-slate-400 font-mono">
                      {formatDuration(currentTime)} / {formatDuration(audioDuration || 0)}
                    </p>
                  </div>
                </div>

                {!isLoading && (
                  <button
                    type="button"
                    onClick={() => triggerAnalyze(file)}
                    className="px-3 py-1.5 bg-slate-800 hover:bg-slate-700 text-slate-200 rounded-lg font-medium text-xs transition-colors shrink-0 flex items-center gap-1.5"
                  >
                    <RefreshCw className="w-3 h-3 text-indigo-400" />
                    <span>Re-Analyze</span>
                  </button>
                )}
              </div>

              {/* Audio scrubber */}
              <div className="flex items-center gap-3">
                <input
                  type="range"
                  min="0"
                  max={audioDuration || 1}
                  step="0.05"
                  value={currentTime}
                  onChange={(e) => handleSeek(parseFloat(e.target.value))}
                  className="w-full h-1.5 bg-slate-800 rounded-lg appearance-none cursor-pointer accent-indigo-500"
                />
              </div>
            </div>
          )}

          {/* Loading Indicator */}
          {isLoading && (
            <div className="mt-6 p-4 rounded-2xl bg-indigo-950/40 border border-indigo-500/20 flex items-center gap-3 animate-pulse">
              <Loader2 className="w-5 h-5 text-indigo-400 animate-spin shrink-0" />
              <div>
                <p className="text-xs font-semibold text-indigo-200">
                  Executing Pipeline Orchestration
                </p>
                <p className="text-[11px] text-indigo-400/90">{loadingStage}</p>
              </div>
            </div>
          )}

          {/* Error Banner */}
          {error && (
            <div className="mt-6 p-4 rounded-2xl bg-rose-500/10 border border-rose-500/20 text-rose-300 text-xs flex items-center gap-3">
              <AlertCircle className="w-5 h-5 text-rose-400 shrink-0" />
              <div className="flex-1">
                <p className="font-semibold">Analysis Failed</p>
                <p className="opacity-90">{error}</p>
              </div>
            </div>
          )}
        </section>

        {/* RESULTS SECTION */}
        {result && (
          <section className="space-y-6">
            {/* Top Toolbar: Metadata & Export Actions */}
            <div className="flex flex-wrap items-center justify-between gap-4 px-5 py-3.5 bg-slate-900/80 border border-slate-800 rounded-2xl text-xs text-slate-400 shadow-lg">
              <div className="flex items-center gap-3">
                <CheckCircle2 className="w-4 h-4 text-emerald-400 shrink-0" />
                <span className="font-semibold text-slate-200">Inference Complete</span>
                <span className="text-slate-600">•</span>
                <span className="text-indigo-300 font-mono">{result.audio?.filename}</span>
                <span className="text-slate-600">•</span>
                <span className="px-2 py-0.5 rounded bg-indigo-500/20 text-indigo-300 border border-indigo-500/30 font-medium">
                  {result.audio?.detected_language}
                </span>
              </div>

              {/* Export Buttons */}
              <div className="flex items-center gap-2">
                <button
                  type="button"
                  onClick={handleExportJSON}
                  className="px-3 py-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-200 border border-slate-700 font-medium flex items-center gap-1.5 transition-colors"
                  title="Download full analysis payload as JSON"
                >
                  <Download className="w-3.5 h-3.5 text-indigo-400" />
                  <span>Export JSON</span>
                </button>

                <button
                  type="button"
                  onClick={handleExportCSV}
                  className="px-3 py-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-200 border border-slate-700 font-medium flex items-center gap-1.5 transition-colors"
                  title="Download transcript segments as CSV"
                >
                  <FileText className="w-3.5 h-3.5 text-emerald-400" />
                  <span>Export CSV</span>
                </button>
              </div>
            </div>

            {/* Grid: Sentiment & Transcript */}
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
              {/* Sentiment Card (Color-Coded: Red for Negative, Green for Positive) */}
              <div
                className={`lg:col-span-1 rounded-3xl p-6 border shadow-xl flex flex-col justify-between ${sentimentTheme?.bg} ${sentimentTheme?.border}`}
              >
                <div>
                  <div className="flex items-center justify-between mb-4">
                    <h3 className="text-sm font-semibold tracking-wide uppercase text-slate-300 flex items-center gap-2">
                      <SentimentIcon className={`w-4 h-4 ${sentimentTheme?.text}`} />
                      Sentiment Analysis
                    </h3>
                    <span
                      className={`px-3 py-1 rounded-full text-xs font-bold border ${sentimentTheme?.badgeBg} ${sentimentTheme?.text} ${sentimentTheme?.border}`}
                    >
                      {result.sentiment_analysis?.sentiment_label || sentimentTheme?.label}
                    </span>
                  </div>

                  {/* Primary Score Display */}
                  <div className="my-6 text-center">
                    <div
                      className={`text-5xl font-extrabold tracking-tight ${sentimentTheme?.text}`}
                    >
                      {Math.round((result.sentiment_analysis?.confidence_score || 0) * 100)}%
                    </div>
                    <div className="text-xs uppercase font-semibold tracking-wider text-slate-400 mt-1">
                      {result.sentiment_analysis?.overall_sentiment} Polarity
                    </div>
                  </div>

                  {/* Distribution Bar */}
                  {result.sentiment_analysis?.distribution && (
                    <div className="space-y-2 mt-4 pt-4 border-t border-slate-700/40">
                      <div className="flex justify-between text-xs text-slate-400">
                        <span>Distribution</span>
                        <span>
                          Neg: {Math.round(result.sentiment_analysis.distribution.negative * 100)}% | Pos: {Math.round(result.sentiment_analysis.distribution.positive * 100)}%
                        </span>
                      </div>
                      <div className="h-2.5 w-full bg-slate-800 rounded-full overflow-hidden flex">
                        <div
                          style={{
                            width: `${(result.sentiment_analysis.distribution.negative || 0) * 100}%`,
                          }}
                          className="bg-rose-500 h-full"
                          title="Negative"
                        />
                        <div
                          style={{
                            width: `${(result.sentiment_analysis.distribution.neutral || 0) * 100}%`,
                          }}
                          className="bg-amber-400 h-full"
                          title="Neutral"
                        />
                        <div
                          style={{
                            width: `${(result.sentiment_analysis.distribution.positive || 0) * 100}%`,
                          }}
                          className="bg-emerald-500 h-full"
                          title="Positive"
                        />
                      </div>
                    </div>
                  )}

                  {/* Aspect Sentiments */}
                  {result.sentiment_analysis?.aspect_sentiments && result.sentiment_analysis.aspect_sentiments.length > 0 && (
                    <div className="mt-6 space-y-2.5">
                      <p className="text-xs font-medium text-slate-300 uppercase tracking-wider">
                        Aspect Breakdown
                      </p>
                      {result.sentiment_analysis.aspect_sentiments.map((asp, i) => {
                        const aspTheme = getSentimentTheme(asp.sentiment)
                        return (
                          <div
                            key={i}
                            className="p-2.5 rounded-xl bg-slate-900/60 border border-slate-800 text-xs flex items-center justify-between"
                          >
                            <div>
                              <p className="font-semibold text-slate-200">{asp.aspect}</p>
                              {asp.context && (
                                <p className="text-[11px] text-slate-400 italic mt-0.5 truncate max-w-[180px]">
                                  "{asp.context}"
                                </p>
                              )}
                            </div>
                            <span
                              className={`px-2 py-0.5 rounded text-[10px] font-bold ${aspTheme.badgeBg} ${aspTheme.text}`}
                            >
                              {asp.sentiment}
                            </span>
                          </div>
                        )
                      })}
                    </div>
                  )}
                </div>

                <div className="text-[11px] text-slate-500 mt-6 pt-4 border-t border-slate-800 flex items-center justify-between">
                  <span>Model: RoBERTa</span>
                  <span>Confidence: {(result.sentiment_analysis?.confidence_score || 0).toFixed(2)}</span>
                </div>
              </div>

              {/* Transcript & Extracted Entities Panel */}
              <div className="lg:col-span-2 bg-slate-900/90 border border-slate-800 rounded-3xl p-6 sm:p-8 shadow-xl flex flex-col justify-between">
                <div>
                  <div className="flex items-center justify-between mb-4">
                    <h3 className="text-sm font-semibold tracking-wide uppercase text-slate-300 flex items-center gap-2">
                      <FileAudio className="w-4 h-4 text-indigo-400" />
                      Speech Transcript & Entity Highlights
                    </h3>
                    <span className="px-2.5 py-0.5 rounded-full text-xs font-medium bg-indigo-500/20 text-indigo-300 border border-indigo-500/30">
                      {result.transcription?.detected_language || 'Detected'}
                    </span>
                  </div>

                  {/* Inline Highlighted Transcript */}
                  <div className="p-5 rounded-2xl bg-slate-950/80 border border-slate-800/80 text-slate-200 text-base leading-relaxed tracking-normal font-sans selection:bg-indigo-500/30 mb-6">
                    {renderHighlightedTranscript(
                      result.transcription?.transcript || '',
                      result.entity_extraction?.entities || []
                    )}
                  </div>

                  {/* Synchronized Segments Timeline with Interactive Seeking */}
                  {result.transcription?.segments && result.transcription.segments.length > 0 && (
                    <div className="mb-6">
                      <div className="flex items-center justify-between mb-3">
                        <h4 className="text-xs font-semibold uppercase tracking-wider text-slate-400 flex items-center gap-1.5">
                          <Clock className="w-3.5 h-3.5 text-indigo-400" />
                          Time-Aligned Segments (Click to jump audio)
                        </h4>

                        {/* Search input for segments */}
                        <div className="relative w-44">
                          <Search className="w-3 h-3 text-slate-500 absolute left-2.5 top-2.5" />
                          <input
                            type="text"
                            placeholder="Filter speech..."
                            value={searchQuery}
                            onChange={(e) => setSearchQuery(e.target.value)}
                            className="w-full bg-slate-950 border border-slate-800 rounded-lg pl-7 pr-2.5 py-1 text-[11px] text-slate-200 placeholder-slate-500 focus:outline-none focus:border-indigo-500"
                          />
                        </div>
                      </div>

                      <div className="space-y-2 max-h-48 overflow-y-auto pr-1">
                        {filteredSegments.map((seg) => {
                          const isActive =
                            currentTime >= seg.start_time && currentTime <= seg.end_time
                          return (
                            <div
                              key={seg.id}
                              onClick={() => handleSeek(seg.start_time)}
                              className={`p-3 rounded-xl border text-xs cursor-pointer transition-all flex items-start justify-between gap-3 ${
                                isActive
                                  ? 'bg-indigo-600/20 border-indigo-500 text-white shadow-md shadow-indigo-500/10'
                                  : 'bg-slate-950/40 border-slate-800/70 text-slate-300 hover:border-slate-700 hover:bg-slate-950/70'
                              }`}
                            >
                              <div className="flex-1">
                                <p className="font-medium">{seg.text}</p>
                              </div>
                              <div className="flex items-center gap-2 shrink-0 text-[11px] font-mono text-slate-400">
                                <span className="px-1.5 py-0.5 rounded bg-slate-800 text-slate-300">
                                  {formatDuration(seg.start_time)} - {formatDuration(seg.end_time)}
                                </span>
                              </div>
                            </div>
                          )
                        })}
                      </div>
                    </div>
                  )}

                  {/* Extracted Entities Explorer with Category Filters */}
                  <div className="mt-6">
                    <div className="flex flex-wrap items-center justify-between mb-3 gap-2">
                      <h4 className="text-xs font-semibold uppercase tracking-wider text-slate-400 flex items-center gap-1.5">
                        <Tag className="w-3.5 h-3.5 text-indigo-400" />
                        Extracted Named Entities ({filteredEntities.length})
                      </h4>

                      {/* Category Filter Pills */}
                      <div className="flex flex-wrap items-center gap-1 text-[10px]">
                        {['ALL', 'ORGANIZATION', 'LOCATION', 'PERSON', 'MONEY', 'APPLICATION_ID'].map((cat) => (
                          <button
                            key={cat}
                            type="button"
                            onClick={() => setSelectedEntityCategory(cat)}
                            className={`px-2 py-0.5 rounded-md font-medium border transition-colors ${
                              selectedEntityCategory === cat
                                ? 'bg-indigo-600 border-indigo-500 text-white'
                                : 'bg-slate-950/60 border-slate-800 text-slate-400 hover:text-slate-200'
                            }`}
                          >
                            {cat}
                          </button>
                        ))}
                      </div>
                    </div>

                    <div className="grid grid-cols-1 sm:grid-cols-2 gap-2.5">
                      {filteredEntities.map((ent, i) => {
                        const badgeStyle = getEntityBadgeStyle(ent.label)
                        return (
                          <div
                            key={i}
                            className="p-3 rounded-xl bg-slate-950/50 border border-slate-800/80 flex items-center justify-between hover:border-slate-700 transition-colors"
                          >
                            <div className="truncate pr-2">
                              <span className="text-sm font-semibold text-white block truncate">
                                {ent.entity}
                              </span>
                              <span className="text-[10px] text-slate-500">
                                Confidence: {Math.round(ent.confidence * 100)}%
                              </span>
                            </div>
                            <span
                              className={`px-2 py-0.5 rounded text-[10px] font-bold border shrink-0 uppercase tracking-wider font-mono ${badgeStyle}`}
                            >
                              {ent.label}
                            </span>
                          </div>
                        )
                      })}

                      {filteredEntities.length === 0 && (
                        <div className="col-span-2 text-center py-4 text-xs text-slate-500">
                          No entities matching the selected category.
                        </div>
                      )}
                    </div>
                  </div>
                </div>

                {/* Reset Action */}
                <div className="mt-8 pt-4 border-t border-slate-800 flex items-center justify-between text-xs">
                  <span className="text-slate-500">
                    Engine: Whisper + RoBERTa + BERT NER
                  </span>
                  <button
                    type="button"
                    onClick={() => {
                      setFile(null)
                      setResult(null)
                    }}
                    className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-300 transition-colors font-medium"
                  >
                    <RefreshCw className="w-3.5 h-3.5" />
                    <span>Analyze another audio</span>
                  </button>
                </div>
              </div>
            </div>
          </section>
        )}
      </main>

      {/* Footer */}
      <footer className="mt-16 text-center text-xs text-slate-500">
        BhashaSense Enterprise Speech Engine • FastAPI Backend (:8000) + React Vite (:5173)
      </footer>
    </div>
  )
}
