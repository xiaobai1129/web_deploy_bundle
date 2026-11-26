import { useCallback, useEffect, useMemo, useRef, useState, type ChangeEvent } from 'react';
import { Cpu, Download, ShieldCheck, SlidersHorizontal, Sparkles, Upload } from 'lucide-react';
import {
  encodeStreamPrior7d,
  type ModelScale,
  type ModelTier,
} from './core/csr1';
import { getWeightUrl, resolveModelOption } from './core/modelsMeta';
import {
  CodecSREngine,
  type CompareViewMode,
  type GpuDeviceSummary,
} from './engine/webgpuRunner';
import { runGoldenVerificationSuite, type GoldenCaseResult } from './engine/goldenVerify';
import {
  loadDemoVideos,
  probeLocalVideoFile,
  type VideoSourceMeta,
} from './video/videoProbe';
import {
  exportSuperResolvedMp4,
  type ExportProgress,
  type ExportResult,
} from './video/mp4Exporter';
import { ComparisonViewport } from './components/ComparisonViewport';
import { GoldenVerifyModal } from './components/GoldenVerifyModal';
import { ExportModal } from './components/ExportModal';

const PRIOR_LABELS = ['QP 强度', 'HEVC 标志', '直播码控', '对数 BPP', 'RD 复杂度', '屏幕内容', '先验启用'];

export default function App() {
  // Core WebGPU engine state
  const engineRef = useRef<CodecSREngine | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const canvasCtxRef = useRef<GPUCanvasContext | null>(null);
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const fileInputRef = useRef<HTMLInputElement | null>(null);

  const [gpuInfo, setGpuInfo] = useState<GpuDeviceSummary | null>(null);
  const [initError, setInitError] = useState<string | null>(null);
  const [engineReady, setEngineReady] = useState(false);

  // Video source catalog & selection
  const [demoVideos, setDemoVideos] = useState<VideoSourceMeta[]>([]);
  const [uploadedVideos, setUploadedVideos] = useState<VideoSourceMeta[]>([]);
  const [activeSource, setActiveSource] = useState<VideoSourceMeta | null>(null);
  const [uploadError, setUploadError] = useState<string | null>(null);

  // Model & 7D Prior Controls
  const [tier, setTier] = useState<ModelTier>('S');
  const [scale, setScale] = useState<ModelScale>(2.0);
  const [usePrior, setUsePrior] = useState(true);
  const [qpMean, setQpMean] = useState(37.8);
  const [isScreenContent, setIsScreenContent] = useState(true);
  const [isLiveVbv, setIsLiveVbv] = useState(false);
  const [showPriorDetail, setShowPriorDetail] = useState(false);

  // Display & Playback state
  const [viewMode, setViewMode] = useState<CompareViewMode>(0);
  const [splitX, setSplitX] = useState(0.5);
  const [zoom, setZoom] = useState(1);
  const [centerX, setCenterX] = useState(0.5);
  const [centerY, setCenterY] = useState(0.5);
  const [isPlaying, setIsPlaying] = useState(false);
  const [isMuted, setIsMuted] = useState(true);
  const [isLoop, setIsLoop] = useState(true);
  const [currentTime, setCurrentTime] = useState(0);
  const [duration, setDuration] = useState(0);

  // Telemetry
  const [bakeMs, setBakeMs] = useState(0);
  const [inferenceMs, setInferenceMs] = useState(0);
  const [modelLoading, setModelLoading] = useState(false);
  const [runtimeError, setRuntimeError] = useState<string | null>(null);

  // Golden verification modal
  const [goldenOpen, setGoldenOpen] = useState(false);
  const [goldenRunning, setGoldenRunning] = useState(false);
  const [goldenResults, setGoldenResults] = useState<GoldenCaseResult[]>([]);
  const [goldenError, setGoldenError] = useState<string | null>(null);

  // Offline MP4 Export state
  const [exportBitrateMbps, setExportBitrateMbps] = useState(15);
  const [exportModalOpen, setExportModalOpen] = useState(false);
  const [exporting, setExporting] = useState(false);
  const [exportProgress, setExportProgress] = useState<ExportProgress | null>(null);
  const [exportResult, setExportResult] = useState<ExportResult | null>(null);
  const [exportError, setExportError] = useState<string | null>(null);
  const exportAbortRef = useRef<AbortController | null>(null);

  // Refs for latest render parameters (for zero-overhead canvas redraws & rVFC callbacks)
  const displayOptsRef = useRef({ splitX, viewMode, zoom, centerX, centerY });
  displayOptsRef.current = { splitX, viewMode, zoom, centerX, centerY };
  const frameBusyRef = useRef(false);
  const exportingRef = useRef(false);
  exportingRef.current = exporting;

  const activeModel = useMemo(
    () => resolveModelOption(tier, scale, usePrior),
    [tier, scale, usePrior],
  );

  const prior7d = useMemo(() => {
    if (!activeSource) return [0.55, 0, 0, 0.5, 0.5, 0, 1];
    return encodeStreamPrior7d({
      width: activeSource.width,
      height: activeSource.height,
      fps: activeSource.fps,
      bitrateKbps: activeSource.bitrateKbps,
      qpMean,
      codec: activeSource.codec,
      isLive: isLiveVbv,
      isScreenContent,
      hasPrior: usePrior || tier === 'P',
    });
  }, [activeSource, qpMean, isLiveVbv, isScreenContent, usePrior, tier]);

  const outputWidth = activeSource ? Math.round(activeSource.width * scale) : 1920;
  const outputHeight = activeSource ? Math.round(activeSource.height * scale) : 1080;

  // Render current GPU textures to display canvas without re-running inference
  const redrawCanvasOnly = useCallback(() => {
    const engine = engineRef.current;
    const ctx = canvasCtxRef.current;
    if (!engine || !ctx) return;
    engine.renderToCanvas(ctx, displayOptsRef.current);
  }, []);

  // Process the current <video> frame through WebGPU SR and display it
  const processAndDisplayCurrentFrame = useCallback(async () => {
    const engine = engineRef.current;
    const video = videoRef.current;
    const canvas = canvasRef.current;
    if (!engine || !video || !canvas || exportingRef.current) return;
    if (video.readyState < 2 || video.videoWidth <= 0 || video.videoHeight <= 0) return;
    if (!engine.getActiveBaked()) return;
    if (frameBusyRef.current) return;

    frameBusyRef.current = true;
    try {
      let ctx = canvasCtxRef.current;
      if (!ctx) {
        ctx = canvas.getContext('webgpu');
        if (!ctx) throw new Error('无法获取展示画布的 WebGPU 上下文');
        ctx.configure({
          device: engine.device,
          format: engine.canvasFormat,
          alphaMode: 'opaque',
        });
        canvasCtxRef.current = ctx;
      }

      const t0 = performance.now();
      const { outW, outH } = engine.processFrame(video, video.videoWidth, video.videoHeight, true);
      if (canvas.width !== outW || canvas.height !== outH) {
        canvas.width = outW;
        canvas.height = outH;
      }
      engine.renderToCanvas(ctx, displayOptsRef.current);
      await engine.waitGpuIdle();
      const elapsed = performance.now() - t0;
      setInferenceMs((prev) => (prev === 0 ? elapsed : prev * 0.7 + elapsed * 0.3));
      setRuntimeError(null);
    } catch (err) {
      setRuntimeError(err instanceof Error ? err.message : String(err));
    } finally {
      frameBusyRef.current = false;
    }
  }, []);

  // Select a video source and configure recommended scale & prior defaults
  const selectSource = useCallback((src: VideoSourceMeta) => {
    setActiveSource(src);
    setScale(src.recommendedScale);
    setQpMean(src.defaultQp);
    setIsScreenContent(src.defaultIsScreen);
    setIsLiveVbv(src.defaultIsLive);
    setCurrentTime(0);
    setDuration(src.durationSec);
    setInferenceMs(0);
    setZoom(1);
    setCenterX(0.5);
    setCenterY(0.5);
    setUploadError(null);
  }, []);

  // 1. Initialize WebGPU engine and demo catalog on mount
  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const [engine, demos] = await Promise.all([CodecSREngine.create(), loadDemoVideos()]);
        if (cancelled) {
          engine.destroy();
          return;
        }
        engineRef.current = engine;
        setGpuInfo(engine.gpuInfo);
        setDemoVideos(demos);
        setEngineReady(true);
        if (demos.length > 0) {
          selectSource(demos[0]);
        }
      } catch (err) {
        if (!cancelled) {
          setInitError(err instanceof Error ? err.message : String(err));
        }
      }
    })();

    return () => {
      cancelled = true;
      engineRef.current?.destroy();
      engineRef.current = null;
    };
  }, [selectSource]);

  // 2. Load CSR1 model binary whenever activeModel file changes
  const priorRef = useRef(prior7d);
  priorRef.current = prior7d;

  useEffect(() => {
    if (!engineReady || !engineRef.current) return;
    let cancelled = false;
    setModelLoading(true);
    (async () => {
      try {
        const baked = await engineRef.current!.loadModel(
          getWeightUrl(activeModel.fileName),
          activeModel.usePrior ? priorRef.current : null,
        );
        if (cancelled) return;
        setBakeMs(baked.bakeMs);
        setRuntimeError(null);
        await processAndDisplayCurrentFrame();
      } catch (err) {
        if (!cancelled) {
          setRuntimeError(err instanceof Error ? err.message : String(err));
        }
      } finally {
        if (!cancelled) setModelLoading(false);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [engineReady, activeModel.fileName, activeModel.usePrior, processAndDisplayCurrentFrame]);

  // 3. Hot re-bake 7D prior in < 2ms on CPU whenever sliders/toggles change (no weight reload!)
  useEffect(() => {
    const engine = engineRef.current;
    if (!engineReady || !engine || modelLoading || !engine.getActiveBaked()) return;
    if (!activeModel.usePrior) return;
    try {
      const baked = engine.rebakePrior(prior7d);
      setBakeMs(baked.bakeMs);
      void processAndDisplayCurrentFrame();
    } catch (err) {
      setRuntimeError(err instanceof Error ? err.message : String(err));
    }
  }, [engineReady, modelLoading, activeModel.usePrior, prior7d, processAndDisplayCurrentFrame]);

  // 4. Instant 120fps redraw when split divider, view mode, or zoom/pan changes (zero re-inference!)
  useEffect(() => {
    redrawCanvasOnly();
  }, [splitX, viewMode, zoom, centerX, centerY, redrawCanvasOnly]);

  // 5. Bind hidden <video> element events & requestVideoFrameCallback loop
  useEffect(() => {
    const video = videoRef.current;
    if (!video || !activeSource) return;

    video.src = activeSource.url;
    video.muted = isMuted;
    video.loop = isLoop;
    video.load();

    let rvfcId = 0;
    let rafId = 0;
    let stopped = false;

    const onFrame = () => {
      if (stopped) return;
      setCurrentTime(video.currentTime);
      void processAndDisplayCurrentFrame();
      if (!video.paused && !video.ended) {
        scheduleNext();
      }
    };

    const scheduleNext = () => {
      const vWithRvfc = video as HTMLVideoElement & {
        requestVideoFrameCallback?: (cb: () => void) => number;
        cancelVideoFrameCallback?: (id: number) => void;
      };
      if (typeof vWithRvfc.requestVideoFrameCallback === 'function') {
        rvfcId = vWithRvfc.requestVideoFrameCallback(onFrame);
      } else {
        rafId = requestAnimationFrame(onFrame);
      }
    };

    const handleLoadedData = () => {
      setDuration(Number.isFinite(video.duration) ? video.duration : activeSource.durationSec);
      setCurrentTime(video.currentTime);
      void processAndDisplayCurrentFrame();
    };
    const handleSeeked = () => {
      setCurrentTime(video.currentTime);
      void processAndDisplayCurrentFrame();
    };
    const handlePlay = () => {
      setIsPlaying(true);
      scheduleNext();
    };
    const handlePause = () => {
      setIsPlaying(false);
    };

    video.addEventListener('loadeddata', handleLoadedData);
    video.addEventListener('seeked', handleSeeked);
    video.addEventListener('play', handlePlay);
    video.addEventListener('pause', handlePause);
    video.addEventListener('ended', handlePause);

    return () => {
      stopped = true;
      const vWithRvfc = video as HTMLVideoElement & {
        cancelVideoFrameCallback?: (id: number) => void;
      };
      if (rvfcId && typeof vWithRvfc.cancelVideoFrameCallback === 'function') {
        vWithRvfc.cancelVideoFrameCallback(rvfcId);
      }
      if (rafId) cancelAnimationFrame(rafId);
      video.removeEventListener('loadeddata', handleLoadedData);
      video.removeEventListener('seeked', handleSeeked);
      video.removeEventListener('play', handlePlay);
      video.removeEventListener('pause', handlePause);
      video.removeEventListener('ended', handlePause);
    };
  }, [activeSource, processAndDisplayCurrentFrame]);

  // Sync mute & loop flags onto <video>
  useEffect(() => {
    if (videoRef.current) {
      videoRef.current.muted = isMuted;
      videoRef.current.loop = isLoop;
    }
  }, [isMuted, isLoop]);

  // Upload local video handler
  const handleLocalFile = useCallback(
    async (file: File) => {
      setUploadError(null);
      try {
        const meta = await probeLocalVideoFile(file);
        setUploadedVideos((prev) => [meta, ...prev.slice(0, 5)]);
        selectSource(meta);
      } catch (err) {
        setUploadError(err instanceof Error ? err.message : String(err));
      }
    },
    [selectSource],
  );

  const onFileInputChange = (e: ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      void handleLocalFile(file);
    }
    e.target.value = '';
  };

  // Playback transport controls
  const togglePlay = () => {
    const v = videoRef.current;
    if (!v) return;
    if (v.paused) {
      void v.play().catch(() => {});
    } else {
      v.pause();
    }
  };

  const stepFrame = (delta: number) => {
    const v = videoRef.current;
    if (!v || !activeSource) return;
    v.pause();
    const step = 1 / Math.max(1, activeSource.fps || 30);
    v.currentTime = Math.min(Math.max(0, v.duration || activeSource.durationSec), Math.max(0, v.currentTime + delta * step));
  };

  const seekTo = (t: number) => {
    const v = videoRef.current;
    if (!v) return;
    v.currentTime = t;
    setCurrentTime(t);
  };

  // Golden Vector Verification Suite
  const handleRunGolden = async () => {
    const engine = engineRef.current;
    if (!engine) return;
    setGoldenRunning(true);
    setGoldenError(null);
    setGoldenResults([]);
    try {
      await runGoldenVerificationSuite(engine, (res) => {
        setGoldenResults((prev) => [...prev, res]);
      });
    } catch (err) {
      setGoldenError(err instanceof Error ? err.message : String(err));
    } finally {
      setGoldenRunning(false);
    }
  };

  // Export Super-Resolved MP4 locally
  const handleStartExport = async () => {
    const engine = engineRef.current;
    if (!engine || !activeSource || exporting) return;

    // Pause live playback so GPU is dedicated to fast offline export
    videoRef.current?.pause();
    if (exportResult?.url) {
      URL.revokeObjectURL(exportResult.url);
    }

    const abort = new AbortController();
    exportAbortRef.current = abort;
    setExportModalOpen(true);
    setExporting(true);
    setExportResult(null);
    setExportError(null);
    setExportProgress({
      phase: 'demux',
      doneFrames: 0,
      totalFrames: activeSource.frameCount,
      fps: 0,
      etaSec: 0,
      message: '准备本地 WebGPU 超分导出...',
    });

    try {
      const res = await exportSuperResolvedMp4({
        engine,
        source: activeSource,
        bitrateMbps: exportBitrateMbps,
        tierLabel: tier,
        scale,
        signal: abort.signal,
        onProgress: (p) => setExportProgress(p),
      });
      setExportResult(res);
    } catch (err) {
      if (err instanceof DOMException && err.name === 'AbortError') {
        setExportError('已取消导出任务。');
      } else {
        setExportError(err instanceof Error ? err.message : String(err));
      }
    } finally {
      setExporting(false);
      exportAbortRef.current = null;
      // Restore current frame on live comparison canvas after export finishes
      setTimeout(() => {
        void processAndDisplayCurrentFrame();
      }, 50);
    }
  };

  const triggerDownload = (res: ExportResult) => {
    const a = document.createElement('a');
    a.href = res.url;
    a.download = res.fileName;
    document.body.appendChild(a);
    a.click();
    a.remove();
  };

  const highResWarning =
    activeSource && (activeSource.height > 720 || activeSource.durationSec > 35);

  return (
    <div className="app-shell">
      {/* Visually-hidden hardware-decoded video feeder (avoid display:none so Chrome allocates GPU frame backing) */}
      <video
        ref={videoRef}
        playsInline
        preload="auto"
        style={{
          position: 'fixed',
          left: 0,
          top: 0,
          width: 2,
          height: 2,
          opacity: 0.001,
          pointerEvents: 'none',
          zIndex: -1,
        }}
      />
      <input
        ref={fileInputRef}
        type="file"
        accept="video/mp4,video/webm,video/quicktime,video/*"
        style={{ display: 'none' }}
        onChange={onFileInputChange}
      />

      {/* Top Navigation Bar */}
      <header className="topbar">
        <div className="brand">
          <div className="brand-mark">
            <Sparkles size={18} />
          </div>
          <div>
            <div className="brand-title">
              CodecSR V3.5
              <span className="badge-pill">WebGPU 本地算力</span>
              <span className="badge-pill ok">零云端上传</span>
            </div>
            <div className="brand-sub">纯前端实时视频超分对比与本地 MP4 导出工作台</div>
          </div>
        </div>

        <div className="topbar-actions">
          <div className="gpu-chip" title={gpuInfo?.description || initError || '正在初始化 WebGPU...'}>
            <span className={`gpu-dot ${initError ? 'err' : ''}`} />
            <Cpu size={13} />
            <span>{initError ? 'WebGPU 不可用' : gpuInfo ? `${gpuInfo.vendor} · ${gpuInfo.description}` : '初始化 GPU...'}</span>
          </div>

          <button
            className="btn"
            disabled={!engineReady || exporting}
            onClick={() => {
              setGoldenOpen(true);
              if (goldenResults.length === 0 && !goldenRunning) {
                void handleRunGolden();
              }
            }}
          >
            <ShieldCheck size={15} />
            精度对拍
          </button>

          <button
            className="btn btn-primary"
            disabled={!engineReady || !activeSource || exporting || modelLoading}
            onClick={() => void handleStartExport()}
          >
            <Download size={15} />
            导出超分 MP4
          </button>
        </div>
      </header>

      {/* Main Content */}
      {initError ? (
        <div style={{ padding: 32, maxWidth: 680, margin: '40px auto' }}>
          <div className="error-box">
            <strong>WebGPU 初始化失败</strong>
            <div style={{ marginTop: 8 }}>{initError}</div>
          </div>
        </div>
      ) : (
        <main className="workspace">
          <ComparisonViewport
            canvasRef={canvasRef}
            canvasWidth={outputWidth}
            canvasHeight={outputHeight}
            inputWidth={activeSource?.width ?? 960}
            inputHeight={activeSource?.height ?? 540}
            tierLabel={`Client-${tier}`}
            scale={scale}
            viewMode={viewMode}
            splitX={splitX}
            zoom={zoom}
            centerX={centerX}
            centerY={centerY}
            inferenceMs={inferenceMs}
            bakeMs={bakeMs}
            isPlaying={isPlaying}
            isMuted={isMuted}
            isLoop={isLoop}
            currentTime={currentTime}
            duration={duration}
            onViewModeChange={setViewMode}
            onSplitChange={setSplitX}
            onZoomChange={setZoom}
            onCenterChange={(cx, cy) => {
              setCenterX(cx);
              setCenterY(cy);
            }}
            onTogglePlay={togglePlay}
            onStepFrame={stepFrame}
            onSeek={seekTo}
            onToggleMute={() => setIsMuted((m) => !m)}
            onToggleLoop={() => setIsLoop((l) => !l)}
            onDropFile={(f) => void handleLocalFile(f)}
            onTriggerUpload={() => fileInputRef.current?.click()}
          />

          {/* Right Control Panel */}
          <aside className="sidebar">
            {runtimeError && <div className="error-box">{runtimeError}</div>}

            {/* 1. Video Source Selection */}
            <section className="panel-card">
              <div className="panel-title">
                <span>1. 视频源（本地上传 / 演示样片）</span>
                {activeSource && !activeSource.isDemo && <span className="badge-pill ok">本地视频</span>}
              </div>

              <div className="upload-zone" onClick={() => fileInputRef.current?.click()}>
                <Upload size={20} color="#38bdf8" />
                <div className="upload-zone-title">点击选择或拖拽本地视频到画面</div>
                <div className="upload-zone-sub">支持 MP4 / WebM / MOV · 全程在您的显卡本地推理</div>
              </div>

              {uploadError && <div className="error-box">{uploadError}</div>}

              {uploadedVideos.length > 0 && (
                <div className="demo-list">
                  {uploadedVideos.map((u) => (
                    <button
                      key={u.id}
                      className={`demo-item ${activeSource?.id === u.id ? 'active' : ''}`}
                      onClick={() => selectSource(u)}
                    >
                      <div className="demo-item-top">
                        <span style={{ overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                          {u.name}
                        </span>
                        <span className="badge-pill ok">上传</span>
                      </div>
                      <div className="demo-item-desc">{u.description}</div>
                    </button>
                  ))}
                </div>
              )}

              <div className="demo-list">
                {demoVideos.map((d) => (
                  <button
                    key={d.id}
                    className={`demo-item ${activeSource?.id === d.id ? 'active' : ''}`}
                    onClick={() => selectSource(d)}
                  >
                    <div className="demo-item-top">
                      <span>{d.name.replace(/\.mp4$/, '')}</span>
                      <span className="badge-pill">
                        {d.width}×{d.height} · ×{d.recommendedScale}
                      </span>
                    </div>
                    <div className="demo-item-desc">{d.description}</div>
                  </button>
                ))}
              </div>
            </section>

            {/* 2. Model Tier & Scale */}
            <section className="panel-card">
              <div className="panel-title">
                <span>2. 超分模型档位</span>
                <span>{activeModel.sizeKb} KB · {activeModel.gmacs} GMACs</span>
              </div>

              <div className="tier-grid">
                {(['L', 'S', 'P'] as ModelTier[]).map((t) => (
                  <button
                    key={t}
                    className={`tier-btn ${tier === t ? 'active' : ''}`}
                    onClick={() => setTier(t)}
                  >
                    <strong>Client-{t}</strong>
                    <span>{t === 'L' ? '极速低延迟' : t === 'S' ? '标准推荐' : '旗舰沙漏'}</span>
                  </button>
                ))}
              </div>

              <div className="field-row">
                <span>超分倍率</span>
                <div className="segmented">
                  <button className={scale === 2.0 ? 'active' : ''} onClick={() => setScale(2.0)}>
                    ×2.0 (540p→1080p)
                  </button>
                  <button className={scale === 1.5 ? 'active' : ''} onClick={() => setScale(1.5)}>
                    ×1.5 (720p→1080p)
                  </button>
                </div>
              </div>

              {tier !== 'P' && (
                <label className="toggle-row">
                  <span>启用 7D 码流动态先验（抗压缩伪影）</span>
                  <input
                    type="checkbox"
                    checked={usePrior}
                    onChange={(e) => setUsePrior(e.target.checked)}
                  />
                </label>
              )}

              <div style={{ fontSize: 11, color: '#94a3b8', lineHeight: 1.45 }}>
                {activeModel.tagline}
              </div>
            </section>

            {/* 3. 7D Stream Prior Live Baking */}
            <section className="panel-card">
              <div className="panel-title">
                <span style={{ display: 'inline-flex', alignItems: 'center', gap: 5 }}>
                  <SlidersHorizontal size={13} /> 3. 码流先验实时调节
                </span>
                <span style={{ color: '#38bdf8' }}>CPU {bakeMs.toFixed(2)} ms 热烘焙</span>
              </div>

              <div>
                <div className="field-row" style={{ marginBottom: 4 }}>
                  <span>压缩伪影修复强度（虚拟 QP）</span>
                  <strong>QP {qpMean.toFixed(1)}</strong>
                </div>
                <input
                  type="range"
                  className="slider"
                  min={20}
                  max={42}
                  step={0.5}
                  value={qpMean}
                  disabled={!activeModel.usePrior}
                  onChange={(e) => setQpMean(Number(e.target.value))}
                />
                <div className="field-row" style={{ fontSize: 10, color: '#64748b' }}>
                  <span>20 · 轻度增强</span>
                  <span>30 · 标准平衡</span>
                  <span>42 · 强力去块/去振铃</span>
                </div>
              </div>

              <label className="toggle-row">
                <span>文字 / UI / 屏幕内容抗振铃优化</span>
                <input
                  type="checkbox"
                  checked={isScreenContent}
                  disabled={!activeModel.usePrior}
                  onChange={(e) => setIsScreenContent(e.target.checked)}
                />
              </label>

              <label className="toggle-row">
                <span>低码率直播 VBV 码控模式</span>
                <input
                  type="checkbox"
                  checked={isLiveVbv}
                  disabled={!activeModel.usePrior}
                  onChange={(e) => setIsLiveVbv(e.target.checked)}
                />
              </label>

              <button
                className="btn"
                style={{ padding: '5px 10px', fontSize: 11 }}
                onClick={() => setShowPriorDetail((s) => !s)}
              >
                {showPriorDetail ? '收起 7D 先验向量' : '查看实时 7D 先验向量'}
              </button>

              {showPriorDetail && (
                <div className="prior-grid">
                  {PRIOR_LABELS.map((lbl, idx) => (
                    <span key={lbl}>
                      {lbl} <b>{prior7d[idx].toFixed(3)}</b>
                    </span>
                  ))}
                </div>
              )}
            </section>

            {/* 4. Local MP4 Export Card */}
            <section className="panel-card">
              <div className="panel-title">
                <span>4. 本地导出超分 MP4</span>
                <span>{outputWidth}×{outputHeight}</span>
              </div>

              <div className="field-row">
                <span>目标码率</span>
                <div className="segmented">
                  {[8, 15, 25].map((mbps) => (
                    <button
                      key={mbps}
                      className={exportBitrateMbps === mbps ? 'active' : ''}
                      onClick={() => setExportBitrateMbps(mbps)}
                    >
                      {mbps} Mbps
                    </button>
                  ))}
                </div>
              </div>

              {highResWarning && (
                <div className="notice-box">
                  当前视频分辨率较高或时长较长，导出耗时将随总帧数线性增加；建议优先选用 <strong>Client-L / S</strong> 档位以获得流畅速度。
                </div>
              )}

              <button
                className="btn btn-primary"
                style={{ width: '100%', padding: '10px 14px' }}
                disabled={!engineReady || !activeSource || exporting || modelLoading}
                onClick={() => void handleStartExport()}
              >
                <Download size={15} />
                {exporting ? '正在导出中...' : `开始本地超分并导出 MP4`}
              </button>

              {exportResult && !exporting && (
                <button
                  className="btn"
                  style={{ width: '100%' }}
                  onClick={() => setExportModalOpen(true)}
                >
                  查看上次导出预览 & 再次下载
                </button>
              )}
            </section>
          </aside>
        </main>
      )}

      <GoldenVerifyModal
        open={goldenOpen}
        running={goldenRunning}
        results={goldenResults}
        error={goldenError}
        onRun={() => void handleRunGolden()}
        onClose={() => setGoldenOpen(false)}
      />

      <ExportModal
        open={exportModalOpen}
        exporting={exporting}
        progress={exportProgress}
        result={exportResult}
        error={exportError}
        onCancel={() => exportAbortRef.current?.abort()}
        onClose={() => setExportModalOpen(false)}
        onDownload={triggerDownload}
      />
    </div>
  );
}
