import { useCallback, useEffect, useMemo, useRef, useState, type ChangeEvent } from 'react';
import {
  ChevronDown,
  ChevronRight,
  Cpu,
  Download,
  Film,
  ShieldCheck,
  SlidersHorizontal,
  Sparkles,
  Upload,
} from 'lucide-react';
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
  computeWorkingResolution,
  formatCodecDisplayName,
  loadDemoVideos,
  probeLocalVideoFile,
  recommendScaleForResolution,
  type InputResMode,
  type VideoSourceMeta,
} from './video/videoProbe';
import {
  exportSuperResolvedMp4,
  type ExportProgress,
  type ExportResult,
} from './video/mp4Exporter';
import { ComparisonViewport, type StageDisplayMode } from './components/ComparisonViewport';
import { GoldenVerifyModal } from './components/GoldenVerifyModal';
import { ExportModal } from './components/ExportModal';

const PRIOR_LABELS = ['QP 强度', 'HEVC 标志', '直播码控', '对数 BPP', 'RD 复杂度', '屏幕内容', '先验启用'];
const CLIP_LIMIT_SEC = 10;

const TIER_META: Record<ModelTier, { name: string; sub: string }> = {
  L: { name: '极速', sub: '更快响应' },
  S: { name: '均衡', sub: '日常推荐' },
  P: { name: '精修', sub: '细节最佳' },
};

const DEMO_SHORT_TITLES: Record<string, string> = {
  '01_ui_slide_editing_540p_x2.mp4': '文字与 PPT 录屏',
  '02_cg_china_speed_540p_x2.mp4': 'CG 动画线条',
  '03_detail_cactus_540p_x2.mp4': '低码率纹理去块',
  '04_motion_jockey_720p_x1_5.mp4': '高速运动实拍',
};

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
  const [inputResMode, setInputResMode] = useState<InputResMode>('original');

  // Model & 7D Prior Controls (shared by BOTH live preview AND MP4 export)
  const [tier, setTier] = useState<ModelTier>('S');
  const [scale, setScale] = useState<ModelScale>(2.0);
  const [usePrior, setUsePrior] = useState(true);
  const [qpMean, setQpMean] = useState(37.8);
  const [isScreenContent, setIsScreenContent] = useState(true);
  const [isLiveVbv, setIsLiveVbv] = useState(false);
  const [showAdvanced, setShowAdvanced] = useState(false);

  // Display & Playback state
  const [viewMode, setViewMode] = useState<CompareViewMode>(0);
  const [splitX, setSplitX] = useState(0.5);
  const [zoom, setZoom] = useState(1);
  const [centerX, setCenterX] = useState(0.5);
  const [centerY, setCenterY] = useState(0.5);
  const [stageDisplayMode, setStageDisplayMode] = useState<StageDisplayMode>('fit');
  const [stageFillPct, setStageFillPct] = useState(100);
  const displayBoxRef = useRef({ w: 0, h: 0 });
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
  const [exportClipMode, setExportClipMode] = useState<'first10s' | 'full'>('full');
  const [exportModalOpen, setExportModalOpen] = useState(false);
  const [exporting, setExporting] = useState(false);
  const [exportProgress, setExportProgress] = useState<ExportProgress | null>(null);
  const [exportResult, setExportResult] = useState<ExportResult | null>(null);
  const [exportError, setExportError] = useState<string | null>(null);
  const exportAbortRef = useRef<AbortController | null>(null);

  const activeModel = useMemo(
    () => resolveModelOption(tier, scale, usePrior),
    [tier, scale, usePrior],
  );

  // Working GPU input resolution (auto-capped for >720p portrait/landscape uploads unless user picks 'original')
  const workingRes = useMemo(() => {
    if (!activeSource) return { workW: 960, workH: 540, wasDownscaled: false };
    return computeWorkingResolution(activeSource.width, activeSource.height, inputResMode);
  }, [activeSource, inputResMode]);

  const outputWidth = Math.round(workingRes.workW * scale);
  const outputHeight = Math.round(workingRes.workH * scale);

  // Refs for latest render parameters (for zero-overhead canvas redraws & rVFC callbacks)
  const displayOptsRef = useRef({ splitX, viewMode, zoom, centerX, centerY });
  displayOptsRef.current = { splitX, viewMode, zoom, centerX, centerY };
  const workingResRef = useRef(workingRes);
  workingResRef.current = workingRes;
  const outputSizeRef = useRef({ outputWidth, outputHeight });
  outputSizeRef.current = { outputWidth, outputHeight };
  const inFlightGpuRef = useRef(0);
  const lastTelemetryAtRef = useRef(0);
  const exportingRef = useRef(false);
  exportingRef.current = exporting;

  const prior7d = useMemo(() => {
    if (!activeSource) return [0.55, 0, 0, 0.5, 0.5, 0, 1];
    return encodeStreamPrior7d({
      width: workingRes.workW,
      height: workingRes.workH,
      fps: activeSource.fps,
      bitrateKbps: activeSource.bitrateKbps,
      qpMean,
      codec: activeSource.codec,
      isLive: isLiveVbv,
      isScreenContent,
      hasPrior: usePrior || tier === 'P',
    });
  }, [activeSource, workingRes.workW, workingRes.workH, qpMean, isLiveVbv, isScreenContent, usePrior, tier]);

  // Size preview canvas backing buffer to stay razor-sharp even when small 144p outputs are scaled up to fill stage
  const syncPreviewCanvasBackingSize = useCallback((targetOutW: number, targetOutH: number) => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const dispW = displayBoxRef.current.w || canvas.clientWidth || targetOutW;
    const dispH = displayBoxRef.current.h || canvas.clientHeight || targetOutH;
    const scaleFactor = Math.min(
      2560 / Math.max(1, targetOutW, targetOutH),
      Math.max(1, (dispW * dpr) / Math.max(1, targetOutW), (dispH * dpr) / Math.max(1, targetOutH)),
    );
    const backingW = Math.max(2, Math.round(targetOutW * scaleFactor));
    const backingH = Math.max(2, Math.round(targetOutH * scaleFactor));
    if (canvas.width !== backingW || canvas.height !== backingH) {
      canvas.width = backingW;
      canvas.height = backingH;
    }
  }, []);

  // Render current GPU textures to display canvas without re-running inference
  const redrawCanvasOnly = useCallback(() => {
    const engine = engineRef.current;
    const ctx = canvasCtxRef.current;
    if (!engine || !ctx) return;
    const { outputWidth: ow, outputHeight: oh } = outputSizeRef.current;
    syncPreviewCanvasBackingSize(ow, oh);
    engine.renderToCanvas(ctx, displayOptsRef.current);
  }, [syncPreviewCanvasBackingSize]);

  const handleDisplaySizeChange = useCallback(
    (w: number, h: number) => {
      displayBoxRef.current = { w, h };
      redrawCanvasOnly();
    },
    [redrawCanvasOnly],
  );

  // Process the current <video> frame through WebGPU SR and display it in a SINGLE non-blocking GPU submit
  const processAndDisplayCurrentFrame = useCallback(() => {
    const engine = engineRef.current;
    const video = videoRef.current;
    const canvas = canvasRef.current;
    if (!engine || !video || !canvas || exportingRef.current) return;
    if (video.readyState < 2 || video.videoWidth <= 0 || video.videoHeight <= 0) return;
    if (!engine.getActiveBaked()) return;
    if (inFlightGpuRef.current >= 1) return;

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

      const baked = engine.getActiveBaked()!;
      const { workW, workH } = workingResRef.current;
      const targetOutW = Math.round(workW * baked.scalars.scale);
      const targetOutH = Math.round(workH * baked.scalars.scale);
      syncPreviewCanvasBackingSize(targetOutW, targetOutH);

      const t0 = performance.now();
      inFlightGpuRef.current++;
      engine.processFrame(
        video,
        video.videoWidth,
        video.videoHeight,
        true,
        {
          ctx,
          opts: displayOptsRef.current,
        },
        { workW, workH },
      );

      engine
        .waitGpuIdle()
        .then(() => {
          const elapsed = performance.now() - t0;
          const now = performance.now();
          if (lastTelemetryAtRef.current === 0 || now - lastTelemetryAtRef.current > 100) {
            lastTelemetryAtRef.current = now;
            setInferenceMs((prev) => (prev === 0 ? elapsed : prev * 0.75 + elapsed * 0.25));
          }
        })
        .catch(() => {})
        .finally(() => {
          inFlightGpuRef.current = Math.max(0, inFlightGpuRef.current - 1);
        });

      setRuntimeError(null);
    } catch (err) {
      inFlightGpuRef.current = Math.max(0, inFlightGpuRef.current - 1);
      setRuntimeError(err instanceof Error ? err.message : String(err));
    }
  }, [syncPreviewCanvasBackingSize]);

  // Select a video source and configure smart working resolution, scale, clip mode & prior defaults
  const selectSource = useCallback((src: VideoSourceMeta) => {
    setActiveSource(src);
    setInputResMode(src.defaultInputMode);
    setScale(src.recommendedScale);
    setQpMean(src.defaultQp);
    setIsScreenContent(src.defaultIsScreen);
    setIsLiveVbv(src.defaultIsLive);
    setExportClipMode(src.durationSec > CLIP_LIMIT_SEC + 0.25 ? 'first10s' : 'full');
    setCurrentTime(0);
    setDuration(src.durationSec);
    setInferenceMs(0);
    lastTelemetryAtRef.current = 0;
    setZoom(1);
    setCenterX(0.5);
    setCenterY(0.5);
    setStageDisplayMode('fit');
    setStageFillPct(100);
    setUploadError(null);
  }, []);

  const handleInputModeChange = (mode: InputResMode) => {
    setInputResMode(mode);
    setInferenceMs(0);
    lastTelemetryAtRef.current = 0;
    if (activeSource) {
      const next = computeWorkingResolution(activeSource.width, activeSource.height, mode);
      setScale(recommendScaleForResolution(next.workW, next.workH));
    }
  };

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
        processAndDisplayCurrentFrame();
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

  // 3. Hot re-bake 7D prior in < 2ms on CPU whenever sliders/toggles/workingRes change
  useEffect(() => {
    const engine = engineRef.current;
    if (!engineReady || !engine || modelLoading || !engine.getActiveBaked()) return;
    if (!activeModel.usePrior) {
      processAndDisplayCurrentFrame();
      return;
    }
    try {
      const baked = engine.rebakePrior(prior7d);
      setBakeMs(baked.bakeMs);
      processAndDisplayCurrentFrame();
    } catch (err) {
      setRuntimeError(err instanceof Error ? err.message : String(err));
    }
  }, [engineReady, modelLoading, activeModel.usePrior, prior7d, workingRes, processAndDisplayCurrentFrame]);

  // 4. Instant 120fps redraw when split divider, view mode, or zoom/pan changes (zero re-inference!)
  useEffect(() => {
    redrawCanvasOnly();
  }, [splitX, viewMode, zoom, centerX, centerY, redrawCanvasOnly]);

  // 5. Bind visually-hidden <video> element events & requestVideoFrameCallback loop
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
      processAndDisplayCurrentFrame();
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
      processAndDisplayCurrentFrame();
    };
    const handleSeeked = () => {
      setCurrentTime(video.currentTime);
      processAndDisplayCurrentFrame();
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

  useEffect(() => {
    if (videoRef.current) {
      videoRef.current.muted = isMuted;
      videoRef.current.loop = isLoop;
    }
  }, [isMuted, isLoop]);

  const handleLocalFile = useCallback(
    async (file: File) => {
      setUploadError(null);
      try {
        const meta = await probeLocalVideoFile(file);
        setUploadedVideos((prev) => [meta, ...prev.slice(0, 4)]);
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
    v.currentTime = Math.min(
      Math.max(0, v.duration || activeSource.durationSec),
      Math.max(0, v.currentTime + delta * step),
    );
  };

  const seekTo = (t: number) => {
    const v = videoRef.current;
    if (!v) return;
    v.currentTime = t;
    setCurrentTime(t);
  };

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

  const isLongVideo = Boolean(activeSource && activeSource.durationSec > CLIP_LIMIT_SEC + 0.25);
  const effectiveExportDurSec = activeSource
    ? isLongVideo && exportClipMode === 'first10s'
      ? CLIP_LIMIT_SEC
      : activeSource.durationSec
    : 0;
  const estExportFrames = activeSource
    ? Math.max(1, Math.round(effectiveExportDurSec * (activeSource.fps || 30)))
    : 0;

  const handleStartExport = async () => {
    const engine = engineRef.current;
    if (!engine || !activeSource || exporting) return;

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
      totalFrames: estExportFrames,
      fps: 0,
      etaSec: 0,
      message: '准备本地超分导出...',
    });

    try {
      const res = await exportSuperResolvedMp4({
        engine,
        source: activeSource,
        workW: workingRes.workW,
        workH: workingRes.workH,
        bitrateMbps: exportBitrateMbps,
        tierLabel: tier,
        scale,
        maxDurationSec: isLongVideo && exportClipMode === 'first10s' ? CLIP_LIMIT_SEC : undefined,
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
      setTimeout(() => {
        processAndDisplayCurrentFrame();
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

  const strengthLabel = qpMean <= 26 ? '轻度' : qpMean <= 34 ? '适中' : '强力';

  return (
    <div className="app-shell">
      {/* Visually-hidden hardware-decoded video feeder */}
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

      {/* Minimalist Top Bar */}
      <header className="topbar">
        <div className="brand">
          <div className="brand-mark">
            <Sparkles size={17} />
          </div>
          <div className="brand-text">
            <div className="brand-title">
              CodecSR
              <span className="badge-pill ok">本地 GPU 超分</span>
            </div>
            <div className="brand-sub">画质增强实时对比 · 全程本地处理不上传</div>
          </div>
        </div>

        <div className="topbar-actions">
          <div className="gpu-chip" title={gpuInfo?.description || initError || '正在初始化 WebGPU...'}>
            <span className={`gpu-dot ${initError ? 'err' : ''}`} />
            <Cpu size={13} />
            <span>
              {initError
                ? 'WebGPU 不可用'
                : gpuInfo
                  ? gpuInfo.description || gpuInfo.vendor
                  : '初始化 GPU...'}
            </span>
          </div>

          <button
            className="btn btn-ghost"
            disabled={!engineReady || exporting}
            onClick={() => {
              setGoldenOpen(true);
              if (goldenResults.length === 0 && !goldenRunning) {
                void handleRunGolden();
              }
            }}
            title="运行 6 组标准向量验证当前 GPU 推理精度"
            aria-label="精度校验"
          >
            <ShieldCheck size={14} />
            <span className="hide-mobile">精度校验</span>
          </button>

          <button
            className="btn btn-primary"
            disabled={!engineReady || !activeSource || exporting || modelLoading}
            onClick={() => void handleStartExport()}
          >
            <Download size={14} />
            导出 MP4
          </button>
        </div>
      </header>

      {/* Main Content */}
      {initError ? (
        <div style={{ padding: 32, maxWidth: 620, margin: '48px auto' }}>
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
            inputWidth={workingRes.workW}
            inputHeight={workingRes.workH}
            tierLabel={TIER_META[tier].name}
            scale={scale}
            viewMode={viewMode}
            splitX={splitX}
            zoom={zoom}
            centerX={centerX}
            centerY={centerY}
            stageDisplayMode={stageDisplayMode}
            stageFillPct={stageFillPct}
            inferenceMs={inferenceMs}
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
            onStageDisplayModeChange={setStageDisplayMode}
            onStageFillPctChange={setStageFillPct}
            onDisplaySizeChange={handleDisplaySizeChange}
            onTogglePlay={togglePlay}
            onStepFrame={stepFrame}
            onSeek={seekTo}
            onToggleMute={() => setIsMuted((m) => !m)}
            onToggleLoop={() => setIsLoop((l) => !l)}
            onDropFile={(f) => void handleLocalFile(f)}
            onTriggerUpload={() => fileInputRef.current?.click()}
          />

          {/* Right Control Sidebar — Clean 3-Step Flow */}
          <aside className="sidebar">
            {runtimeError && <div className="error-box">{runtimeError}</div>}

            {/* Step 1: Video Source */}
            <section className="panel-card">
              <div className="panel-title">
                <span className="panel-title-left">
                  <Film size={13} /> 选择视频
                </span>
                {activeSource && (
                  <span className={`badge-pill ${activeSource.isDemo ? '' : 'ok'}`}>
                    {activeSource.isDemo ? '演示样片' : '本地视频'}
                  </span>
                )}
              </div>

              <div className="upload-zone" onClick={() => fileInputRef.current?.click()}>
                <div className="upload-icon-wrap">
                  <Upload size={16} />
                </div>
                <div className="upload-zone-body">
                  <div className="upload-zone-title">上传本地视频</div>
                  <div className="upload-zone-sub">点击选择或拖拽到左侧画面 · MP4 / MOV / WebM</div>
                </div>
              </div>

              {uploadError && <div className="error-box">{uploadError}</div>}

              {activeSource && !activeSource.isDemo && activeSource.isHighRes && (
                <div className="info-box">
                  <div className="info-box-head">
                    <strong>超清原片 ({activeSource.width}×{activeSource.height})</strong>
                    <span>{formatCodecDisplayName(activeSource.codec)}</span>
                  </div>
                  <div className="segmented" style={{ marginTop: 8, width: '100%' }}>
                    <button
                      style={{ flex: 1, justifyContent: 'center' }}
                      className={inputResMode === 'auto_720p' ? 'active' : ''}
                      onClick={() => handleInputModeChange('auto_720p')}
                    >
                      720p 流畅
                    </button>
                    <button
                      style={{ flex: 1, justifyContent: 'center' }}
                      className={inputResMode === 'auto_540p' ? 'active' : ''}
                      onClick={() => handleInputModeChange('auto_540p')}
                    >
                      540p 极速
                    </button>
                    <button
                      style={{ flex: 1, justifyContent: 'center' }}
                      className={inputResMode === 'original' ? 'active' : ''}
                      onClick={() => handleInputModeChange('original')}
                    >
                      原画尺寸
                    </button>
                  </div>
                </div>
              )}

              {uploadedVideos.length > 0 && (
                <div className="source-group">
                  <div className="source-group-label">已导入</div>
                  <div className="demo-list">
                    {uploadedVideos.map((u) => (
                      <button
                        key={u.id}
                        className={`demo-item ${activeSource?.id === u.id ? 'active' : ''}`}
                        onClick={() => selectSource(u)}
                      >
                        <div className="demo-item-top">
                          <span className="demo-name">{u.name}</span>
                          <span className="demo-meta">
                            {u.width}×{u.height} · {u.durationSec.toFixed(0)}s
                          </span>
                        </div>
                      </button>
                    ))}
                  </div>
                </div>
              )}

              <div className="source-group">
                <div className="source-group-label">内置样片</div>
                <div className="demo-list">
                  {demoVideos.map((d) => (
                    <button
                      key={d.id}
                      className={`demo-item ${activeSource?.id === d.id ? 'active' : ''}`}
                      onClick={() => selectSource(d)}
                    >
                      <div className="demo-item-top">
                        <span className="demo-name">
                          {DEMO_SHORT_TITLES[d.name] ?? d.name.replace(/\.mp4$/, '')}
                        </span>
                        <span className="demo-meta">
                          {d.height}p · ×{d.recommendedScale}
                        </span>
                      </div>
                    </button>
                  ))}
                </div>
              </div>
            </section>

            {/* Step 2: Enhancement Settings */}
            <section className="panel-card">
              <div className="panel-title">
                <span className="panel-title-left">
                  <SlidersHorizontal size={13} /> 画质增强
                </span>
                <span className="res-preview-pill">
                  {workingRes.workW}×{workingRes.workH} → {outputWidth}×{outputHeight}
                </span>
              </div>

              <div className="tier-grid">
                {(['L', 'S', 'P'] as ModelTier[]).map((t) => (
                  <button
                    key={t}
                    className={`tier-btn ${tier === t ? 'active' : ''}`}
                    onClick={() => setTier(t)}
                  >
                    <strong>{TIER_META[t].name}</strong>
                    <span>{TIER_META[t].sub}</span>
                  </button>
                ))}
              </div>

              <div className="field-row">
                <span>放大倍率</span>
                <div className="segmented">
                  <button className={scale === 2.0 ? 'active' : ''} onClick={() => setScale(2.0)}>
                    ×2.0
                  </button>
                  <button className={scale === 1.5 ? 'active' : ''} onClick={() => setScale(1.5)}>
                    ×1.5
                  </button>
                </div>
              </div>

              <div className="slider-block">
                <div className="field-row">
                  <span>去模糊强度</span>
                  <span className="strength-readout">
                    <strong>{strengthLabel}</strong>
                    <small>QP {qpMean.toFixed(0)}</small>
                  </span>
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
                <div className="slider-ticks">
                  <span>柔和</span>
                  <span>均衡</span>
                  <span>强力去块</span>
                </div>
              </div>

              <div className="scene-chip-row">
                <button
                  type="button"
                  className={`scene-chip ${isScreenContent ? 'active' : ''}`}
                  disabled={!activeModel.usePrior}
                  onClick={() => setIsScreenContent((v) => !v)}
                  title="针对文字、字幕、PPT、录屏减少边缘白边重影"
                >
                  <span className="chip-dot" />
                  文字 / 字幕锐化
                </button>
                <button
                  type="button"
                  className={`scene-chip ${isLiveVbv ? 'active' : ''}`}
                  disabled={!activeModel.usePrior}
                  onClick={() => setIsLiveVbv((v) => !v)}
                  title="针对低码率直播或严重压缩画面增强去块效果"
                >
                  <span className="chip-dot" />
                  低码率去块
                </button>
              </div>

              {/* Collapsible Advanced / Engineering Details */}
              <button
                type="button"
                className="accordion-trigger"
                onClick={() => setShowAdvanced((s) => !s)}
              >
                <span>高级参数（Client-{tier} · {activeModel.sizeKb} KB）</span>
                {showAdvanced ? <ChevronDown size={14} /> : <ChevronRight size={14} />}
              </button>

              {showAdvanced && (
                <div className="advanced-drawer">
                  {tier !== 'P' && (
                    <label className="toggle-row">
                      <span>启用 7D 动态先验</span>
                      <input
                        type="checkbox"
                        checked={usePrior}
                        onChange={(e) => setUsePrior(e.target.checked)}
                      />
                    </label>
                  )}
                  <div className="field-row" style={{ fontSize: 11 }}>
                    <span>核烘焙耗时</span>
                    <strong>{bakeMs.toFixed(2)} ms</strong>
                  </div>
                  <div className="prior-grid">
                    {PRIOR_LABELS.map((lbl, idx) => (
                      <span key={lbl}>
                        {lbl} <b>{prior7d[idx].toFixed(2)}</b>
                      </span>
                    ))}
                  </div>
                </div>
              )}
            </section>

            {/* Step 3: Export */}
            <section className="panel-card">
              <div className="panel-title">
                <span className="panel-title-left">
                  <Download size={13} /> 导出视频
                </span>
                <span className="hint-inline">
                  {Math.max(16, outputWidth & ~1)}×{Math.max(16, outputHeight & ~1)} MP4
                </span>
              </div>

              {isLongVideo && activeSource && (
                <div className="field-row">
                  <span>导出范围</span>
                  <div className="segmented">
                    <button
                      className={exportClipMode === 'first10s' ? 'active' : ''}
                      onClick={() => setExportClipMode('first10s')}
                    >
                      前 10 秒
                    </button>
                    <button
                      className={exportClipMode === 'full' ? 'active' : ''}
                      onClick={() => setExportClipMode('full')}
                    >
                      全部 ({Math.round(activeSource.durationSec)}s)
                    </button>
                  </div>
                </div>
              )}

              <div className="field-row">
                <span>画质码率</span>
                <div className="segmented">
                  {[
                    { v: 8, label: '标准' },
                    { v: 15, label: '高清' },
                    { v: 25, label: '极清' },
                  ].map((item) => (
                    <button
                      key={item.v}
                      className={exportBitrateMbps === item.v ? 'active' : ''}
                      onClick={() => setExportBitrateMbps(item.v)}
                      title={`${item.v} Mbps`}
                    >
                      {item.label}
                    </button>
                  ))}
                </div>
              </div>

              <button
                className="btn btn-primary"
                style={{ width: '100%', padding: '10px 14px' }}
                disabled={!engineReady || !activeSource || exporting || modelLoading}
                onClick={() => void handleStartExport()}
              >
                <Download size={15} />
                {exporting
                  ? '正在导出...'
                  : isLongVideo && exportClipMode === 'first10s'
                    ? '导出前 10 秒超分 MP4'
                    : '导出超分 MP4'}
              </button>

              {exportResult && !exporting && (
                <button
                  className="btn btn-subtle"
                  style={{ width: '100%' }}
                  onClick={() => setExportModalOpen(true)}
                >
                  查看上次导出预览 &amp; 下载
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
