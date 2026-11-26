import {
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
  type DragEvent,
  type PointerEvent,
} from 'react';
import {
  ChevronLeft,
  ChevronRight,
  Columns2,
  Eye,
  EyeOff,
  Flame,
  Maximize2,
  Minimize2,
  MoveHorizontal,
  Pause,
  Play,
  Repeat,
  Upload,
  Volume2,
  VolumeX,
  ZoomIn,
} from 'lucide-react';
import type { CompareViewMode } from '../engine/webgpuRunner';

export type StageDisplayMode = 'fit' | 'native';

interface ComparisonViewportProps {
  canvasRef: React.RefObject<HTMLCanvasElement | null>;
  canvasWidth: number;
  canvasHeight: number;
  inputWidth: number;
  inputHeight: number;
  tierLabel: string;
  scale: number;
  viewMode: CompareViewMode;
  splitX: number;
  zoom: number;
  centerX: number;
  centerY: number;
  stageDisplayMode: StageDisplayMode;
  stageFillPct: number;
  inferenceMs: number;
  isPlaying: boolean;
  isMuted: boolean;
  isLoop: boolean;
  currentTime: number;
  duration: number;
  onViewModeChange: (m: CompareViewMode) => void;
  onSplitChange: (x: number) => void;
  onZoomChange: (z: number) => void;
  onCenterChange: (cx: number, cy: number) => void;
  onStageDisplayModeChange: (mode: StageDisplayMode) => void;
  onStageFillPctChange: (pct: number) => void;
  onDisplaySizeChange?: (width: number, height: number) => void;
  onTogglePlay: () => void;
  onStepFrame: (delta: number) => void;
  onSeek: (timeSec: number) => void;
  onToggleMute: () => void;
  onToggleLoop: () => void;
  onDropFile: (file: File) => void;
  onTriggerUpload: () => void;
}

function formatTime(sec: number): string {
  if (!Number.isFinite(sec) || sec < 0) return '00:00.0';
  const mins = Math.floor(sec / 60);
  const rem = sec - mins * 60;
  return `${String(mins).padStart(2, '0')}:${rem.toFixed(1).padStart(4, '0')}`;
}

export function ComparisonViewport({
  canvasRef,
  canvasWidth,
  canvasHeight,
  inputWidth,
  inputHeight,
  tierLabel,
  scale,
  viewMode,
  splitX,
  zoom,
  centerX,
  centerY,
  stageDisplayMode,
  stageFillPct,
  inferenceMs,
  isPlaying,
  isMuted,
  isLoop,
  currentTime,
  duration,
  onViewModeChange,
  onSplitChange,
  onZoomChange,
  onCenterChange,
  onStageDisplayModeChange,
  onStageFillPctChange,
  onDisplaySizeChange,
  onTogglePlay,
  onStepFrame,
  onSeek,
  onToggleMute,
  onToggleLoop,
  onDropFile,
  onTriggerUpload,
}: ComparisonViewportProps) {
  const stageRef = useRef<HTMLDivElement | null>(null);
  const frameRef = useRef<HTMLDivElement | null>(null);
  const dragModeRef = useRef<'none' | 'split' | 'pan'>('none');
  const panStartRef = useRef({ x: 0, y: 0, cx: 0.5, cy: 0.5 });
  const [isDraggingFile, setIsDraggingFile] = useState(false);
  const [stageSize, setStageSize] = useState({ w: 0, h: 0 });
  const [showOverlays, setShowOverlays] = useState(true);

  // Measure available stage container so even tiny 144p/144x144 videos can scale smoothly up to full stage
  useEffect(() => {
    const el = stageRef.current;
    if (!el) return;
    const measure = () => {
      const rect = el.getBoundingClientRect();
      setStageSize({ w: Math.max(0, rect.width), h: Math.max(0, rect.height) });
    };
    measure();
    if (typeof ResizeObserver === 'undefined') return;
    const ro = new ResizeObserver(measure);
    ro.observe(el);
    return () => ro.disconnect();
  }, []);

  const computedFrameSize = useMemo(() => {
    const safeOutW = Math.max(2, canvasWidth);
    const safeOutH = Math.max(2, canvasHeight);
    if (stageSize.w <= 0 || stageSize.h <= 0) {
      return null;
    }

    const pad = 28;
    const availW = Math.max(120, stageSize.w - pad);
    const availH = Math.max(120, stageSize.h - pad);
    const aspect = safeOutW / safeOutH;

    let fitW = availW;
    let fitH = fitW / aspect;
    if (fitH > availH) {
      fitH = availH;
      fitW = fitH * aspect;
    }

    if (stageDisplayMode === 'native') {
      const natW = Math.min(safeOutW, fitW);
      const natH = natW / aspect;
      return {
        w: Math.max(48, Math.round(natW)),
        h: Math.max(48, Math.round(natH)),
        nativeRatioPct: Math.round((natW / safeOutW) * 100),
      };
    }

    const pct = Math.min(100, Math.max(30, stageFillPct)) / 100;
    const w = Math.max(96, Math.round(fitW * pct));
    const h = Math.max(72, Math.round(fitH * pct));
    return {
      w,
      h,
      nativeRatioPct: Math.round((w / safeOutW) * 100),
    };
  }, [stageSize.w, stageSize.h, canvasWidth, canvasHeight, stageDisplayMode, stageFillPct]);

  useEffect(() => {
    if (computedFrameSize && onDisplaySizeChange) {
      onDisplaySizeChange(computedFrameSize.w, computedFrameSize.h);
    }
  }, [computedFrameSize, onDisplaySizeChange]);

  const updateSplitFromClientX = useCallback(
    (clientX: number) => {
      const el = frameRef.current;
      if (!el) return;
      const rect = el.getBoundingClientRect();
      if (rect.width <= 0) return;
      const next = Math.min(0.98, Math.max(0.02, (clientX - rect.left) / rect.width));
      onSplitChange(next);
    },
    [onSplitChange],
  );

  const onPointerDown = (e: PointerEvent<HTMLDivElement>) => {
    const el = frameRef.current;
    if (!el) return;
    const rect = el.getBoundingClientRect();
    const relX = (e.clientX - rect.left) / Math.max(1, rect.width);

    const nearDivider = viewMode === 0 && Math.abs(relX - splitX) < 0.05;
    if (zoom > 1.0 && !nearDivider) {
      dragModeRef.current = 'pan';
      panStartRef.current = { x: e.clientX, y: e.clientY, cx: centerX, cy: centerY };
    } else if (viewMode === 0) {
      dragModeRef.current = 'split';
      updateSplitFromClientX(e.clientX);
    } else {
      return;
    }
    e.currentTarget.setPointerCapture(e.pointerId);
  };

  const onPointerMove = (e: PointerEvent<HTMLDivElement>) => {
    if (dragModeRef.current === 'split') {
      updateSplitFromClientX(e.clientX);
    } else if (dragModeRef.current === 'pan') {
      const el = frameRef.current;
      if (!el) return;
      const rect = el.getBoundingClientRect();
      const dx = (e.clientX - panStartRef.current.x) / Math.max(1, rect.width) / zoom;
      const dy = (e.clientY - panStartRef.current.y) / Math.max(1, rect.height) / zoom;
      const halfSpan = 0.5 / zoom;
      const nx = Math.min(1 - halfSpan, Math.max(halfSpan, panStartRef.current.cx - dx));
      const ny = Math.min(1 - halfSpan, Math.max(halfSpan, panStartRef.current.cy - dy));
      onCenterChange(nx, ny);
    }
  };

  const onPointerUp = (e: PointerEvent<HTMLDivElement>) => {
    if (dragModeRef.current !== 'none') {
      dragModeRef.current = 'none';
      try {
        e.currentTarget.releasePointerCapture(e.pointerId);
      } catch {
        // ignore
      }
    }
  };

  const onDragOver = (e: DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    if (!isDraggingFile) setIsDraggingFile(true);
  };

  const onDragLeave = (e: DragEvent<HTMLDivElement>) => {
    if (e.currentTarget.contains(e.relatedTarget as Node | null)) return;
    setIsDraggingFile(false);
  };

  const onDrop = (e: DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setIsDraggingFile(false);
    const file = e.dataTransfer.files?.[0];
    if (file && file.type.startsWith('video/')) {
      onDropFile(file);
    }
  };

  const estFps = inferenceMs > 0.1 ? Math.min(240, 1000 / inferenceMs) : 0;

  return (
    <div className="viewport-col">
      {/* Top Viewport Controls */}
      <div className="viewport-toolbar">
        <div className="segmented" role="tablist" aria-label="对比视图">
          <button className={viewMode === 0 ? 'active' : ''} onClick={() => onViewModeChange(0)}>
            <Columns2 size={13} /> 分屏对比
          </button>
          <button className={viewMode === 1 ? 'active' : ''} onClick={() => onViewModeChange(1)}>
            <Eye size={13} /> 仅看修复
          </button>
          <button className={viewMode === 2 ? 'active' : ''} onClick={() => onViewModeChange(2)}>
            仅看原画
          </button>
          <button className={viewMode === 3 ? 'active' : ''} onClick={() => onViewModeChange(3)}>
            <Flame size={13} /> 差异图
          </button>
        </div>

        <div className="toolbar-right-group">
          {/* Adjustable Video Frame Display Size (solves tiny 144p / 144x144 display!) */}
          <div className="size-control-pill" title="调节视频在画布中的显示大小（不影响实际导出分辨率）">
            <span className="size-control-label">画幅</span>
            <button
              className={`mini-chip-btn ${stageDisplayMode === 'native' ? 'active' : ''}`}
              onClick={() => onStageDisplayModeChange('native')}
              title="按视频实际输出像素 1:1 显示"
            >
              <Minimize2 size={11} />
              1:1
            </button>
            <input
              type="range"
              className="mini-slider"
              min={30}
              max={100}
              step={2}
              value={stageDisplayMode === 'native' ? 35 : stageFillPct}
              onChange={(e) => {
                onStageDisplayModeChange('fit');
                onStageFillPctChange(Number(e.target.value));
              }}
              aria-label="画面显示大小"
            />
            <button
              className={`mini-chip-btn ${stageDisplayMode === 'fit' && stageFillPct >= 98 ? 'active' : ''}`}
              onClick={() => {
                onStageDisplayModeChange('fit');
                onStageFillPctChange(100);
              }}
              title="自动放大并铺满整个预览区域"
            >
              <Maximize2 size={11} />
              铺满
            </button>
            {computedFrameSize && (
              <span className="size-ratio-badge" title={`显示尺寸 ${computedFrameSize.w}×${computedFrameSize.h}`}>
                {computedFrameSize.nativeRatioPct}%
              </span>
            )}
          </div>

          {/* Detail Loupe Zoom */}
          <div className="segmented" aria-label="局部细节放大">
            {[1, 2, 4].map((z) => (
              <button
                key={z}
                className={zoom === z ? 'active' : ''}
                onClick={() => {
                  onZoomChange(z);
                  if (z === 1) onCenterChange(0.5, 0.5);
                }}
                title={z === 1 ? '恢复正常视角' : `局部细节放大 ${z} 倍（可拖拽画面查看边缘）`}
              >
                {z === 1 ? <ZoomIn size={12} /> : null}
                {z}×
              </button>
            ))}
          </div>

          {/* Toggle floating tags & split handle visibility */}
          <button
            type="button"
            className={`btn overlay-toggle-btn ${showOverlays ? '' : 'active'}`}
            onClick={() => setShowOverlays((v) => !v)}
            title={showOverlays ? '隐藏画面上的对比标签与圆点图标' : '显示画面上的对比标签与圆点图标'}
          >
            {showOverlays ? <EyeOff size={13} /> : <Eye size={13} />}
            {showOverlays ? '隐藏提示' : '显示提示'}
          </button>
        </div>
      </div>

      {/* Main Comparison Stage */}
      <div
        ref={stageRef}
        className={`stage-shell ${isDraggingFile ? 'dragging' : ''}`}
        onDragOver={onDragOver}
        onDragLeave={onDragLeave}
        onDrop={onDrop}
      >
        <div
          ref={frameRef}
          className="canvas-frame"
          style={
            computedFrameSize
              ? {
                  width: `${computedFrameSize.w}px`,
                  height: `${computedFrameSize.h}px`,
                  cursor: zoom > 1 ? 'grab' : viewMode === 0 ? 'ew-resize' : 'default',
                }
              : {
                  aspectRatio: `${Math.max(1, canvasWidth)} / ${Math.max(1, canvasHeight)}`,
                  cursor: zoom > 1 ? 'grab' : viewMode === 0 ? 'ew-resize' : 'default',
                }
          }
          onPointerDown={onPointerDown}
          onPointerMove={onPointerMove}
          onPointerUp={onPointerUp}
        >
          <canvas
            ref={canvasRef}
            width={Math.max(2, canvasWidth)}
            height={Math.max(2, canvasHeight)}
            className="sr-canvas"
          />

          {viewMode === 0 && (
            <>
              {showOverlays && (
                <>
                  <div className="corner-tag left">原画</div>
                  <div className="corner-tag right">
                    超分修复 · {tierLabel} (×{scale})
                  </div>
                </>
              )}
              <div
                className={`split-line ${showOverlays ? '' : 'subtle'}`}
                style={{ left: `${splitX * 100}%` }}
              >
                {showOverlays && (
                  <div className="split-handle">
                    <MoveHorizontal size={15} />
                  </div>
                )}
              </div>
            </>
          )}

          {showOverlays && viewMode === 1 && (
            <div className="corner-tag right">超分修复 · {tierLabel} (×{scale})</div>
          )}
          {showOverlays && viewMode === 2 && <div className="corner-tag left">原画（双三次插值）</div>}
          {showOverlays && viewMode === 3 && (
            <div className="corner-tag right">修复差异热力图 ×10</div>
          )}

          {showOverlays && zoom > 1 && (
            <div className="zoom-hint">
              {zoom}× 细节放大 · 拖拽画面可平移
            </div>
          )}
        </div>

        {/* Subtle Floating Resolution & Speed Status */}
        {showOverlays && (
          <div className="stage-meta-pill">
            <span>
              <b>{inputWidth}×{inputHeight}</b> → <b>{canvasWidth}×{canvasHeight}</b>
            </span>
            {inferenceMs > 0 && (
              <>
                <span className="meta-sep">·</span>
                <span>
                  <b>{inferenceMs.toFixed(1)} ms</b>
                  {estFps > 0 ? ` (${estFps.toFixed(0)} fps)` : ''}
                </span>
              </>
            )}
          </div>
        )}

        {isDraggingFile && (
          <div className="drop-overlay">
            <Upload size={34} />
            <div>松开鼠标，立即在本地超分此视频</div>
          </div>
        )}
      </div>

      {/* Bottom Playback Bar */}
      <div className="transport-bar">
        <button className="btn btn-primary btn-icon" onClick={onTogglePlay} title={isPlaying ? '暂停' : '播放'}>
          {isPlaying ? <Pause size={15} /> : <Play size={15} />}
        </button>
        <button className="btn btn-icon" onClick={() => onStepFrame(-1)} title="上一帧">
          <ChevronLeft size={15} />
        </button>
        <button className="btn btn-icon" onClick={() => onStepFrame(1)} title="下一帧">
          <ChevronRight size={15} />
        </button>

        <span className="time-readout">
          {formatTime(currentTime)} / {formatTime(duration)}
        </span>

        <input
          type="range"
          className="timeline-range"
          min={0}
          max={Math.max(0.01, duration)}
          step={0.01}
          value={Math.min(currentTime, duration)}
          onChange={(e) => onSeek(Number(e.target.value))}
          aria-label="播放进度"
        />

        <button
          className={`btn btn-icon ${isLoop ? 'btn-active-icon' : ''}`}
          onClick={onToggleLoop}
          title={isLoop ? '循环播放：开' : '循环播放：关'}
        >
          <Repeat size={14} />
        </button>
        <button
          className="btn btn-icon"
          onClick={onToggleMute}
          title={isMuted ? '取消静音' : '静音'}
        >
          {isMuted ? <VolumeX size={14} /> : <Volume2 size={14} />}
        </button>
        <button className="btn" onClick={onTriggerUpload}>
          <Upload size={14} /> 换个视频
        </button>
      </div>
    </div>
  );
}
