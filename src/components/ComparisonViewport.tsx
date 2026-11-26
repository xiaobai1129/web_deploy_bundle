import { useCallback, useRef, useState, type DragEvent, type PointerEvent } from 'react';
import {
  ChevronLeft,
  ChevronRight,
  Columns2,
  Eye,
  Flame,
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
  inferenceMs: number;
  bakeMs: number;
  isPlaying: boolean;
  isMuted: boolean;
  isLoop: boolean;
  currentTime: number;
  duration: number;
  onViewModeChange: (m: CompareViewMode) => void;
  onSplitChange: (x: number) => void;
  onZoomChange: (z: number) => void;
  onCenterChange: (cx: number, cy: number) => void;
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
  inferenceMs,
  bakeMs,
  isPlaying,
  isMuted,
  isLoop,
  currentTime,
  duration,
  onViewModeChange,
  onSplitChange,
  onZoomChange,
  onCenterChange,
  onTogglePlay,
  onStepFrame,
  onSeek,
  onToggleMute,
  onToggleLoop,
  onDropFile,
  onTriggerUpload,
}: ComparisonViewportProps) {
  const frameRef = useRef<HTMLDivElement | null>(null);
  const dragModeRef = useRef<'none' | 'split' | 'pan'>('none');
  const panStartRef = useRef({ x: 0, y: 0, cx: 0.5, cy: 0.5 });
  const [isDraggingFile, setIsDraggingFile] = useState(false);

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

    // If zoomed in and not clicking near the split divider, allow dragging to pan inspection area
    const nearDivider = viewMode === 0 && Math.abs(relX - splitX) < 0.045;
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
      <div className="viewport-toolbar">
        <div style={{ display: 'flex', alignItems: 'center', gap: 10, flexWrap: 'wrap' }}>
          <div className="segmented" role="tablist" aria-label="对比显示模式">
            <button className={viewMode === 0 ? 'active' : ''} onClick={() => onViewModeChange(0)}>
              <Columns2 size={13} /> 分屏对比
            </button>
            <button className={viewMode === 1 ? 'active' : ''} onClick={() => onViewModeChange(1)}>
              <Eye size={13} /> 仅看超分
            </button>
            <button className={viewMode === 2 ? 'active' : ''} onClick={() => onViewModeChange(2)}>
              原画 (Bicubic)
            </button>
            <button className={viewMode === 3 ? 'active' : ''} onClick={() => onViewModeChange(3)}>
              <Flame size={13} /> 差异热力图
            </button>
          </div>

          <div className="segmented" aria-label="细节放大倍率">
            {[1, 2, 4].map((z) => (
              <button
                key={z}
                className={zoom === z ? 'active' : ''}
                onClick={() => {
                  onZoomChange(z);
                  if (z === 1) onCenterChange(0.5, 0.5);
                }}
              >
                {z === 1 ? <ZoomIn size={12} /> : null}
                {z}×
              </button>
            ))}
          </div>
        </div>

        <div className="telemetry-pills">
          <span className="telemetry-pill">
            <strong>{inputWidth}×{inputHeight}</strong> → <strong>{canvasWidth}×{canvasHeight}</strong>
          </span>
          <span className="telemetry-pill">
            推理 <strong>{inferenceMs > 0 ? `${inferenceMs.toFixed(1)} ms` : '--'}</strong>
            {estFps > 0 ? ` (${estFps.toFixed(0)} fps)` : ''}
          </span>
          <span className="telemetry-pill">
            核烘焙 <strong>{bakeMs > 0 ? `${bakeMs.toFixed(2)} ms` : '--'}</strong>
          </span>
        </div>
      </div>

      <div
        className={`stage-shell ${isDraggingFile ? 'dragging' : ''}`}
        onDragOver={onDragOver}
        onDragLeave={onDragLeave}
        onDrop={onDrop}
      >
        <div
          ref={frameRef}
          className="canvas-frame"
          style={{
            aspectRatio: `${Math.max(1, canvasWidth)} / ${Math.max(1, canvasHeight)}`,
            cursor: zoom > 1 ? 'grab' : viewMode === 0 ? 'ew-resize' : 'default',
          }}
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
              <div className="corner-tag left">Bicubic 原画双三次插值</div>
              <div className="corner-tag right">
                CodecSR V3.5 · {tierLabel} (×{scale})
              </div>
              <div className="split-line" style={{ left: `${splitX * 100}%` }}>
                <div className="split-handle">
                  <MoveHorizontal size={16} />
                </div>
              </div>
            </>
          )}

          {viewMode === 1 && (
            <div className="corner-tag right">CodecSR V3.5 · {tierLabel} (×{scale})</div>
          )}
          {viewMode === 2 && <div className="corner-tag left">Bicubic 原画基线</div>}
          {viewMode === 3 && (
            <div className="corner-tag right">残差恢复热力图 |SR − Bicubic| × 10</div>
          )}

          {zoom > 1 && (
            <div className="zoom-hint">
              已启用 {zoom}× 细节放大 · 在画面空白处拖拽可平移检查边缘与文字细节
            </div>
          )}
        </div>

        {isDraggingFile && (
          <div className="drop-overlay">
            <Upload size={34} />
            <div>松开鼠标，立即在本地 WebGPU 上超分此视频</div>
          </div>
        )}
      </div>

      <div className="transport-bar">
        <button className="btn btn-primary btn-icon" onClick={onTogglePlay} title={isPlaying ? '暂停' : '播放'}>
          {isPlaying ? <Pause size={16} /> : <Play size={16} />}
        </button>
        <button className="btn btn-icon" onClick={() => onStepFrame(-1)} title="上一帧">
          <ChevronLeft size={16} />
        </button>
        <button className="btn btn-icon" onClick={() => onStepFrame(1)} title="下一帧">
          <ChevronRight size={16} />
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
          className="btn btn-icon"
          onClick={onToggleLoop}
          title={isLoop ? '循环播放：开' : '循环播放：关'}
          style={{ color: isLoop ? '#38bdf8' : undefined }}
        >
          <Repeat size={15} />
        </button>
        <button
          className="btn btn-icon"
          onClick={onToggleMute}
          title={isMuted ? '取消静音' : '静音'}
        >
          {isMuted ? <VolumeX size={15} /> : <Volume2 size={15} />}
        </button>
        <button className="btn" onClick={onTriggerUpload}>
          <Upload size={14} /> 上传视频
        </button>
      </div>
    </div>
  );
}
