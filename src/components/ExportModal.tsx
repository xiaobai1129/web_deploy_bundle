import { CheckCircle2, Download, Film, Loader2, X } from 'lucide-react';
import type { ExportProgress, ExportResult } from '../video/mp4Exporter';

interface ExportModalProps {
  open: boolean;
  exporting: boolean;
  progress: ExportProgress | null;
  result: ExportResult | null;
  error: string | null;
  onCancel: () => void;
  onClose: () => void;
  onDownload: (res: ExportResult) => void;
}

function formatSize(bytes: number): string {
  if (bytes >= 1024 * 1024) return `${(bytes / (1024 * 1024)).toFixed(2)} MB`;
  return `${(bytes / 1024).toFixed(1)} KB`;
}

export function ExportModal({
  open,
  exporting,
  progress,
  result,
  error,
  onCancel,
  onClose,
  onDownload,
}: ExportModalProps) {
  if (!open) return null;

  const pct =
    progress && progress.totalFrames > 0
      ? Math.min(100, Math.round((progress.doneFrames / progress.totalFrames) * 100))
      : 0;

  return (
    <div className="modal-backdrop" onClick={() => !exporting && onClose()}>
      <div className="modal-card" onClick={(e) => e.stopPropagation()}>
        <div className="modal-header">
          <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
            <Film size={18} color="#38bdf8" />
            <h3>{exporting ? '正在本地 WebGPU 导出超分 MP4' : result ? '超分 MP4 导出完成' : '导出状态'}</h3>
          </div>
          {!exporting && (
            <button className="btn btn-icon" onClick={onClose} aria-label="关闭">
              <X size={16} />
            </button>
          )}
        </div>

        {error && <div className="error-box">{error}</div>}

        {exporting && progress && (
          <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
            <div className="field-row">
              <span style={{ display: 'inline-flex', alignItems: 'center', gap: 6 }}>
                <Loader2 size={14} /> {progress.message}
              </span>
              <strong>{pct}%</strong>
            </div>
            <div className="progress-track">
              <div className="progress-fill" style={{ width: `${pct}%` }} />
            </div>
            <div className="field-row" style={{ fontSize: 11 }}>
              <span>处理速度: <strong>{progress.fps.toFixed(1)} fps</strong></span>
              <span>预计剩余: <strong>{progress.etaSec.toFixed(1)} s</strong></span>
            </div>
            <div style={{ display: 'flex', justifyContent: 'flex-end', marginTop: 4 }}>
              <button className="btn btn-danger" onClick={onCancel}>
                取消导出
              </button>
            </div>
          </div>
        )}

        {!exporting && result && (
          <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
            <div
              style={{
                display: 'flex',
                alignItems: 'center',
                gap: 8,
                color: '#34d399',
                fontSize: 13,
                fontWeight: 600,
              }}
            >
              <CheckCircle2 size={16} />
              已完成本地超分与 H.264 封装，全程零云端上传。
            </div>

            <video className="export-preview-video" src={result.url} controls playsInline loop />

            <div className="prior-grid">
              <span>文件名 <b>{result.fileName}</b></span>
              <span>文件大小 <b>{formatSize(result.blob.size)}</b></span>
              <span>输出分辨率 <b>{result.width}×{result.height}</b></span>
              <span>导出范围 <b>{result.clippedToSec !== null ? `前 ${result.clippedToSec} 秒片段` : '完整视频'}</b></span>
              <span>总帧数 <b>{result.frameCount} 帧</b></span>
              <span>总耗时 <b>{result.elapsedSec.toFixed(2)} s</b></span>
              <span>平均速度 <b>{result.avgFps.toFixed(1)} fps</b></span>
              <span>音轨状态 <b>{result.hasAudio ? '已保留原 AAC 音轨' : '无音轨 / 仅视频'}</b></span>
              <span>管线模式 <b>{result.pipelineUsed === 'webcodecs-mp4box' ? 'WebCodecs 硬解直通' : '兼容逐帧模式'}</b></span>
            </div>

            <div style={{ display: 'flex', justifyContent: 'flex-end', gap: 10 }}>
              <button className="btn" onClick={onClose}>
                返回对比工作台
              </button>
              <button className="btn btn-primary" onClick={() => onDownload(result)}>
                <Download size={15} />
                下载超分 MP4
              </button>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
