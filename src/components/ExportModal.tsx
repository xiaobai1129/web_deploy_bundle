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
            <h3>{exporting ? '正在本地导出超分视频' : result ? '超分视频已就绪' : '导出状态'}</h3>
          </div>
          {!exporting && (
            <button className="btn btn-icon" onClick={onClose} aria-label="关闭">
              <X size={16} />
            </button>
          )}
        </div>

        {error && <div className="error-box">{error}</div>}

        {exporting && progress && (
          <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
            <div className="field-row">
              <span style={{ display: 'inline-flex', alignItems: 'center', gap: 7, color: 'var(--text-primary)' }}>
                <Loader2 size={15} className="spin" color="#38bdf8" /> {progress.message}
              </span>
              <strong>{pct}%</strong>
            </div>
            <div className="progress-track">
              <div className="progress-fill" style={{ width: `${pct}%` }} />
            </div>
            <div className="field-row" style={{ fontSize: 12 }}>
              <span>处理速度 <strong>{progress.fps.toFixed(1)} fps</strong></span>
              <span>预计剩余 <strong>{progress.etaSec.toFixed(1)} 秒</strong></span>
            </div>
            <div style={{ display: 'flex', justifyContent: 'flex-end', marginTop: 4 }}>
              <button className="btn btn-danger" onClick={onCancel}>
                取消导出
              </button>
            </div>
          </div>
        )}

        {!exporting && result && (
          <div style={{ display: 'flex', flexDirection: 'column', gap: 14 }}>
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
              已完成本地超分与标准 MP4 封装，可直接预览或下载发送。
            </div>

            <video className="export-preview-video" src={result.url} controls playsInline loop />

            <div className="export-stats-grid">
              <div className="stat-item">
                <span>输出分辨率</span>
                <b>{result.width}×{result.height}</b>
              </div>
              <div className="stat-item">
                <span>文件大小</span>
                <b>{formatSize(result.blob.size)}</b>
              </div>
              <div className="stat-item">
                <span>总帧数 / 速度</span>
                <b>{result.frameCount} 帧 · {result.avgFps.toFixed(0)} fps</b>
              </div>
              <div className="stat-item">
                <span>音轨与范围</span>
                <b>
                  {result.hasAudio ? '保留原音轨' : '仅视频'} ·{' '}
                  {result.clippedToSec !== null ? `前 ${result.clippedToSec}s` : '完整'}
                </b>
              </div>
            </div>

            <div style={{ display: 'flex', justifyContent: 'flex-end', gap: 10 }}>
              <button className="btn" onClick={onClose}>
                返回工作台
              </button>
              <button className="btn btn-primary" onClick={() => onDownload(result)}>
                <Download size={15} />
                保存到本地 MP4
              </button>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
