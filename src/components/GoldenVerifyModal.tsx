import { CheckCircle2, Loader2, ShieldCheck, X, XCircle } from 'lucide-react';
import type { GoldenCaseResult } from '../engine/goldenVerify';

interface GoldenVerifyModalProps {
  open: boolean;
  running: boolean;
  results: GoldenCaseResult[];
  error: string | null;
  onRun: () => void;
  onClose: () => void;
}

export function GoldenVerifyModal({
  open,
  running,
  results,
  error,
  onRun,
  onClose,
}: GoldenVerifyModalProps) {
  if (!open) return null;
  const allPassed = results.length === 6 && results.every((r) => r.passed);

  return (
    <div className="modal-backdrop" onClick={onClose}>
      <div className="modal-card" onClick={(e) => e.stopPropagation()}>
        <div className="modal-header">
          <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
            <ShieldCheck size={18} color="#38bdf8" />
            <h3>WebGPU 像素级精度对拍（Golden Vectors）</h3>
          </div>
          <button className="btn btn-icon" onClick={onClose} aria-label="关闭">
            <X size={16} />
          </button>
        </div>

        <div style={{ fontSize: 12, color: '#94a3b8', lineHeight: 1.6 }}>
          在当前本地 GPU 上运行全部 6 组正式版模型（L/S/P × 540p/720p），与 PyTorch <code>baked_fp16</code> 标准输出逐像素比对。验收标准：<strong>PSNR ≥ 70.0 dB</strong> 且最大误差 <strong>&lt; 0.002</strong>。
        </div>

        {error && <div className="error-box">{error}</div>}

        {results.length > 0 && (
          <table className="data-table">
            <thead>
              <tr>
                <th>用例</th>
                <th>输出尺寸</th>
                <th>烘焙耗时</th>
                <th>GPU 耗时</th>
                <th>PSNR</th>
                <th>最大误差</th>
                <th>状态</th>
              </tr>
            </thead>
            <tbody>
              {results.map((r) => (
                <tr key={r.caseId}>
                  <td><code>{r.caseId}</code></td>
                  <td>{r.outputShape[1]}×{r.outputShape[0]}</td>
                  <td>{r.bakeMs.toFixed(2)} ms</td>
                  <td>{r.gpuMs.toFixed(2)} ms</td>
                  <td style={{ color: r.passed ? '#34d399' : '#fb7185', fontWeight: 700 }}>
                    {r.psnrDb.toFixed(2)} dB
                  </td>
                  <td>{r.maxAbsErr.toExponential(2)}</td>
                  <td>
                    {r.passed ? (
                      <span style={{ display: 'inline-flex', alignItems: 'center', gap: 4, color: '#34d399' }}>
                        <CheckCircle2 size={14} /> PASS
                      </span>
                    ) : (
                      <span style={{ display: 'inline-flex', alignItems: 'center', gap: 4, color: '#fb7185' }}>
                        <XCircle size={14} /> FAIL
                      </span>
                    )}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}

        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginTop: 4 }}>
          <div style={{ fontSize: 12, color: allPassed ? '#34d399' : '#94a3b8' }}>
            {running
              ? `正在本地 WebGPU 依次校验 (${results.length}/6)...`
              : allPassed
                ? '6/6 全部通过！当前 WebGPU 推理引擎与论文正式版 100% 数学等价。'
                : '点击右侧按钮在本地 GPU 上开始校验。'}
          </div>
          <button className="btn btn-primary" disabled={running} onClick={onRun}>
            {running ? <Loader2 size={14} className="spin" /> : <ShieldCheck size={14} />}
            {running ? '校验中...' : results.length > 0 ? '重新运行对拍' : '开始 6 模型对拍'}
          </button>
        </div>
      </div>
    </div>
  );
}
