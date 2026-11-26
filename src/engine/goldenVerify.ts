import { bakeCsr1ToSsbo, parseBinaryBlob } from '../core/csr1';
import type { CodecSREngine } from './webgpuRunner';

export interface GoldenCaseResult {
  caseId: string;
  weightFile: string;
  inputShape: [number, number];
  outputShape: [number, number];
  bakeMs: number;
  gpuMs: number;
  psnrDb: number;
  maxAbsErr: number;
  passed: boolean;
}

interface GoldenManifestCase {
  case_id: string;
  weight_file: string;
  input_file: string;
  input_shape: [number, number];
  prior_7d: number[];
  expected_output_file: string;
  expected_output_shape: [number, number];
}

function computePsnr(a: Float32Array, b: Float32Array): { psnrDb: number; maxAbsErr: number } {
  let mse = 0;
  let maxAbs = 0;
  const n = Math.min(a.length, b.length);
  for (let i = 0; i < n; i++) {
    const d = Math.abs(a[i] - b[i]);
    if (d > maxAbs) maxAbs = d;
    mse += d * d;
  }
  mse /= Math.max(1, n);
  const psnrDb = mse <= 1e-12 ? 99.99 : 10 * Math.log10(1.0 / mse);
  return { psnrDb, maxAbsErr: maxAbs };
}

/**
 * Runs all 6 official PyTorch golden test vectors directly on the local WebGPU device.
 * Acceptance threshold: PSNR >= 70.0 dB and max_abs_err < 0.002.
 */
export async function runGoldenVerificationSuite(
  engine: CodecSREngine,
  onCaseDone?: (res: GoldenCaseResult, index: number, total: number) => void,
): Promise<GoldenCaseResult[]> {
  const manifestResp = await fetch('/golden_vectors/golden_manifest.json');
  if (!manifestResp.ok) {
    throw new Error(`无法加载 golden_manifest.json (${manifestResp.status})`);
  }
  const manifest = (await manifestResp.json()) as { cases: GoldenManifestCase[] };
  const results: GoldenCaseResult[] = [];

  for (let i = 0; i < manifest.cases.length; i++) {
    const c = manifest.cases[i];
    const [weightResp, inResp, refResp] = await Promise.all([
      fetch(`/weights/${c.weight_file}`),
      fetch(`/golden_vectors/${c.input_file}`),
      fetch(`/golden_vectors/${c.expected_output_file}`),
    ]);
    const [weightBuf, inBuf, refBuf] = await Promise.all([
      weightResp.arrayBuffer(),
      inResp.arrayBuffer(),
      refResp.arrayBuffer(),
    ]);

    const parsed = parseBinaryBlob(weightBuf);
    const baked = bakeCsr1ToSsbo(parsed, c.prior_7d);
    const inputF32 = new Float32Array(inBuf);
    const refF32 = new Float32Array(refBuf);

    const t0 = performance.now();
    const gotF32 = await engine.runLumaPatch(inputF32, c.input_shape[0], c.input_shape[1], baked);
    const gpuMs = performance.now() - t0;

    const { psnrDb, maxAbsErr } = computePsnr(gotF32, refF32);
    const res: GoldenCaseResult = {
      caseId: c.case_id,
      weightFile: c.weight_file,
      inputShape: c.input_shape,
      outputShape: c.expected_output_shape,
      bakeMs: baked.bakeMs,
      gpuMs,
      psnrDb,
      maxAbsErr,
      passed: psnrDb >= 70.0 && maxAbsErr < 0.002,
    };
    results.push(res);
    onCaseDone?.(res, i, manifest.cases.length);
  }
  return results;
}
