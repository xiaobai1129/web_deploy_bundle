import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { describe, expect, it } from 'vitest';
import { bakeCsr1ToSsbo, encodeStreamPrior7d, fp16ToFp32, parseBinaryBlob } from '../csr1';

interface GoldenCase {
  case_id: string;
  weight_file: string;
  prior_7d: number[];
  baked_ssbo_file: string;
  baked_ssbo_nbytes: number;
  baked_ssbo_half_offsets: Record<string, number>;
  baked_scalars: {
    skip_alpha: number;
    skip_beta: number;
    skip_hr: number;
    trust_delta: number;
  };
}

const bundleRoot = resolve(__dirname, '../../..');

function readArrayBuffer(relPath: string): ArrayBuffer {
  const buf = readFileSync(resolve(bundleRoot, relPath));
  return buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength);
}

describe('CSR1 parser and dynamic FP16 kernel baker', () => {
  const manifest = JSON.parse(
    readFileSync(resolve(bundleRoot, 'golden_vectors/golden_manifest.json'), 'utf-8'),
  ) as { cases: GoldenCase[] };

  it('matches all 6 PyTorch golden baked SSBO vectors within 1 FP16 ULP (< 2e-4)', () => {
    expect(manifest.cases.length).toBe(6);

    for (const c of manifest.cases) {
      const rawWeight = readArrayBuffer(`weights/${c.weight_file}`);
      const parsed = parseBinaryBlob(rawWeight);
      const baked = bakeCsr1ToSsbo(parsed, c.prior_7d);

      expect(baked.halfOffsets).toEqual(c.baked_ssbo_half_offsets);
      expect(baked.ssboBytes.byteLength).toBe(c.baked_ssbo_nbytes);

      const refBuf = readArrayBuffer(`golden_vectors/${c.baked_ssbo_file}`);
      const refU16 = new Uint16Array(refBuf);
      const gotU16 = new Uint16Array(
        baked.ssboBytes.buffer,
        baked.ssboBytes.byteOffset,
        baked.ssboBytes.byteLength >>> 1,
      );
      expect(gotU16.length).toBe(refU16.length);

      let maxAbsDiff = 0;
      let maxUlpDiff = 0;
      let sumAbsDiff = 0;
      let maxIdx = -1;
      for (let i = 0; i < refU16.length; i++) {
        const ulpDiff = Math.abs(gotU16[i] - refU16[i]);
        if (ulpDiff > maxUlpDiff) maxUlpDiff = ulpDiff;
        const diff = Math.abs(fp16ToFp32(gotU16[i]) - fp16ToFp32(refU16[i]));
        sumAbsDiff += diff;
        if (diff > maxAbsDiff) {
          maxAbsDiff = diff;
          maxIdx = i;
        }
      }
      const meanAbsDiff = sumAbsDiff / refU16.length;
      expect(maxUlpDiff, `${c.case_id} max FP16 ULP diff`).toBeLessThanOrEqual(1);
      expect(meanAbsDiff, `${c.case_id} mean FP16 abs diff`).toBeLessThan(1e-6);
      expect(
        maxAbsDiff,
        `${c.case_id} max FP16 diff at idx=${maxIdx} (got=${fp16ToFp32(gotU16[maxIdx])}, ref=${fp16ToFp32(refU16[maxIdx])})`,
      ).toBeLessThan(5e-4);

      expect(Math.abs(baked.scalars.trustDelta - c.baked_scalars.trust_delta)).toBeLessThan(5e-5);
      expect(baked.scalars.skipAlpha).toBeCloseTo(c.baked_scalars.skip_alpha, 6);
      expect(baked.scalars.skipBeta).toBeCloseTo(c.baked_scalars.skip_beta, 6);
      expect(baked.scalars.skipHr).toBeCloseTo(c.baked_scalars.skip_hr, 6);
    }
  });

  it('encodes demo_manifest 7D prior vectors identically to reference', () => {
    const demoManifest = JSON.parse(
      readFileSync(resolve(bundleRoot, 'demo_videos/demo_manifest.json'), 'utf-8'),
    ) as {
      demos: Array<{
        width: number;
        height: number;
        fps: number;
        bitrate_kbps: number;
        segment_qp_mean: number;
        codec: string;
        mode: string;
        is_screen_content: boolean;
        prior_7d: number[];
      }>;
    };

    for (const d of demoManifest.demos) {
      const encoded = encodeStreamPrior7d({
        width: d.width,
        height: d.height,
        fps: d.fps,
        bitrateKbps: d.bitrate_kbps,
        qpMean: d.segment_qp_mean,
        codec: d.codec,
        isLive: d.mode === 'live',
        isScreenContent: d.is_screen_content,
        hasPrior: true,
      });
      for (let i = 0; i < 7; i++) {
        expect(encoded[i]).toBeCloseTo(d.prior_7d[i], 5);
      }
    }
  });
});
