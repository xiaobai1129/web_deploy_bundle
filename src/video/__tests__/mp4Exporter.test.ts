import { describe, expect, it } from 'vitest';
import {
  computeMuxSchedule,
  detectInputHasBFrames,
  selectAvcCandidatesForResolution,
  type BufferedEncodedChunk,
} from '../mp4Exporter';

function mkChunk(timestampUs: number, type: 'key' | 'delta' = 'delta', durationUs = 40_000): BufferedEncodedChunk {
  return {
    data: new Uint8Array([1, 2, 3]),
    type,
    timestampUs,
    durationUs,
  };
}

describe('mp4Exporter B-frame & timestamp scheduling helpers', () => {
  it('detects whether input MP4 samples contain B-frame reordering', () => {
    expect(detectInputHasBFrames([{ cts: 0 }, { cts: 512 }, { cts: 1024 }])).toBe(false);
    expect(detectInputHasBFrames([{ cts: 512 }, { cts: 2048 }, { cts: 1024 }, { cts: 1536 }])).toBe(true);
  });

  it('keeps zero compositionTimeOffset for monotonic realtime-encoded chunks', () => {
    const chunks = [mkChunk(0, 'key'), mkChunk(40_000), mkChunk(80_000), mkChunk(120_000)];
    const scheduled = computeMuxSchedule(chunks, 40_000);

    expect(scheduled.map((c) => c.presentationTimestampUs)).toEqual([0, 40_000, 80_000, 120_000]);
    expect(scheduled.every((c) => c.compositionTimeOffsetUs === 0)).toBe(true);
  });

  it('reconstructs monotonic DTS and non-negative CTTS when encoder emits B-frames out of order', () => {
    // Typical I, P, B, B, P, B, B decode-order PTS sequence at 25fps (40ms step)
    const chunks = [
      mkChunk(0, 'key'),
      mkChunk(120_000),
      mkChunk(40_000),
      mkChunk(80_000),
      mkChunk(240_000),
      mkChunk(160_000),
      mkChunk(200_000),
    ];
    const scheduled = computeMuxSchedule(chunks, 40_000);

    // Every DTS (= PTS - CTO) must be strictly increasing and CTO >= 0
    const dtsList = scheduled.map((c) => c.presentationTimestampUs - c.compositionTimeOffsetUs);
    expect(dtsList).toEqual([0, 40_000, 80_000, 120_000, 160_000, 200_000, 240_000]);
    expect(scheduled.every((c) => c.compositionTimeOffsetUs >= 0)).toBe(true);

    // Relative presentation order must preserve the original frame sequence (0, 3, 1, 2, 6, 4, 5)
    const minPts = Math.min(...scheduled.map((c) => c.presentationTimestampUs));
    const normalizedPts = scheduled.map((c) => c.presentationTimestampUs - minPts);
    expect(normalizedPts).toEqual([0, 120_000, 40_000, 80_000, 240_000, 160_000, 200_000]);
  });

  it('selects resolution-appropriate H.264 levels', () => {
    expect(selectAvcCandidatesForResolution(384, 288)[0]).toBe('avc1.640028');
    expect(selectAvcCandidatesForResolution(1920, 1080)[0]).toBe('avc1.64002a');
    expect(selectAvcCandidatesForResolution(3840, 2160)[0]).toBe('avc1.640034');
  });
});
