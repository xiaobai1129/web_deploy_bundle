import { describe, expect, it } from 'vitest';
import {
  computeWorkingResolution,
  formatCodecDisplayName,
  recommendScaleForResolution,
} from '../videoProbe';

describe('videoProbe resolution and codec helpers', () => {
  it('caps short edge consistently for both portrait and landscape high-res videos', () => {
    // Portrait iPhone 1170x2532 screen recording -> 720p smooth mode
    const p720 = computeWorkingResolution(1170, 2532, 'auto_720p');
    expect(p720.wasDownscaled).toBe(true);
    expect(p720.workW).toBe(720);
    expect(p720.workH % 2).toBe(0);
    expect(p720.workH).toBe(1558);

    // Portrait iPhone 1170x2532 -> 540p smooth mode
    const p540 = computeWorkingResolution(1170, 2532, 'auto_540p');
    expect(p540.wasDownscaled).toBe(true);
    expect(p540.workW).toBe(540);
    expect(p540.workH).toBe(1168);

    // Landscape 4K -> 720p smooth mode
    const l720 = computeWorkingResolution(3840, 2160, 'auto_720p');
    expect(l720.workW).toBe(1280);
    expect(l720.workH).toBe(720);
    expect(l720.wasDownscaled).toBe(true);

    // Original mode preserves native dims
    const orig = computeWorkingResolution(1170, 2532, 'original');
    expect(orig).toEqual({ workW: 1170, workH: 2532, wasDownscaled: false });

    // Standard 540p video stays untouched in auto_720p mode
    const sd = computeWorkingResolution(960, 540, 'auto_720p');
    expect(sd).toEqual({ workW: 960, workH: 540, wasDownscaled: false });
  });

  it('recommends x2.0 for portrait 540p and x1.5 for 720p+ based on short edge', () => {
    expect(recommendScaleForResolution(540, 960)).toBe(2.0);
    expect(recommendScaleForResolution(960, 540)).toBe(2.0);
    expect(recommendScaleForResolution(720, 1558)).toBe(1.5);
    expect(recommendScaleForResolution(1280, 720)).toBe(1.5);
  });

  it('formats common codec identifiers cleanly', () => {
    expect(formatCodecDisplayName('hvc1.1.6.L150.90')).toBe('HEVC (H.265)');
    expect(formatCodecDisplayName('hev1.1.6.L120.90')).toBe('HEVC (H.265)');
    expect(formatCodecDisplayName('avc1.640028')).toBe('H.264 (AVC)');
    expect(formatCodecDisplayName('av01.0.08M.08')).toBe('AV1');
    expect(formatCodecDisplayName('vp09.00.41.08')).toBe('VP9');
  });
});
