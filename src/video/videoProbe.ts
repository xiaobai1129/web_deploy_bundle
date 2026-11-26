import * as MP4Box from 'mp4box';
import { estimateQpFromVideo, type ModelScale } from '../core/csr1';

export type InputResMode = 'auto_720p' | 'auto_540p' | 'original';

export interface DemoManifestItem {
  file: string;
  source_content_id: string;
  description: string;
  width: number;
  height: number;
  target_width: number;
  target_height: number;
  scale: number;
  fps: number;
  frame_count: number;
  duration_sec: number;
  codec: string;
  mode: 'vod' | 'live';
  crf: number;
  bitrate_kbps: number;
  segment_qp_mean: number;
  is_screen_content: boolean;
  file_size_kb: number;
  prior_7d: number[];
}

export interface VideoSourceMeta {
  id: string;
  name: string;
  description: string;
  isDemo: boolean;
  url: string;
  file?: File;
  width: number;
  height: number;
  fps: number;
  durationSec: number;
  frameCount: number;
  bitrateKbps: number;
  codec: string;
  isRotated: boolean;
  isHighRes: boolean;
  defaultInputMode: InputResMode;
  defaultQp: number;
  defaultIsLive: boolean;
  defaultIsScreen: boolean;
  recommendedScale: ModelScale;
  fileSizeBytes: number;
  presetPrior7d?: number[];
}

/**
 * Computes the working GPU input resolution (always even >= 16) given original video dims and mode.
 * Caps the short edge (min(w, h)) so both landscape and portrait phone recordings behave consistently.
 */
export function computeWorkingResolution(
  srcW: number,
  srcH: number,
  mode: InputResMode,
): { workW: number; workH: number; wasDownscaled: boolean } {
  const shortEdge = Math.min(srcW, srcH);
  const maxShort = mode === 'auto_540p' ? 540 : mode === 'auto_720p' ? 720 : Number.POSITIVE_INFINITY;

  if (shortEdge <= maxShort) {
    return {
      workW: Math.max(16, srcW),
      workH: Math.max(16, srcH),
      wasDownscaled: false,
    };
  }

  const ratio = maxShort / shortEdge;
  const workW = Math.max(16, Math.round((srcW * ratio) / 2) * 2);
  const workH = Math.max(16, Math.round((srcH * ratio) / 2) * 2);
  return { workW, workH, wasDownscaled: true };
}

export function recommendScaleForResolution(w: number, h: number): ModelScale {
  return Math.min(w, h) <= 640 ? 2.0 : 1.5;
}

export async function loadDemoVideos(): Promise<VideoSourceMeta[]> {
  const resp = await fetch('/demo_videos/demo_manifest.json');
  if (!resp.ok) {
    throw new Error(`加载演示样片清单失败 (${resp.status})`);
  }
  const json = (await resp.json()) as { demos: DemoManifestItem[] };
  return json.demos.map((d) => ({
    id: `demo:${d.file}`,
    name: d.file,
    description: d.description,
    isDemo: true,
    url: `/demo_videos/${d.file}`,
    width: d.width,
    height: d.height,
    fps: d.fps,
    durationSec: d.duration_sec,
    frameCount: d.frame_count,
    bitrateKbps: d.bitrate_kbps,
    codec: d.codec,
    isRotated: false,
    isHighRes: Math.min(d.width, d.height) > 720,
    defaultInputMode: 'original',
    defaultQp: Math.round(d.segment_qp_mean * 10) / 10,
    defaultIsLive: d.mode === 'live',
    defaultIsScreen: d.is_screen_content,
    recommendedScale: Math.abs(d.scale - 1.5) < 1e-3 ? 1.5 : 2.0,
    fileSizeBytes: Math.round(d.file_size_kb * 1024),
    presetPrior7d: d.prior_7d,
  }));
}

function readVideoElementMetadata(url: string): Promise<{ width: number; height: number; duration: number }> {
  return new Promise((resolve, reject) => {
    const video = document.createElement('video');
    video.preload = 'metadata';
    video.muted = true;
    video.playsInline = true;

    const cleanup = () => {
      video.onloadedmetadata = null;
      video.onerror = null;
      video.src = '';
    };

    video.onloadedmetadata = () => {
      const width = video.videoWidth;
      const height = video.videoHeight;
      const duration = Number.isFinite(video.duration) && video.duration > 0 ? video.duration : 1;
      cleanup();
      if (width <= 0 || height <= 0) {
        reject(new Error('无法读取该视频文件的画面尺寸，请确认其为浏览器支持的视频格式（MP4/WebM/MOV）。'));
        return;
      }
      resolve({ width, height, duration });
    };

    video.onerror = () => {
      cleanup();
      reject(new Error('浏览器无法解码该视频文件，请确认文件未损坏且编码格式受浏览器支持（H.264 / HEVC / VP9 / AV1）。'));
    };

    video.src = url;
  });
}

interface TopLevelBox {
  type: string;
  offset: number;
  size: number;
}

/**
 * Scans top-level ISO BMFF (MP4 / MOV) boxes using tiny 16-byte slices.
 * Instantly finds `ftyp` and `moov` even at the end of a 500MB+ phone recording!
 */
async function scanMp4TopLevelBoxes(file: File): Promise<TopLevelBox[]> {
  const boxes: TopLevelBox[] = [];
  let offset = 0;
  let guard = 0;

  while (offset + 8 <= file.size && guard < 128) {
    guard++;
    const hdrBuf = await file.slice(offset, Math.min(file.size, offset + 16)).arrayBuffer();
    if (hdrBuf.byteLength < 8) break;
    const view = new DataView(hdrBuf);
    let size = view.getUint32(0, false);
    const type = String.fromCharCode(
      view.getUint8(4),
      view.getUint8(5),
      view.getUint8(6),
      view.getUint8(7),
    );

    if (size === 1) {
      if (hdrBuf.byteLength < 16) break;
      const hi = view.getUint32(8, false);
      const lo = view.getUint32(12, false);
      size = hi * 4294967296 + lo;
    } else if (size === 0) {
      size = file.size - offset;
    }

    if (!Number.isFinite(size) || size < 8) break;
    boxes.push({ type, offset, size });
    offset += size;
  }
  return boxes;
}

interface Mp4ProbeInfo {
  codec: string;
  fps: number;
  frameCount: number;
  bitrateKbps: number;
  codedWidth: number;
  codedHeight: number;
}

async function tryProbeMp4Track(file: File): Promise<Mp4ProbeInfo | null> {
  try {
    const boxes = await scanMp4TopLevelBoxes(file);
    const ftyp = boxes.find((b) => b.type === 'ftyp');
    const moov = boxes.find((b) => b.type === 'moov');
    if (!moov) return null;

    const slicesToFeed: TopLevelBox[] = [];
    if (ftyp) slicesToFeed.push(ftyp);
    if (moov.size <= 32 * 1024 * 1024) {
      slicesToFeed.push(moov);
    } else {
      return null;
    }

    return await new Promise((resolve) => {
      const mp4 = MP4Box.createFile();
      let settled = false;
      const finish = (val: Mp4ProbeInfo | null) => {
        if (settled) return;
        settled = true;
        resolve(val);
      };

      const timer = setTimeout(() => finish(null), 2000);
      mp4.onError = () => {
        clearTimeout(timer);
        finish(null);
      };
      mp4.onReady = (info) => {
        clearTimeout(timer);
        const vt = info.videoTracks?.[0];
        if (!vt) {
          finish(null);
          return;
        }
        const durSec = vt.duration && vt.timescale ? vt.duration / vt.timescale : info.duration / info.timescale;
        const fps = durSec > 0 && vt.nb_samples > 0 ? vt.nb_samples / durSec : 30;
        const bitrateKbps = vt.bitrate ? vt.bitrate / 1000 : durSec > 0 ? (file.size * 8) / 1000 / durSec : 1500;
        finish({
          codec: vt.codec || 'avc1',
          fps: Math.max(1, Math.min(240, Math.round(fps * 100) / 100)),
          frameCount: vt.nb_samples || Math.max(1, Math.round(durSec * 30)),
          bitrateKbps: Math.max(50, Math.round(bitrateKbps * 10) / 10),
          codedWidth: vt.video?.width ?? 0,
          codedHeight: vt.video?.height ?? 0,
        });
      };

      (async () => {
        try {
          for (const box of slicesToFeed) {
            const ab = await file.slice(box.offset, box.offset + box.size).arrayBuffer();
            const mp4Buf = ab as MP4Box.MP4ArrayBuffer;
            mp4Buf.fileStart = box.offset;
            mp4.appendBuffer(mp4Buf);
          }
          mp4.flush();
        } catch {
          clearTimeout(timer);
          finish(null);
        }
      })();
    });
  } catch {
    return null;
  }
}

export function formatCodecDisplayName(codec: string): string {
  const c = codec.toLowerCase();
  if (c.startsWith('hvc1') || c.startsWith('hev1') || c.includes('hevc') || c.includes('265')) return 'HEVC (H.265)';
  if (c.startsWith('avc1') || c.includes('264')) return 'H.264 (AVC)';
  if (c.startsWith('av01') || c.includes('av1')) return 'AV1';
  if (c.startsWith('vp09') || c.includes('vp9')) return 'VP9';
  return codec.toUpperCase();
}

export async function probeLocalVideoFile(file: File): Promise<VideoSourceMeta> {
  const url = URL.createObjectURL(file);
  try {
    const [basic, mp4Info] = await Promise.all([readVideoElementMetadata(url), tryProbeMp4Track(file)]);
    const fps = mp4Info?.fps ?? 30;
    const bitrateKbps =
      mp4Info?.bitrateKbps ?? Math.max(50, Math.round(((file.size * 8) / 1000 / basic.duration) * 10) / 10);
    const frameCount = mp4Info?.frameCount ?? Math.max(1, Math.round(basic.duration * fps));
    const codec = mp4Info?.codec ?? 'h264';

    // Detect 90/270-degree phone recording rotation (coded dimensions swapped relative to display dimensions)
    const isRotated = Boolean(
      mp4Info &&
        mp4Info.codedWidth > 0 &&
        mp4Info.codedHeight > 0 &&
        ((mp4Info.codedWidth > mp4Info.codedHeight) !== (basic.width > basic.height)),
    );

    const shortEdge = Math.min(basic.width, basic.height);
    const isHighRes = shortEdge > 720;
    const defaultInputMode: InputResMode = isHighRes ? 'auto_720p' : 'original';
    const { workW, workH } = computeWorkingResolution(basic.width, basic.height, defaultInputMode);
    const recommendedScale = recommendScaleForResolution(workW, workH);
    const defaultQp = estimateQpFromVideo(basic.width, basic.height, fps, bitrateKbps);
    const lowerName = file.name.toLowerCase();
    const defaultIsScreen =
      lowerName.includes('screen') || lowerName.includes('录屏') || lowerName.includes('ppt') || lowerName.includes('slide');

    return {
      id: `local:${file.name}:${file.size}:${file.lastModified}`,
      name: file.name,
      description: `${basic.width}×${basic.height} · ${fps.toFixed(0)}fps · ${Math.round(bitrateKbps)} kbps · ${formatCodecDisplayName(codec)}`,
      isDemo: false,
      url,
      file,
      width: basic.width,
      height: basic.height,
      fps,
      durationSec: basic.duration,
      frameCount,
      bitrateKbps,
      codec,
      isRotated,
      isHighRes,
      defaultInputMode,
      defaultQp,
      defaultIsLive: false,
      defaultIsScreen,
      recommendedScale,
      fileSizeBytes: file.size,
    };
  } catch (err) {
    URL.revokeObjectURL(url);
    throw err;
  }
}
