import * as MP4Box from 'mp4box';
import { estimateQpFromVideo, type ModelScale } from '../core/csr1';

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
  defaultQp: number;
  defaultIsLive: boolean;
  defaultIsScreen: boolean;
  recommendedScale: ModelScale;
  fileSizeBytes: number;
  presetPrior7d?: number[];
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
        reject(new Error('无法读取该视频文件的画面尺寸，请确认其为受支持的视频格式（MP4/WebM/MOV）。'));
        return;
      }
      resolve({ width, height, duration });
    };

    video.onerror = () => {
      cleanup();
      reject(new Error('浏览器无法解码该视频文件，请尝试 H.264/HEVC MP4 或 WebM 视频。'));
    };

    video.src = url;
  });
}

async function tryProbeMp4Track(
  file: File,
): Promise<{ codec: string; fps: number; frameCount: number; bitrateKbps: number } | null> {
  try {
    // Only parse full buffer for reasonable sizes; for very large files parse header slices if needed
    const sliceBytes = Math.min(file.size, 64 * 1024 * 1024);
    const ab = await file.slice(0, sliceBytes).arrayBuffer();
    const mp4Buf = ab as MP4Box.MP4ArrayBuffer;
    mp4Buf.fileStart = 0;

    return await new Promise((resolve) => {
      const mp4 = MP4Box.createFile();
      let settled = false;
      const finish = (val: { codec: string; fps: number; frameCount: number; bitrateKbps: number } | null) => {
        if (settled) return;
        settled = true;
        resolve(val);
      };

      const timer = setTimeout(() => finish(null), 1500);
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
        });
      };

      mp4.appendBuffer(mp4Buf);
      mp4.flush();
    });
  } catch {
    return null;
  }
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
    const defaultQp = estimateQpFromVideo(basic.width, basic.height, fps, bitrateKbps);
    const recommendedScale: ModelScale = basic.height <= 640 ? 2.0 : 1.5;

    return {
      id: `local:${file.name}:${file.size}:${file.lastModified}`,
      name: file.name,
      description: `本地上传 · ${basic.width}×${basic.height} · ${fps.toFixed(0)} fps · ${Math.round(bitrateKbps)} kbps`,
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
      defaultQp,
      defaultIsLive: false,
      defaultIsScreen: false,
      recommendedScale,
      fileSizeBytes: file.size,
    };
  } catch (err) {
    URL.revokeObjectURL(url);
    throw err;
  }
}
