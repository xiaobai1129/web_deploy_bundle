import * as MP4Box from 'mp4box';
import { ArrayBufferTarget, Muxer } from 'mp4-muxer';
import type { CodecSREngine } from '../engine/webgpuRunner';
import type { VideoSourceMeta } from './videoProbe';

export interface ExportProgress {
  phase: 'demux' | 'processing' | 'flushing' | 'done';
  doneFrames: number;
  totalFrames: number;
  fps: number;
  etaSec: number;
  message: string;
}

export interface ExportResult {
  blob: Blob;
  url: string;
  fileName: string;
  width: number;
  height: number;
  frameCount: number;
  elapsedSec: number;
  avgFps: number;
  hasAudio: boolean;
  clippedToSec: number | null;
  pipelineUsed: 'webcodecs-mp4box' | 'video-element-fallback';
}

export interface ExportOptions {
  engine: CodecSREngine;
  source: VideoSourceMeta;
  workW: number;
  workH: number;
  bitrateMbps: number;
  tierLabel: string;
  scale: number;
  maxDurationSec?: number; // e.g. 10 for "first 10 seconds" clip mode
  signal: AbortSignal;
  onProgress: (p: ExportProgress) => void;
}

export interface BufferedEncodedChunk {
  data: Uint8Array;
  type: 'key' | 'delta';
  timestampUs: number;
  durationUs: number;
  meta?: EncodedVideoChunkMetadata;
}

export interface ScheduledMuxChunk extends BufferedEncodedChunk {
  presentationTimestampUs: number;
  compositionTimeOffsetUs: number;
}

/**
 * Selects H.264 AVC codec strings whose level fits the target output resolution.
 * Avoids forcing Level 5.2 on tiny videos (e.g. 384x288) while still supporting 4K@60fps.
 */
export function selectAvcCandidatesForResolution(width: number, height: number): string[] {
  const pixels = width * height;
  if (pixels <= 921_600) {
    // <= 720p (use Level 4.0/4.2 first so 15-25 Mbps never exceeds Level 3.1 CPB limits)
    return [
      'avc1.640028', // High 4.0
      'avc1.64002a', // High 4.2
      'avc1.64001f', // High 3.1
      'avc1.4d0028', // Main 4.0
      'avc1.4d001f', // Main 3.1
      'avc1.42e028', // Constrained Baseline 4.0
      'avc1.42001f', // Baseline 3.1
    ];
  }
  if (pixels <= 2_088_960) {
    // <= 1080p
    return [
      'avc1.64002a', // High 4.2
      'avc1.640028', // High 4.0
      'avc1.640032', // High 5.0
      'avc1.4d002a', // Main 4.2
      'avc1.4d0028', // Main 4.0
      'avc1.42e02a', // Constrained Baseline 4.2
      'avc1.420028', // Baseline 4.0
    ];
  }
  // > 1080p (1440p / 4K)
  return [
    'avc1.640034', // High 5.2
    'avc1.640033', // High 5.1
    'avc1.640032', // High 5.0
    'avc1.4d0034', // Main 5.2
    'avc1.4d0033', // Main 5.1
    'avc1.42e033', // Constrained Baseline 5.1
  ];
}

/**
 * Checks whether demuxed MP4 samples contain B-frame reordering (non-monotonic CTS).
 */
export function detectInputHasBFrames(samples: Array<{ cts: number }>): boolean {
  for (let i = 1; i < samples.length; i++) {
    if (samples[i].cts < samples[i - 1].cts) {
      return true;
    }
  }
  return false;
}

/**
 * Computes valid MP4 mux timestamps (PTS + compositionTimeOffset) for encoded video chunks.
 *
 * - When `latencyMode: 'realtime'` disables B-frames (the normal path), PTS === DTS and CTO === 0.
 * - If any platform encoder still emits B-frames in decode order (non-monotonic `timestampUs`),
 *   this reconstructs monotonic DTS from sorted PTS values and computes non-negative
 *   `compositionTimeOffsetUs` (CTTS) so players never display frames out of order ("ping-pong jitter").
 */
export function computeMuxSchedule(
  chunks: BufferedEncodedChunk[],
  fallbackDurationUs: number,
): ScheduledMuxChunk[] {
  if (chunks.length === 0) return [];
  const safeFallbackDur = Math.max(1, Math.round(fallbackDurationUs));

  let hasReordering = false;
  for (let i = 1; i < chunks.length; i++) {
    if (chunks[i].timestampUs < chunks[i - 1].timestampUs) {
      hasReordering = true;
      break;
    }
  }

  if (!hasReordering) {
    const out: ScheduledMuxChunk[] = [];
    let lastTs = -1;
    for (let i = 0; i < chunks.length; i++) {
      const c = chunks[i];
      let ts = Math.max(0, Math.round(c.timestampUs));
      if (lastTs >= 0 && ts <= lastTs) {
        ts = lastTs + 1;
      }
      lastTs = ts;
      const nextTs = i + 1 < chunks.length ? Math.max(ts + 1, Math.round(chunks[i + 1].timestampUs)) : ts + safeFallbackDur;
      const dur = c.durationUs > 0 ? Math.round(c.durationUs) : Math.max(1, nextTs - ts);
      out.push({
        ...c,
        durationUs: dur,
        presentationTimestampUs: ts,
        compositionTimeOffsetUs: 0,
      });
    }
    return out;
  }

  // Reorder-safe path: reconstruct monotonic DTS from sorted presentation timestamps
  const sortedPts = chunks.map((c) => Math.max(0, Math.round(c.timestampUs))).sort((a, b) => a - b);
  for (let i = 1; i < sortedPts.length; i++) {
    if (sortedPts[i] <= sortedPts[i - 1]) {
      sortedPts[i] = sortedPts[i - 1] + 1;
    }
  }

  let minOffset = 0;
  for (let i = 0; i < chunks.length; i++) {
    const diff = Math.max(0, Math.round(chunks[i].timestampUs)) - sortedPts[i];
    if (diff < minOffset) {
      minOffset = diff;
    }
  }
  const shiftUs = minOffset < 0 ? -minOffset : 0;

  return chunks.map((c, i) => {
    const dtsUs = sortedPts[i];
    const ptsUs = Math.max(0, Math.round(c.timestampUs)) + shiftUs;
    const ctoUs = Math.max(0, ptsUs - dtsUs);
    const dtsDur =
      i + 1 < sortedPts.length ? Math.max(1, sortedPts[i + 1] - dtsUs) : c.durationUs > 0 ? Math.round(c.durationUs) : safeFallbackDur;
    return {
      ...c,
      durationUs: dtsDur,
      presentationTimestampUs: ptsUs,
      compositionTimeOffsetUs: ctoUs,
    };
  });
}

async function pickH264EncoderConfig(
  width: number,
  height: number,
  fps: number,
  bitrateBps: number,
): Promise<VideoEncoderConfig> {
  if (typeof VideoEncoder === 'undefined') {
    throw new Error('当前浏览器不支持 WebCodecs VideoEncoder，请使用最新 Chrome / Edge 浏览器导出 MP4。');
  }
  const safeFps = Math.max(1, Math.min(120, Math.round(fps)));
  const candidates = selectAvcCandidatesForResolution(width, height);

  // CRITICAL: `latencyMode: 'realtime'` disables B-frame reordering in VideoToolbox / MediaFoundation /
  // OpenH264 so encoded chunks arrive in strict presentation order without back-and-forth jitter.
  for (const hw of ['prefer-hardware', 'no-preference'] as const) {
    for (const codec of candidates) {
      const cfg: VideoEncoderConfig = {
        codec,
        width,
        height,
        bitrate: bitrateBps,
        framerate: safeFps,
        latencyMode: 'realtime',
        bitrateMode: 'variable',
        hardwareAcceleration: hw,
        avc: { format: 'avc' },
      };
      try {
        const res = await VideoEncoder.isConfigSupported(cfg);
        if (res.supported) {
          return {
            ...(res.config ?? cfg),
            latencyMode: 'realtime',
            avc: { format: 'avc' },
          };
        }
      } catch {
        // try without bitrateMode
      }

      const fallbackCfg: VideoEncoderConfig = {
        codec,
        width,
        height,
        bitrate: bitrateBps,
        framerate: safeFps,
        latencyMode: 'realtime',
        hardwareAcceleration: hw,
        avc: { format: 'avc' },
      };
      try {
        const res = await VideoEncoder.isConfigSupported(fallbackCfg);
        if (res.supported) {
          return {
            ...(res.config ?? fallbackCfg),
            latencyMode: 'realtime',
            avc: { format: 'avc' },
          };
        }
      } catch {
        // try next candidate
      }
    }
  }
  throw new Error(
    `浏览器 H.264 编码器不支持当前输出分辨率 (${width}×${height} @ ${safeFps}fps)。建议在右侧开启「720p/540p 流畅模式」降低输出尺寸后再导出。`,
  );
}

function extractCodecDescription(mp4: MP4Box.MP4File, trackId: number): Uint8Array | undefined {
  const trak = mp4.getTrackById(trackId);
  const entry = trak?.mdia?.minf?.stbl?.stsd?.entries?.[0];
  if (!entry) return undefined;
  const box = entry.avcC ?? entry.hvcC ?? entry.vpcC ?? entry.av1C;
  if (!box) return undefined;
  const stream = new MP4Box.DataStream(undefined, 0, MP4Box.DataStream.BIG_ENDIAN);
  box.write(stream);
  return new Uint8Array(stream.buffer, 8);
}

interface DemuxedMp4 {
  videoTrack: MP4Box.MP4Track;
  videoDescription?: Uint8Array;
  videoSamples: MP4Box.MP4Sample[];
  audioTrack?: MP4Box.MP4Track;
  audioSamples: MP4Box.MP4Sample[];
}

async function demuxMp4Source(source: VideoSourceMeta): Promise<DemuxedMp4> {
  const mp4 = MP4Box.createFile();
  let videoTrack: MP4Box.MP4Track | null = null;
  let audioTrack: MP4Box.MP4Track | undefined;
  const videoSamples: MP4Box.MP4Sample[] = [];
  const audioSamples: MP4Box.MP4Sample[] = [];
  let parseErr: string | null = null;

  mp4.onError = (e) => {
    parseErr = `MP4 解封装失败: ${e}`;
  };

  mp4.onReady = (info) => {
    if (!info.videoTracks?.length) {
      parseErr = 'MP4 文件中未找到视频轨';
      return;
    }
    videoTrack = info.videoTracks[0];
    mp4.setExtractionOptions(videoTrack.id, 'video', {
      nbSamples: Math.max(1000, videoTrack.nb_samples),
    });

    const aTrack = info.audioTracks?.find((t) => t.codec?.toLowerCase().startsWith('mp4a'));
    if (aTrack && aTrack.audio) {
      audioTrack = aTrack;
      mp4.setExtractionOptions(aTrack.id, 'audio', {
        nbSamples: Math.max(2000, aTrack.nb_samples),
      });
    }
    mp4.start();
  };

  mp4.onSamples = (_trackId, user, samples) => {
    if (user === 'video') {
      for (const s of samples) videoSamples.push(s);
    } else if (user === 'audio') {
      for (const s of samples) audioSamples.push(s);
    }
  };

  if (source.file) {
    // Stream in 8MB slices to avoid doubling memory on 150MB+ phone recordings
    const CHUNK = 8 * 1024 * 1024;
    for (let off = 0; off < source.file.size; off += CHUNK) {
      const ab = await source.file.slice(off, Math.min(source.file.size, off + CHUNK)).arrayBuffer();
      const mp4Buf = ab as MP4Box.MP4ArrayBuffer;
      mp4Buf.fileStart = off;
      mp4.appendBuffer(mp4Buf);
      if (parseErr) throw new Error(parseErr);
    }
  } else {
    const resp = await fetch(source.url);
    if (!resp.ok) throw new Error(`读取演示视频失败 (${resp.status})`);
    const ab = await resp.arrayBuffer();
    const mp4Buf = ab as MP4Box.MP4ArrayBuffer;
    mp4Buf.fileStart = 0;
    mp4.appendBuffer(mp4Buf);
  }

  mp4.flush();
  await new Promise((r) => setTimeout(r, 0));

  if (parseErr) throw new Error(parseErr);
  const vt = videoTrack as MP4Box.MP4Track | null;
  if (!vt || videoSamples.length === 0) {
    throw new Error('MP4 中未提取到有效视频帧');
  }
  return {
    videoTrack: vt,
    videoDescription: extractCodecDescription(mp4, vt.id),
    videoSamples,
    audioTrack,
    audioSamples,
  };
}

function makeOutputFileName(
  sourceName: string,
  tierLabel: string,
  scale: number,
  clippedSec: number | null,
): string {
  const base = sourceName.replace(/\.[^.]+$/, '');
  const scaleTag = Math.abs(scale - 1.5) < 1e-3 ? 'x1_5' : 'x2';
  const clipTag = clippedSec !== null ? `_first${Math.round(clippedSec)}s` : '';
  return `${base}_CodecSR_${tierLabel}_${scaleTag}${clipTag}.mp4`;
}

function popEarliestFrame(queue: VideoFrame[]): VideoFrame | undefined {
  if (queue.length === 0) return undefined;
  let minIdx = 0;
  let minTs = queue[0].timestamp;
  for (let i = 1; i < queue.length; i++) {
    if (queue[i].timestamp < minTs) {
      minTs = queue[i].timestamp;
      minIdx = i;
    }
  }
  const [frame] = queue.splice(minIdx, 1);
  return frame;
}

async function exportViaMp4BoxWebCodecs(
  opts: ExportOptions,
  encW: number,
  encH: number,
): Promise<ExportResult> {
  const { engine, source, workW, workH, bitrateMbps, tierLabel, scale, maxDurationSec, signal, onProgress } = opts;
  const startWall = performance.now();
  const clippedSec =
    maxDurationSec && maxDurationSec > 0 && source.durationSec > maxDurationSec + 0.25
      ? maxDurationSec
      : null;

  onProgress({
    phase: 'demux',
    doneFrames: 0,
    totalFrames: source.frameCount,
    fps: 0,
    etaSec: 0,
    message: '正在解析视频码流...',
  });

  const demuxed = await demuxMp4Source(source);
  if (signal.aborted) throw new DOMException('已取消导出', 'AbortError');

  const decoderCfg: VideoDecoderConfig = {
    codec: demuxed.videoTrack.codec,
    codedWidth: demuxed.videoTrack.video?.width ?? source.width,
    codedHeight: demuxed.videoTrack.video?.height ?? source.height,
    description: demuxed.videoDescription,
  };
  const decSupport = await VideoDecoder.isConfigSupported(decoderCfg);
  if (!decSupport.supported) {
    throw new Error(`VideoDecoder 不支持 ${decoderCfg.codec}`);
  }

  // Filter samples when exporting a time-bounded clip (e.g. first 10s)
  const firstVideoCtsSec = demuxed.videoSamples[0].cts / demuxed.videoSamples[0].timescale;
  const maxCtsSec = clippedSec !== null ? firstVideoCtsSec + clippedSec : Number.POSITIVE_INFINITY;
  const maxUsCutoff =
    clippedSec !== null ? Math.round((firstVideoCtsSec + clippedSec) * 1_000_000) : Number.POSITIVE_INFINITY;

  // Keep a small B-frame decode margin (+0.5s) then trim exact presentation timestamps on output
  const feedSamples: MP4Box.MP4Sample[] = [];
  for (const s of demuxed.videoSamples) {
    const dtsSec = s.dts / s.timescale;
    if (clippedSec !== null && dtsSec > maxCtsSec + 0.5 && feedSamples.length > 0) {
      break;
    }
    feedSamples.push(s);
  }

  const inputHasBFrames = detectInputHasBFrames(feedSamples);
  // Keep a small 2-frame PTS reorder window when input has B-frames (safe for compact hardware decoder pools)
  const reorderWindow = inputHasBFrames ? 2 : 0;

  const audioSamplesToMux: MP4Box.MP4Sample[] = [];
  if (demuxed.audioSamples.length > 0) {
    const firstAudioSec = demuxed.audioSamples[0].cts / demuxed.audioSamples[0].timescale;
    const maxAudioSec = clippedSec !== null ? firstAudioSec + clippedSec : Number.POSITIVE_INFINITY;
    for (const a of demuxed.audioSamples) {
      if (a.cts / a.timescale <= maxAudioSec) {
        audioSamplesToMux.push(a);
      }
    }
  }

  const fps =
    demuxed.videoTrack.duration && demuxed.videoTrack.timescale
      ? demuxed.videoSamples.length / (demuxed.videoTrack.duration / demuxed.videoTrack.timescale)
      : source.fps;
  const estimatedTargetFrames =
    clippedSec !== null
      ? Math.max(1, Math.min(feedSamples.length, Math.round(clippedSec * fps)))
      : feedSamples.length;

  onProgress({
    phase: 'demux',
    doneFrames: 0,
    totalFrames: estimatedTargetFrames,
    fps: 0,
    etaSec: 0,
    message: '正在初始化本地 H.264 编码器...',
  });

  // Keep target bitrate sensible for tiny resolutions (e.g. 384x288) while respecting 8-25 Mbps for HD
  const effectiveBitrateBps = Math.min(
    Math.round(bitrateMbps * 1_000_000),
    Math.max(2_500_000, Math.round(encW * encH * Math.max(15, fps) * 2.5)),
  );
  const encoderCfg = await pickH264EncoderConfig(encW, encH, fps, effectiveBitrateBps);

  const target = new ArrayBufferTarget();
  const hasAudio = Boolean(demuxed.audioTrack?.audio && audioSamplesToMux.length > 0);
  const muxer = new Muxer({
    target,
    video: {
      codec: 'avc',
      width: encW,
      height: encH,
    },
    audio:
      hasAudio && demuxed.audioTrack?.audio
        ? {
            codec: 'aac',
            numberOfChannels: demuxed.audioTrack.audio.channel_count,
            sampleRate: demuxed.audioTrack.audio.sample_rate,
          }
        : undefined,
    fastStart: 'in-memory',
    firstTimestampBehavior: 'offset',
  });

  const defaultDurUs = Math.round(1_000_000 / Math.max(1, fps));
  const encodedVideoChunks: BufferedEncodedChunk[] = [];
  const decodedQueue: VideoFrame[] = [];
  let notifyFrameAvailable: (() => void) | null = null;
  let decoderDone = false;
  let decoderError: Error | null = null;
  let encoderError: Error | null = null;
  let consumerError: Error | null = null;

  const wakeConsumer = () => {
    if (notifyFrameAvailable) {
      const fn = notifyFrameAvailable;
      notifyFrameAvailable = null;
      fn();
    }
  };
  signal.addEventListener('abort', wakeConsumer, { once: true });

  const encoder = new VideoEncoder({
    output: (chunk, meta) => {
      try {
        const data = new Uint8Array(chunk.byteLength);
        chunk.copyTo(data);
        encodedVideoChunks.push({
          data,
          type: chunk.type,
          timestampUs: chunk.timestamp,
          durationUs: chunk.duration && chunk.duration > 0 ? chunk.duration : defaultDurUs,
          meta,
        });
      } catch (e) {
        encoderError = e instanceof Error ? e : new Error(String(e));
        wakeConsumer();
      }
    },
    error: (e) => {
      encoderError = e instanceof Error ? e : new Error(String(e));
      wakeConsumer();
    },
  });
  encoder.configure(encoderCfg);

  const decoder = new VideoDecoder({
    output: (frame) => {
      decodedQueue.push(frame);
      wakeConsumer();
    },
    error: (e) => {
      decoderError = e instanceof Error ? e : new Error(String(e));
      wakeConsumer();
    },
  });
  decoder.configure(decoderCfg);

  const keyframeInterval = Math.max(15, Math.min(120, Math.round(fps * 2)));
  let processedFrames = 0;
  let lastFedPtsUs = -1;

  onProgress({
    phase: 'processing',
    doneFrames: 0,
    totalFrames: estimatedTargetFrames,
    fps: 0,
    etaSec: 0,
    message: `正在启动本地 WebGPU 超分渲染 (0/${estimatedTargetFrames} 帧)...`,
  });

  const consumerPromise = (async () => {
    try {
      while (true) {
        if (signal.aborted) break;
        if (decoderError) throw decoderError;
        if (encoderError) throw encoderError;

        // Hold up to `reorderWindow` frames until decoder finishes so we always pop in strict PTS order
        if (!decoderDone && decodedQueue.length <= reorderWindow) {
          await new Promise<void>((r) => {
            notifyFrameAvailable = r;
          });
          continue;
        }

        const frame = popEarliestFrame(decodedQueue);
        if (!frame) {
          if (decoderDone) break;
          await new Promise<void>((r) => {
            notifyFrameAvailable = r;
          });
          continue;
        }

        try {
          if (clippedSec !== null && frame.timestamp > maxUsCutoff) {
            continue;
          }

          // Apply encoder backpressure BEFORE GPU SR + readback
          while (encoder.encodeQueueSize > 6 && !signal.aborted && !encoderError && !decoderError) {
            await new Promise((r) => setTimeout(r, 4));
          }
          if (signal.aborted) break;
          if (encoderError) throw encoderError;
          if (decoderError) throw decoderError;

          let tsUs = Math.max(0, Math.round(frame.timestamp));
          if (lastFedPtsUs >= 0 && tsUs <= lastFedPtsUs) {
            tsUs = lastFedPtsUs + defaultDurUs;
          }
          lastFedPtsUs = tsUs;
          const durUs = frame.duration && frame.duration > 0 ? Math.round(frame.duration) : defaultDurUs;

          const w = frame.displayWidth || frame.codedWidth;
          const h = frame.displayHeight || frame.codedHeight;
          // Direct GPU srTex -> MAP_READ -> synchronous 2D canvas -> VideoFrame (zero WebGPU swapchain race!)
          const outFrame = await engine.processFrameToVideoFrame(
            frame,
            w,
            h,
            { timestamp: tsUs, duration: durUs, width: encW, height: encH },
            { workW, workH },
          );
          try {
            encoder.encode(outFrame, { keyFrame: processedFrames % keyframeInterval === 0 });
          } finally {
            outFrame.close();
          }
          processedFrames++;

          const elapsedSec = Math.max(0.01, (performance.now() - startWall) / 1000);
          const curFps = processedFrames / elapsedSec;
          const totalEst = Math.max(processedFrames, estimatedTargetFrames);
          const etaSec = curFps > 0 ? Math.max(0, (totalEst - processedFrames) / curFps) : 0;
          onProgress({
            phase: 'processing',
            doneFrames: processedFrames,
            totalFrames: totalEst,
            fps: curFps,
            etaSec,
            message: `本地 WebGPU 超分渲染中 (${processedFrames}/${totalEst} 帧)`,
          });
        } finally {
          frame.close();
        }
      }
    } catch (err) {
      consumerError = err instanceof Error ? err : new Error(String(err));
      throw consumerError;
    }
  })();

  try {
    const maxDecodedQueue = Math.max(6, reorderWindow + 4);
    for (let i = 0; i < feedSamples.length; i++) {
      if (signal.aborted) throw new DOMException('已取消导出', 'AbortError');
      if (decoderError) throw decoderError;
      if (encoderError) throw encoderError;
      if (consumerError) throw consumerError;

      while (
        (decoder.decodeQueueSize > 8 || decodedQueue.length > maxDecodedQueue) &&
        !signal.aborted &&
        !decoderError &&
        !encoderError &&
        !consumerError
      ) {
        await new Promise((r) => setTimeout(r, 4));
      }
      if (signal.aborted) throw new DOMException('已取消导出', 'AbortError');
      if (decoderError) throw decoderError;
      if (encoderError) throw encoderError;
      if (consumerError) throw consumerError;

      const s = feedSamples[i];
      const chunk = new EncodedVideoChunk({
        type: s.is_sync ? 'key' : 'delta',
        timestamp: Math.round((s.cts * 1_000_000) / s.timescale),
        duration: Math.round((s.duration * 1_000_000) / s.timescale),
        data: s.data,
      });
      decoder.decode(chunk);
    }

    await decoder.flush();
    decoderDone = true;
    wakeConsumer();

    await consumerPromise;
    if (signal.aborted) throw new DOMException('已取消导出', 'AbortError');

    onProgress({
      phase: 'flushing',
      doneFrames: processedFrames,
      totalFrames: processedFrames,
      fps: processedFrames / Math.max(0.01, (performance.now() - startWall) / 1000),
      etaSec: 0,
      message: '正在封装最终 MP4 文件...',
    });

    await encoder.flush();
    if (encoderError) throw encoderError;

    const scheduledChunks = computeMuxSchedule(encodedVideoChunks, defaultDurUs);
    encodedVideoChunks.length = 0;
    for (const c of scheduledChunks) {
      muxer.addVideoChunkRaw(
        c.data,
        c.type,
        c.presentationTimestampUs,
        c.durationUs,
        c.meta,
        c.compositionTimeOffsetUs,
      );
    }

    if (hasAudio && audioSamplesToMux.length > 0) {
      for (const a of audioSamplesToMux) {
        const tsUs = Math.max(0, Math.round((a.cts * 1_000_000) / a.timescale));
        const durUs = Math.max(1, Math.round((a.duration * 1_000_000) / a.timescale));
        muxer.addAudioChunkRaw(a.data, 'key', tsUs, durUs);
      }
    }

    muxer.finalize();
  } finally {
    signal.removeEventListener('abort', wakeConsumer);
    decoderDone = true;
    wakeConsumer();
    // Ensure consumerPromise rejection is observed even when producer threw first
    consumerPromise.catch(() => {});
    for (const leftover of decodedQueue) leftover.close();
    decodedQueue.length = 0;
    try {
      if (decoder.state !== 'closed') decoder.close();
    } catch {
      // ignore
    }
    try {
      if (encoder.state !== 'closed') encoder.close();
    } catch {
      // ignore
    }
  }

  const blob = new Blob([target.buffer], { type: 'video/mp4' });
  const elapsedSec = Math.max(0.01, (performance.now() - startWall) / 1000);
  const res: ExportResult = {
    blob,
    url: URL.createObjectURL(blob),
    fileName: makeOutputFileName(source.name, tierLabel, scale, clippedSec),
    width: encW,
    height: encH,
    frameCount: processedFrames,
    elapsedSec,
    avgFps: processedFrames / elapsedSec,
    hasAudio,
    clippedToSec: clippedSec,
    pipelineUsed: 'webcodecs-mp4box',
  };

  onProgress({
    phase: 'done',
    doneFrames: processedFrames,
    totalFrames: processedFrames,
    fps: res.avgFps,
    etaSec: 0,
    message: '超分 MP4 导出完成',
  });
  return res;
}

async function exportViaVideoElementFallback(
  opts: ExportOptions,
  encW: number,
  encH: number,
): Promise<ExportResult> {
  const { engine, source, workW, workH, bitrateMbps, tierLabel, scale, maxDurationSec, signal, onProgress } = opts;
  const startWall = performance.now();
  const clippedSec =
    maxDurationSec && maxDurationSec > 0 && source.durationSec > maxDurationSec + 0.25
      ? maxDurationSec
      : null;
  const targetDurSec = clippedSec ?? source.durationSec;

  const fps = Math.max(1, Math.min(60, Math.round(source.fps || 30)));
  const totalFrames = Math.max(1, Math.floor(targetDurSec * fps));
  const encoderCfg = await pickH264EncoderConfig(encW, encH, fps, Math.round(bitrateMbps * 1_000_000));

  const target = new ArrayBufferTarget();
  const muxer = new Muxer({
    target,
    video: { codec: 'avc', width: encW, height: encH },
    fastStart: 'in-memory',
    firstTimestampBehavior: 'offset',
  });

  const durUs = Math.round(1_000_000 / fps);
  const encodedVideoChunks: BufferedEncodedChunk[] = [];
  let encoderError: Error | null = null;
  const encoder = new VideoEncoder({
    output: (chunk, meta) => {
      try {
        const data = new Uint8Array(chunk.byteLength);
        chunk.copyTo(data);
        encodedVideoChunks.push({
          data,
          type: chunk.type,
          timestampUs: chunk.timestamp,
          durationUs: chunk.duration && chunk.duration > 0 ? chunk.duration : durUs,
          meta,
        });
      } catch (e) {
        encoderError = e instanceof Error ? e : new Error(String(e));
      }
    },
    error: (e) => {
      encoderError = e instanceof Error ? e : new Error(String(e));
    },
  });
  encoder.configure(encoderCfg);

  const video = document.createElement('video');
  video.muted = true;
  video.playsInline = true;
  video.preload = 'auto';
  video.src = source.url;

  await new Promise<void>((resolve, reject) => {
    video.onloadeddata = () => resolve();
    video.onerror = () => reject(new Error('无法加载视频进行逐帧导出'));
  });

  const stepSec = 1 / fps;
  const keyframeInterval = Math.max(15, fps * 2);
  let processed = 0;

  try {
    for (let i = 0; i < totalFrames; i++) {
      if (signal.aborted) throw new DOMException('已取消导出', 'AbortError');
      if (encoderError) throw encoderError;

      const tSec = Math.min(Math.max(0, source.durationSec - 0.001), i * stepSec);
      if (Math.abs(video.currentTime - tSec) > 1e-4 || video.readyState < 2) {
        await new Promise<void>((resolve) => {
          const onSeeked = () => {
            video.removeEventListener('seeked', onSeeked);
            resolve();
          };
          video.addEventListener('seeked', onSeeked);
          video.currentTime = Math.max(0, tSec);
        });
      }

      while (encoder.encodeQueueSize > 6 && !signal.aborted && !encoderError) {
        await new Promise((r) => setTimeout(r, 4));
      }
      if (signal.aborted) throw new DOMException('已取消导出', 'AbortError');
      if (encoderError) throw encoderError;

      const outFrame = await engine.processFrameToVideoFrame(
        video,
        video.videoWidth,
        video.videoHeight,
        { timestamp: i * durUs, duration: durUs, width: encW, height: encH },
        { workW, workH },
      );
      encoder.encode(outFrame, { keyFrame: i % keyframeInterval === 0 });
      outFrame.close();
      processed++;

      const elapsedSec = Math.max(0.01, (performance.now() - startWall) / 1000);
      const curFps = processed / elapsedSec;
      const etaSec = curFps > 0 ? Math.max(0, (totalFrames - processed) / curFps) : 0;
      onProgress({
        phase: 'processing',
        doneFrames: processed,
        totalFrames,
        fps: curFps,
        etaSec,
        message: `兼容模式逐帧超分中 (${processed}/${totalFrames} 帧)`,
      });
    }

    onProgress({
      phase: 'flushing',
      doneFrames: processed,
      totalFrames: processed,
      fps: processed / Math.max(0.01, (performance.now() - startWall) / 1000),
      etaSec: 0,
      message: '正在封装最终 MP4 文件...',
    });

    await encoder.flush();
    if (encoderError) throw encoderError;

    const scheduledChunks = computeMuxSchedule(encodedVideoChunks, durUs);
    encodedVideoChunks.length = 0;
    for (const c of scheduledChunks) {
      muxer.addVideoChunkRaw(
        c.data,
        c.type,
        c.presentationTimestampUs,
        c.durationUs,
        c.meta,
        c.compositionTimeOffsetUs,
      );
    }

    muxer.finalize();
  } finally {
    video.src = '';
    try {
      if (encoder.state !== 'closed') encoder.close();
    } catch {
      // ignore
    }
  }

  const blob = new Blob([target.buffer], { type: 'video/mp4' });
  const elapsedSec = Math.max(0.01, (performance.now() - startWall) / 1000);
  const res: ExportResult = {
    blob,
    url: URL.createObjectURL(blob),
    fileName: makeOutputFileName(source.name, tierLabel, scale, clippedSec),
    width: encW,
    height: encH,
    frameCount: processed,
    elapsedSec,
    avgFps: processed / elapsedSec,
    hasAudio: false,
    clippedToSec: clippedSec,
    pipelineUsed: 'video-element-fallback',
  };
  onProgress({
    phase: 'done',
    doneFrames: processed,
    totalFrames: processed,
    fps: res.avgFps,
    etaSec: 0,
    message: '超分 MP4 导出完成',
  });
  return res;
}

/**
 * Exports the video super-resolved with local WebGPU into a downloadable H.264 MP4.
 * - Uses chunked MP4Box + WebCodecs VideoDecoder (H.264 / HEVC / VP9 / AV1) when unrotated & supported.
 * - Automatically falls back to browser-rotated <video> stepping for MOV/WebM/rotated phone clips.
 * - Supports optional `maxDurationSec` (e.g. first 10s fast clip export).
 */
export async function exportSuperResolvedMp4(opts: ExportOptions): Promise<ExportResult> {
  const rawOutW = Math.round(opts.workW * opts.scale);
  const rawOutH = Math.round(opts.workH * opts.scale);
  // H.264 strictly requires even dimensions
  const encW = Math.max(16, rawOutW & ~1);
  const encH = Math.max(16, rawOutH & ~1);

  if (typeof VideoDecoder !== 'undefined' && !opts.source.isRotated) {
    try {
      return await exportViaMp4BoxWebCodecs(opts, encW, encH);
    } catch (err) {
      if (opts.signal.aborted || (err instanceof DOMException && err.name === 'AbortError')) {
        throw err;
      }
      console.warn('[CodecSR Export] MP4Box/VideoDecoder path failed, falling back to <video> stepping:', err);
    }
  }

  return await exportViaVideoElementFallback(opts, encW, encH);
}
