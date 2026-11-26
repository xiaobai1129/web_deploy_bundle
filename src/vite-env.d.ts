/// <reference types="vite/client" />
/// <reference types="@webgpu/types" />

declare module 'mp4box' {
  export interface MP4Track {
    id: number;
    type: 'video' | 'audio' | string;
    codec: string;
    timescale: number;
    duration: number;
    nb_samples: number;
    video?: {
      width: number;
      height: number;
    };
    audio?: {
      sample_rate: number;
      channel_count: number;
    };
    bitrate?: number;
  }

  export interface MP4Info {
    duration: number;
    timescale: number;
    isFragmented: boolean;
    tracks: MP4Track[];
    videoTracks: MP4Track[];
    audioTracks: MP4Track[];
  }

  export interface MP4Sample {
    number: number;
    track_id: number;
    data: Uint8Array;
    size: number;
    cts: number;
    dts: number;
    duration: number;
    is_sync: boolean;
    timescale: number;
  }

  export interface MP4ArrayBuffer extends ArrayBuffer {
    fileStart: number;
  }

  export interface MP4File {
    onReady: ((info: MP4Info) => void) | null;
    onError: ((err: string) => void) | null;
    onSamples: ((trackId: number, user: unknown, samples: MP4Sample[]) => void) | null;
    appendBuffer(buffer: MP4ArrayBuffer): number;
    flush(): void;
    start(): void;
    stop(): void;
    setExtractionOptions(trackId: number, user?: unknown, options?: { nbSamples?: number }): void;
    releaseUsedSamples(trackId: number, sampleNumber: number): void;
    getTrackById(trackId: number): {
      mdia: {
        minf: {
          stbl: {
            stsd: {
              entries: Array<{
                avcC?: { write: (stream: DataStream) => void };
                hvcC?: { write: (stream: DataStream) => void };
                vpcC?: { write: (stream: DataStream) => void };
                av1C?: { write: (stream: DataStream) => void };
              }>;
            };
          };
        };
      };
    };
  }

  export class DataStream {
    buffer: ArrayBuffer;
    position: number;
    endianness: number;
    static BIG_ENDIAN: number;
    constructor(arrayBuffer?: ArrayBuffer, byteOffset?: number, endianness?: number);
  }

  export function createFile(): MP4File;
}
