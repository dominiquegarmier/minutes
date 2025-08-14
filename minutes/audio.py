"""Audio recording functionality for desktop audio capture."""

from __future__ import annotations

import logging
import re
import subprocess
import tempfile
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Deque
from typing import Iterable
from typing import Optional

import numpy as np
import sounddevice as sd  # type: ignore[import-untyped]
import soundfile as sf  # type: ignore[import-untyped]

from .utils import linear_resample_mono
from .utils import to_mono

DEFAULT_SR = 48_000  # typical desktop sample rate; we'll resample to 16k mono
TARGET_SR = 16_000


def get_input_device_map() -> dict[int, int]:
    """Get mapping from continuous 0-N indexing to actual sounddevice IDs."""
    devices = sd.query_devices()
    device_map = {}
    continuous_idx = 0
    
    for i, device in enumerate(devices):
        dev_dict = dict(device)  # type: ignore[arg-type]
        channels = dev_dict.get('max_input_channels', 0)
        if channels > 0:  # Only include input-capable devices
            device_map[continuous_idx] = i
            continuous_idx += 1
    
    return device_map


def map_continuous_to_device_id(continuous_id: int) -> int:
    """Map continuous device index (0-N) to actual sounddevice ID."""
    device_map = get_input_device_map()
    return device_map.get(continuous_id, continuous_id)  # Fallback to same ID if not found


@dataclass
class AudioConfig:
    """Configuration for desktop audio capture."""

    device: Optional[str] = None  # Comma-separated device numbers or single device
    sample_rate: int = DEFAULT_SR
    block_size: int = 2048
    dtype: str = "float32"
    loopback: bool = True  # Windows WASAPI loopback. Ignored if unsupported.
    enable_microphone: bool = True  # Also capture microphone input
    microphone_device: Optional[str] = None  # Specific mic device if needed
    
    def get_device_list(self) -> list[int]:
        """Parse device string into list of actual device IDs (maps continuous numbering to sounddevice IDs)."""
        if not self.device:
            return []
        try:
            continuous_ids = [int(d.strip()) for d in self.device.split(",")]
            return [map_continuous_to_device_id(cid) for cid in continuous_ids]
        except ValueError:
            # Fallback for legacy single device name/id
            try:
                continuous_id = int(self.device)
                return [map_continuous_to_device_id(continuous_id)]
            except ValueError:
                return []  # Legacy name-based device - handled by existing logic


class AudioRecorder:
    """Continuously captures audio from multiple devices and yields overlapping WAV segments.

    For multi-device capture, uses a round-robin approach instead of real-time mixing
    to avoid audio corruption issues. Each device gets processed in sequence with
    overlapping segments. Single device operation works as before.

    Process flow:
    1. Start background audio capture threads (one per device)
    2. Each callback accumulates frames in device-specific buffers
    3. segment_generator() processes devices round-robin style
    4. Yields temp WAV files from individual devices or simple concatenation
    """

    def __init__(
        self,
        cfg: AudioConfig,
        segment_secs: float = 20.0,
        overlap_secs: float = 5.0,
    ) -> None:
        self.cfg = cfg
        self.segment_secs = segment_secs
        self.overlap_secs = overlap_secs

        # Master mixed buffer
        self._mixed_buf: Deque[np.ndarray] = deque(maxlen=int(10 * cfg.sample_rate))
        self._mixed_lock = threading.Lock()
        
        # Per-device buffers for multi-device mixing
        self._device_bufs: dict[int, Deque[np.ndarray]] = {}
        self._device_locks: dict[int, threading.Lock] = {}
        
        self._stop = threading.Event()
        self._threads: list[threading.Thread] = []
        self._mixer_thread: Optional[threading.Thread] = None

    def _make_callback(self, device_id: int):
        """Create a callback function for a specific device."""
        def callback(indata, frames, time_info, status):  # type: ignore[no-untyped-def]
            if status:
                logging.debug("Device %d audio status: %s", device_id, status)
            
            with self._device_locks[device_id]:
                # Ensure we always store as stereo for consistent processing
                if indata.ndim == 1 or indata.shape[1] == 1:
                    # Mono input - duplicate to stereo
                    stereo_data = np.column_stack([indata.flatten(), indata.flatten()])
                else:
                    # Already stereo or multi-channel - take first 2 channels
                    stereo_data = indata[:, :2] if indata.shape[1] > 2 else indata
                self._device_bufs[device_id].append(stereo_data.copy())
        
        return callback

    def _mixer_loop(self) -> None:
        """Mix audio from all device buffers into the master buffer."""
        while not self._stop.is_set():
            all_chunks = []
            
            # Collect chunks from all devices
            for device_id in self._device_bufs:
                with self._device_locks[device_id]:
                    if self._device_bufs[device_id]:
                        chunk = np.concatenate(list(self._device_bufs[device_id]), axis=0)
                        self._device_bufs[device_id].clear()
                        if chunk.size > 0:
                            all_chunks.append(chunk)
            
            if all_chunks:
                if len(all_chunks) == 1:
                    # Single device - use directly
                    mixed = all_chunks[0]
                else:
                    # Multi-device - simple concatenation
                    mixed = np.concatenate(all_chunks, axis=0)
                
                with self._mixed_lock:
                    self._mixed_buf.append(mixed)
            
            time.sleep(0.01)
    
    def start(self) -> None:
        """Start audio capture from one or more devices."""
        if self._threads:
            return  # Already started
        
        device_list = self.cfg.get_device_list()
        if not device_list:
            device_list = [self._resolve_legacy_device()]
        
        if len(device_list) == 1:
            # Single device - use original working approach
            self._start_single_device(device_list[0])
        else:
            # Multi-device - use subprocess approach
            self._start_multi_device_subprocess(device_list)
    
    def _start_single_device(self, device_id: int) -> None:
        """Start single device recording (original working code)."""
        devices = sd.query_devices()
        self._device_bufs[device_id] = deque(maxlen=int(10 * self.cfg.sample_rate))
        self._device_locks[device_id] = threading.Lock()
        
        def run() -> None:
            dev_info = devices[device_id]
            channels = dev_info['max_input_channels']
            callback = self._make_callback(device_id)
            
            stream_kwargs = dict(
                samplerate=self.cfg.sample_rate,
                blocksize=self.cfg.block_size,
                dtype=self.cfg.dtype,
                channels=channels,
                callback=callback,
                device=device_id,
            )
            
            if hasattr(sd, "WASAPI_LOOPBACK") and self.cfg.loopback:
                wasapi = sd.WasapiSettings(loopback=True)
                stream_kwargs["extra_settings"] = wasapi
            
            try:
                with sd.InputStream(**stream_kwargs):
                    logging.info("Audio stream started for device %d: %s", device_id, dev_info['name'])
                    while not self._stop.is_set():
                        time.sleep(0.05)
            except Exception as e:
                logging.error("Failed to start audio stream for device %d: %s", device_id, e)
        
        thread = threading.Thread(target=run, name=f"AudioRecorder-{device_id}", daemon=True)
        thread.start()
        self._threads.append(thread)
    
    def _start_multi_device_subprocess(self, device_list: list[int]) -> None:
        """Start multi-device recording using subprocess approach."""
        self._temp_files: list[Path] = []
        self._processes: list[subprocess.Popen] = []
        
        # Get device names for ffmpeg mapping
        devices = sd.query_devices()
        
        # Launch subprocess for each device
        for device_id in device_list:
            device_name = devices[device_id]['name']
            temp_file = Path(tempfile.gettempdir()) / f"device_{device_id}_audio.wav"
            self._temp_files.append(temp_file)
            
            # Map sounddevice name to ffmpeg device
            ffmpeg_device = self._get_ffmpeg_device_id(device_name)
            
            if ffmpeg_device is not None:
                cmd = [
                    'ffmpeg', '-y', '-f', 'avfoundation',
                    '-i', f':{ffmpeg_device}',  # Use mapped ffmpeg device ID
                    '-ac', '2', '-ar', str(self.cfg.sample_rate),
                    str(temp_file)
                ]
                
                try:
                    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    self._processes.append(proc)
                    logging.info("Started ffmpeg recording for device %d (%s) -> ffmpeg device %d", 
                               device_id, device_name, ffmpeg_device)
                except Exception as e:
                    logging.error("Failed to start ffmpeg for device %d: %s", device_id, e)
            else:
                logging.warning("Device %d (%s) not found in ffmpeg devices", device_id, device_name)
    
    def _get_ffmpeg_device_id(self, device_name: str) -> Optional[int]:
        """Map sounddevice name to ffmpeg avfoundation device ID."""
        try:
            result = subprocess.run([
                'ffmpeg', '-f', 'avfoundation', '-list_devices', 'true', '-i', ''
            ], capture_output=True, text=True, timeout=10)
            
            lines = result.stderr.split('\n')
            in_audio_section = False
            
            for line in lines:
                if 'AVFoundation audio devices:' in line:
                    in_audio_section = True
                    continue
                    
                if in_audio_section and '] [' in line and device_name in line:
                    # Extract device ID: [AVFoundation indev @ 0x...] [0] BlackHole 2ch
                    match = re.search(r'\[(\d+)\]\s+' + re.escape(device_name), line)
                    if match:
                        return int(match.group(1))
                        
        except Exception as e:
            logging.error("Failed to get ffmpeg device mapping: %s", e)
        
        return None
    
    def _resolve_legacy_device(self) -> int:
        """Resolve legacy device name/id to device number."""
        if not self.cfg.device:
            return sd.default.device[0] or 0  # Default input device
        
        devices = sd.query_devices()
        # Try as device name
        for i, dev in enumerate(devices):
            if dev['name'] == self.cfg.device:
                return i
        
        # Try as device ID
        try:
            device_id = int(self.cfg.device)
            if 0 <= device_id < len(devices):
                return device_id
        except ValueError:
            pass
        
        logging.warning("Device not found: %s, using default", self.cfg.device)
        return sd.default.device[0] or 0

    def stop(self) -> None:
        """Stop all audio capture threads, mixer, and subprocesses."""
        self._stop.set()
        
        # Stop all device capture threads
        for thread in self._threads:
            thread.join(timeout=2)
        self._threads.clear()
        
        # Stop mixer thread
        if self._mixer_thread is not None:
            self._mixer_thread.join(timeout=2)
            self._mixer_thread = None
            
        # Stop ffmpeg processes
        if hasattr(self, '_processes'):
            for proc in self._processes:
                try:
                    proc.terminate()
                    proc.wait(timeout=5)
                except:
                    pass
            self._processes.clear()
            
        # Clean up temp files
        if hasattr(self, '_temp_files'):
            for temp_file in self._temp_files:
                try:
                    if temp_file.exists():
                        temp_file.unlink()
                except:
                    pass
            self._temp_files.clear()

    def segment_generator(self) -> Iterable[Path]:
        """Yield temp WAV file paths for each overlapping segment."""
        if hasattr(self, '_temp_files'):
            # Multi-device subprocess mode - read from temp files
            yield from self._segment_generator_multidevice()
        else:
            # Single device mode - use original logic
            yield from self._segment_generator_single()
    
    def _segment_generator_single(self) -> Iterable[Path]:
        """Original single device segment generator."""
        segment_samples = int(self.segment_secs * self.cfg.sample_rate)
        hop_samples = int(max(1, (self.segment_secs - self.overlap_secs) * self.cfg.sample_rate))
        accum = np.zeros((0, 2), dtype=np.float32)
        last_emit = 0

        while not self._stop.is_set():
            device_id = list(self._device_bufs.keys())[0]
            with self._device_locks[device_id]:
                if self._device_bufs[device_id]:
                    chunk = np.concatenate(list(self._device_bufs[device_id]), axis=0)
                    self._device_bufs[device_id].clear()
                else:
                    chunk = np.zeros((0, 2), dtype=np.float32)
            
            if chunk.size:
                accum = np.concatenate([accum, chunk], axis=0)

            if (accum.shape[0] - last_emit >= hop_samples and accum.shape[0] >= segment_samples):
                start = accum.shape[0] - segment_samples
                seg = accum[start:accum.shape[0]]
                wav_path = self._write_temp_wav(seg, self.cfg.sample_rate)
                yield wav_path
                last_emit = accum.shape[0]

            time.sleep(0.1)
    
    def _segment_generator_multidevice(self) -> Iterable[Path]:
        """Multi-device segment generator using ffmpeg temp files."""
        
        while not self._stop.is_set():
            time.sleep(self.segment_secs - self.overlap_secs)  # Wait for segment duration
            
            # Combine all device audio files into one segment
            combined_file = Path(tempfile.gettempdir()) / f"combined_{uuid.uuid4().hex}.wav"
            
            # Use ffmpeg to mix all device files
            input_args = []
            for temp_file in self._temp_files:
                if temp_file.exists():
                    input_args.extend(['-i', str(temp_file)])
            
            if input_args:
                cmd = ['ffmpeg', '-y'] + input_args + [
                    '-filter_complex', f'amix=inputs={len(self._temp_files)}:duration=shortest',
                    '-ac', '1', '-ar', '16000',  # Mono, 16kHz for whisper
                    str(combined_file)
                ]
                
                try:
                    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=10)
                    if combined_file.exists():
                        yield combined_file
                except Exception as e:
                    logging.error("Failed to combine audio files: %s", e)

    @staticmethod
    def _write_temp_wav(audio_stereo: np.ndarray, sr: int) -> Path:
        mono = to_mono(audio_stereo)
        mono16 = linear_resample_mono(mono, sr, TARGET_SR)
        # Normalize softly to avoid clipping
        if mono16.size and np.max(np.abs(mono16)) > 0:
            mono16 = mono16 / np.max(np.abs(mono16)) * 0.95
        tmp = Path(tempfile.gettempdir()) / f"minute_taker_{uuid.uuid4().hex}.wav"
        sf.write(tmp.as_posix(), mono16, TARGET_SR, subtype="PCM_16")
        return tmp
