"""
Media handling for WebRTC (Video UI and Audio Streams).
Requires: pip install opencv-python av
"""

import asyncio
import queue
import threading
import cv2
import numpy as np
import sounddevice as sd
from PIL import Image, ImageDraw, ImageFont
from av import VideoFrame, AudioFrame
from aiortc.mediastreams import VideoStreamTrack, AudioStreamTrack, MediaStreamTrack


class CameraVideoStreamTrack(VideoStreamTrack):
    """Captures local webcam video and sends it via WebRTC."""
    kind = "video"

    def __init__(self):
        super().__init__()
        self.cap = cv2.VideoCapture(0)
        self._running = True

    async def recv(self) -> VideoFrame:
        pts, time_base = await self.next_timestamp()
        
        loop = asyncio.get_running_loop()
        
        def get_frame():
            ret, frame = self.cap.read()
            if not ret or frame is None:
                # Fallback black frame if camera is unavailable
                frame = np.zeros((480, 640, 3), dtype=np.uint8)
            return frame
            
        # Read from camera in an executor to avoid blocking the asyncio loop
        frame = await loop.run_in_executor(None, get_frame)
        
        # OpenCV reads BGR, av needs RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        vf = VideoFrame.from_ndarray(frame, format="rgb24")
        vf.pts = pts
        vf.time_base = time_base
        return vf

    def stop(self):
        super().stop()
        self._running = False
        if self.cap:
            self.cap.release()


class MicrophoneAudioTrack(AudioStreamTrack):
    """Captures local microphone audio and sends it via WebRTC."""
    kind = "audio"

    def __init__(self, sample_rate=48000):
        super().__init__()
        self.sample_rate = sample_rate
        self.audio_queue = queue.Queue(maxsize=100)
        self._running = True
        
        # Start a parallel sounddevice stream for WebRTC audio
        self.stream = sd.InputStream(
            samplerate=self.sample_rate, 
            channels=1, 
            dtype='int16', 
            callback=self._audio_callback
        )
        self.stream.start()

    def _audio_callback(self, indata, frames, time, status):
        if not self.audio_queue.full() and self._running:
            self.audio_queue.put_nowait(indata.copy())

    async def recv(self) -> AudioFrame:
        pts, time_base = await self.next_timestamp()
        
        loop = asyncio.get_running_loop()
        
        def get_audio():
            while self._running:
                try:
                    return self.audio_queue.get(timeout=0.1)
                except queue.Empty:
                    pass
            raise Exception("Track stopped")

        # Wait for audio data without blocking the event loop
        indata = await loop.run_in_executor(None, get_audio)

        # aiortc expects layout='mono' for 1 channel
        frame = AudioFrame.from_ndarray(indata.T, format='s16', layout='mono')
        frame.sample_rate = self.sample_rate
        frame.pts = pts
        frame.time_base = time_base
        return frame

    def stop(self):
        super().stop()
        self._running = False
        self.stream.stop()
        self.stream.close()


class VideoUI:
    """Displays remote WebRTC video and overlays subtitles using OpenCV."""
    def __init__(self, window_name="SherpaConnect Video"):
        self.window_name = window_name
        self._current_text = ""
        self._frame_queue = queue.Queue(maxsize=2)
        self._running = True
        
        # Try explicitly loading Windows fonts that support Devanagari (Hindi)
        font_paths = [
            "nirmala.ttf",
            r"C:\Windows\Fonts\Nirmala UI\Nirmala UI Regular.ttc",
            r"C:\Windows\Fonts\mangal.ttf"
        ]
        
        self._font = None
        for fp in font_paths:
            try:
                self._font = ImageFont.truetype(fp, 24)
                break
            except IOError:
                continue
                
        if self._font is None:
            print("⚠️ WARNING: Could not find Nirmala UI font! Hindi text may show as weird symbols.")
            self._font = ImageFont.load_default()

        self._display_thread = threading.Thread(target=self._display_loop, daemon=True)
        self._display_thread.start()

    def update_text(self, text: str):
        self._current_text = text

    async def consume_video_track(self, track: MediaStreamTrack):
        while self._running:
            try:
                frame = await track.recv()
                img = frame.to_ndarray(format="bgr24")
                if not self._frame_queue.full():
                    self._frame_queue.put_nowait(img)
            except Exception:
                break

    def stop(self):
        self._running = False

    def _display_loop(self):
        """OpenCV GUI loop (must run in its own thread)."""
        while self._running:
            try:
                img = self._frame_queue.get(timeout=0.1)
                
                # Draw the subtitles at the bottom
                if self._current_text:
                    # Convert OpenCV BGR array to PIL Image
                    img_pil = Image.fromarray(img)
                    draw = ImageDraw.Draw(img_pil)
                    
                    try:
                        bbox = draw.textbbox((0, 0), self._current_text, font=self._font)
                        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
                    except AttributeError:
                        tw, th = draw.textsize(self._current_text, font=self._font)
                        
                    x, y = 20, img.shape[0] - th - 30
                    # Black background rectangle for text readability
                    draw.rectangle((x - 5, y - 5, x + tw + 5, y + th + 5), fill=(0, 0, 0))
                    # White text
                    draw.text((x, y), self._current_text, font=self._font, fill=(255, 255, 255))
                    
                    # Convert back to OpenCV array
                    img = np.array(img_pil)
                
                cv2.imshow(self.window_name, img)
            except queue.Empty:
                pass
            
            # Poll window events
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
        cv2.destroyAllWindows()


class AudioReceiver:
    """Plays remote WebRTC audio. Mutes automatically if TTS is preferred."""
    def __init__(self, sample_rate=48000):
        self._running = True
        self.muted = False

    def set_mute(self, muted: bool):
        """Enable or disable remote WebRTC audio."""
        self.muted = muted

    async def consume_audio_track(self, track: MediaStreamTrack):
        """Receives audio frames from aiortc and plays them via sounddevice."""
        
        audio_queue = queue.Queue(maxsize=20)
        
        def audio_callback(outdata, frames, time, status):
            try:
                data = audio_queue.get_nowait()
                if self.muted:
                    outdata.fill(0)
                else:
                    if len(data) < len(outdata):
                        outdata[:len(data)] = data
                        outdata[len(data):].fill(0)
                    else:
                        outdata[:] = data[:len(outdata)]
            except queue.Empty:
                outdata.fill(0)

        stream = sd.OutputStream(samplerate=48000, channels=1, dtype='int16', callback=audio_callback)
        stream.start()

        while self._running:
            try:
                frame = await track.recv()
                if self.muted:
                    continue
                    
                nda = frame.to_ndarray()
                if nda.shape[0] > 1:
                    nda = nda[0:1] # Extract mono channel
                
                if not audio_queue.full():
                    audio_queue.put(nda.T)
            except Exception:
                break
                
        stream.stop()
        stream.close()

    def stop(self):
        self._running = False