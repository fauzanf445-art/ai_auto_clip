from pathlib import Path
from typing import Optional, Callable, Tuple, Any

import cv2
from mediapipe.tasks.python.core.base_options import BaseOptions
from mediapipe.tasks.python.vision.face_landmarker import FaceLandmarker, FaceLandmarkerOptions
from mediapipe.tasks.python.vision.core.vision_task_running_mode import VisionTaskRunningMode
from mediapipe.tasks.python.vision.core.image import Image, ImageFormat

import numpy as np

from src.application.context import SessionContext
from src.domain.interfaces import IMediapipeAdapter, TrackResult, ILogger
from src.domain.exceptions import VideoProcessingError

class MediaPipeAdapter(IMediapipeAdapter):
    """
    Implementasi IFaceTracker menggunakan MediaPipe Face Landmarker (Tasks API).
    """

    def __init__(self, model_path: str, logger: ILogger):
        self.model_path = model_path
        self.logger = logger

        self._verified_delegate: Optional[BaseOptions.Delegate] = None
        self._landmarker: Optional[FaceLandmarker] = None

    def _setup_hardware_delegate(self, ctx: SessionContext) -> BaseOptions.Delegate:
        """Mendeteksi hardware terbaik (GPU/CPU) satu kali dan menyimpan hasilnya."""
        if self._verified_delegate is not None:
            return self._verified_delegate

        if not Path(self.model_path).exists():
            raise VideoProcessingError(f"Model MediaPipe tidak ditemukan di: {self.model_path}")

        try:
            # Uji validitas GPU dengan membuat instance dummy singkat
            ctx.logger.debug("🔍 MediaPipe: Memverifikasi dukungan GPU...")
            base_options = BaseOptions(model_asset_path=self.model_path, delegate=BaseOptions.Delegate.GPU)
            options = FaceLandmarkerOptions(base_options=base_options, running_mode=VisionTaskRunningMode.VIDEO)
            test_task = FaceLandmarker.create_from_options(options)
            test_task.close()
            
            self._verified_delegate = BaseOptions.Delegate.GPU
            ctx.logger.info("🚀 MediaPipe: GPU Delegate diverifikasi dan diaktifkan.")
        except Exception as e:
            ctx.logger.warning(f"⚠️ MediaPipe GPU tidak didukung atau gagal inisialisasi. Menggunakan CPU. Detail: {e}")
            self._verified_delegate = BaseOptions.Delegate.CPU
        
        return self._verified_delegate

    def _create_options(self, delegate: BaseOptions.Delegate) -> FaceLandmarkerOptions:
        """Helper privat untuk membangun konfigurasi FaceLandmarkerOptions."""
        return FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=self.model_path, delegate=delegate),
            running_mode=VisionTaskRunningMode.VIDEO,
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5
        )

    def ensure_model(self, ctx: SessionContext) -> None:
        """
        Memastikan resource siap. Karena kita butuh reset timestamp, 
        metode ini sekarang fokus pada penyiapan hardware delegate dan 
        inisialisasi instance landmarker baru.
        """
        delegate = self._setup_hardware_delegate(ctx)
        
        # Tutup instance lama jika ada sebelum membuat yang baru (Reset State)
        if self._landmarker:
            self.close(ctx)
            
        options = self._create_options(delegate)
        self._landmarker = FaceLandmarker.create_from_options(options)
        ctx.logger.debug("🆕 MediaPipe: Instance Landmarker baru dibuat (Timestamp Reset).")

    def close(self, ctx: SessionContext) -> None:
        """Membersihkan resource MediaPipe secara formal."""
        if self._landmarker:
            try:
                self._landmarker.close()
                ctx.logger.debug("🧹 MediaPipe Landmarker ditutup.")
            except Exception as e:
                ctx.logger.warning(f"⚠️ Gagal menutup MediaPipe Landmarker: {e}")
            finally:
                self._landmarker = None

    def track_and_crop(self, ctx: SessionContext, input_path: str, output_path: str, progress_callback: Optional[Callable[[int, int], None]] = None) -> TrackResult:
        # Pastikan model sudah dimuat
        self.ensure_model(ctx)
        
        if not self._landmarker:
            raise VideoProcessingError("Gagal menginisialisasi MediaPipe Landmarker.")

        cap = None
        out = None

        try:
            cap = cv2.VideoCapture(input_path)
            if not cap.isOpened():
                raise VideoProcessingError(f"Gagal membuka video input: {input_path}")

            # Properti Video Asli
            orig_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            orig_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            if fps <= 0:
                fps = 30.0 # Fallback jika FPS tidak valid
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            # Validasi dimensi video
            if orig_width == 0 or orig_height == 0:
                raise VideoProcessingError(f"Dimensi video tidak valid ({orig_width}x{orig_height}) pada {input_path}")

            # Setup Output Video (9:16)
            target_aspect_ratio = 9 / 16
            out_height = orig_height
            out_width = int(out_height * target_aspect_ratio)
            
            # Setup Video Writer
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            fourcc = cv2.VideoWriter.fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (out_width, out_height))
            
            # Tracking Variables
            frame_idx = 0
            last_timestamp_ms = -1
            
            # EMA (Exponential Moving Average) Variables
            ema_alpha = 0.2  # Faktor smoothing (0.0 - 1.0)
            current_center_x = float(orig_width // 2) # Mulai dari tengah
            actual_crop_width = min(out_width, orig_width)
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                # 1. Hitung timestamp (Monotonically Increasing)
                timestamp_ms = self._get_timestamp_ms(frame_idx, fps, last_timestamp_ms)
                last_timestamp_ms = timestamp_ms

                # 2. Deteksi pada setiap frame
                detection_result = self._process_detection(frame, timestamp_ms)

                # 3. Update Center dengan Landmark Hidung (Index 4) & EMA
                current_center_x = self._update_ema_center(detection_result, orig_width, current_center_x, ema_alpha)

                # 4. Hitung Koordinat Crop dengan presisi dan batasan
                x1, x2 = self._calculate_crop_boundaries(current_center_x, actual_crop_width, orig_width)

                # 5. Crop & Resize Cerdas (Hanya jika perlu)
                # Selalu resize untuk memastikan dimensi output konsisten.
                final_frame = cv2.resize(frame[:, x1:x2], (out_width, out_height), interpolation=cv2.INTER_AREA)
                
                out.write(final_frame)

                frame_idx += 1
                if progress_callback:
                    progress_callback(frame_idx, total_frames)

            return TrackResult(
                tracked_video=str(output_path),
                width=out_width,
                height=out_height
            )

        except Exception as e:
            ctx.logger.error(f"Error during video processing: {e}", exc_info=True)
            raise VideoProcessingError(f"Gagal melakukan tracking wajah: {e}", original_exception=e)
        finally:
            # Penting: Tutup landmarker setelah selesai satu file video
            self.close(ctx)
            if cap:
                cap.release()
            if out:
                out.release()

    def _get_timestamp_ms(self, frame_idx: int, fps: float, last_timestamp_ms: int) -> int:
        """Hitung timestamp yang selalu naik untuk MediaPipe."""
        timestamp_ms = int((frame_idx * 1000) / fps)
        if timestamp_ms <= last_timestamp_ms:
            timestamp_ms = last_timestamp_ms + 1
        return timestamp_ms

    def _process_detection(self, frame: np.ndarray, timestamp_ms: int) -> Any:
        """Menjalankan inferensi MediaPipe pada frame tunggal."""
        if self._landmarker is None:
            raise VideoProcessingError("MediaPipe Landmarker belum diinisialisasi.")

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = Image(image_format=ImageFormat.SRGB, data=rgb_frame)
        return self._landmarker.detect_for_video(mp_image, timestamp_ms)

    def _update_ema_center(self, detection_result: Any, orig_width: int, current_center_x: float, ema_alpha: float) -> float:
        """Menghitung posisi tengah horizontal baru menggunakan EMA."""
        if not detection_result.face_landmarks:
            return current_center_x
        
        nose_landmark = detection_result.face_landmarks[0][4]
        target_x = nose_landmark.x * orig_width
        return (ema_alpha * target_x) + ((1 - ema_alpha) * current_center_x)

    def _calculate_crop_boundaries(self, center_x: float, crop_width: int, orig_width: int) -> Tuple[int, int]:
        """Menghitung batas kiri dan kanan crop dengan clamping agar tidak keluar frame."""
        x1 = int(center_x - crop_width / 2)
        x1 = max(0, min(x1, orig_width - crop_width))
        return x1, x1 + crop_width