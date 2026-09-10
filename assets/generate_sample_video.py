"""
Generates a sample demonstration video clip for testing touchless video player controls.
"""

from pathlib import Path
import cv2
import numpy as np

OUTPUT_DIR = Path(__file__).resolve().parent
OUTPUT_FILE = OUTPUT_DIR / "sample_video.mp4"


def generate_sample_video(filename: Path = OUTPUT_FILE, duration_sec: int = 15, fps: int = 30):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if filename.exists() and filename.stat().st_size > 10000:
        print(f"[SampleVideo] Video already exists at {filename}")
        return filename

    print(f"[SampleVideo] Generating sample demo video: {filename}...")
    width, height = 640, 360
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(filename), fourcc, fps, (width, height))

    total_frames = duration_sec * fps
    for i in range(total_frames):
        # Create dark background gradient
        progress = i / total_frames
        sec = i / fps

        # Background color shifts smoothly
        b = int(25 + 15 * np.sin(progress * np.pi * 2))
        g = int(20 + 20 * np.cos(progress * np.pi * 2))
        r = int(35 + 25 * np.sin(progress * np.pi * 4))
        frame = np.full((height, width, 3), (b, g, r), dtype=np.uint8)

        # Draw a moving glowing neon circle
        cx = int(width * 0.2 + (width * 0.6) * (0.5 + 0.5 * np.sin(progress * np.pi * 4)))
        cy = int(height * 0.45 + 40 * np.cos(progress * np.pi * 6))
        cv2.circle(frame, (cx, cy), 36, (0, 240, 255), -1, cv2.LINE_AA)
        cv2.circle(frame, (cx, cy), 46, (0, 180, 255), 3, cv2.LINE_AA)

        # Title text
        cv2.putText(
            frame,
            "AI GESTURE TOUCHLESS VIDEO DEMO",
            (width // 2 - 200, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        # Current time display
        time_text = f"Time: {int(sec // 60):02d}:{int(sec % 60):02d} / 00:{duration_sec:02d}"
        cv2.putText(
            frame,
            time_text,
            (width // 2 - 90, height - 65),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (200, 230, 255),
            1,
            cv2.LINE_AA,
        )

        # Touchless instructions overlay
        instructions = "Palm = Play | Fist = Pause | Swipe R = +5s | Swipe L = -5s"
        cv2.putText(
            frame,
            instructions,
            (width // 2 - 220, height - 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (150, 170, 200),
            1,
            cv2.LINE_AA,
        )

        # Progress bar
        bar_x = 40
        bar_y = height - 15
        bar_w = width - 80
        bar_h = 6
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (60, 60, 80), -1)
        filled_w = int(bar_w * progress)
        if filled_w > 0:
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + filled_w, bar_y + bar_h), (0, 242, 254), -1)

        out.write(frame)

    out.release()
    print(f"[SampleVideo] Sample video generated successfully: {filename} ({filename.stat().st_size} bytes)")
    return filename


if __name__ == "__main__":
    generate_sample_video()
