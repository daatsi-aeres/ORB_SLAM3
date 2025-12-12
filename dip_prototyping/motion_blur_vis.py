import numpy as np
import cv2
import pyrealsense2 as rs
import time
from collections import deque
from numpy.fft import fft2, ifft2

# ---------------------------------------------------------
# HELPER CLASSES & FUNCTIONS
# ---------------------------------------------------------

class Smoother:
    """Helps smooth out jittery numbers (like FPS or Feature Counts)."""
    def __init__(self, window_size=10):
        self.data = deque(maxlen=window_size)

    def update(self, value):
        self.data.append(value)
        return sum(self.data) / len(self.data)

def get_motion_label(gyro_vec):
    """
    Strictly classifies motion into HORIZONTAL or VERTICAL based on dominant axis.
    """
    gx, gy, gz = gyro_vec
    
    # gx = Pitch (Vertical Tilting)
    # gz = Yaw (Horizontal Panning)
    
    # Simple noise floor to prevent flickering when completely still
    if abs(gx) < 0.1 and abs(gz) < 0.1:
        return "STATIONARY"

    # Compare absolute magnitude to find dominant direction
    if abs(gz) >= abs(gx):
        return "HORIZONTAL"
    else:
        return "VERTICAL"

def init_realsense():
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.gyro)
    config.enable_stream(rs.stream.accel)
    profile = pipeline.start(config)
    return pipeline, profile

def is_blurry(gray, thresh=200.0):
    """
    Check variance of Laplacian.
    UPDATED: Default threshold raised to 200.0 as requested.
    """
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    score = lap.var()
    return score < thresh, score

def motion_psf(length=7, angle=0):
    EPS = 1e-6
    psf = np.zeros((length, length), dtype=np.float32)
    center = length // 2
    rad = np.deg2rad(angle)
    x0 = center + (length//2 - 1) * np.cos(rad)
    y0 = center + (length//2 - 1) * np.sin(rad)
    x1 = center - (length//2 - 1) * np.cos(rad)
    y1 = center - (length//2 - 1) * np.sin(rad)
    cv2.line(psf, (int(x0), int(y0)), (int(x1), int(y1)), 1, 1)
    psf = psf / (psf.sum() + EPS)
    return psf

def wiener_deblur(img, psf, K=0.05):
    img_float = img.astype(np.float32) / 255.0
    h, w = img.shape
    psf_padded = np.zeros((h, w), np.float32)
    ph, pw = psf.shape
    cy, cx = ph // 2, pw // 2
    psf_padded[h//2 - cy : h//2 - cy + ph, w//2 - cx : w//2 - cx + pw] = psf
    psf_shifted = np.fft.fftshift(psf_padded)
    psf_fft = np.fft.fft2(psf_shifted)
    img_fft = np.fft.fft2(img_float)
    EPS = 1e-8
    wiener = np.conj(psf_fft) / (np.abs(psf_fft)**2 + K + EPS)
    result_fft = wiener * img_fft
    result = np.fft.ifft2(result_fft)
    result = np.abs(result)
    return (result * 255).clip(0, 255).astype(np.uint8)

def imu_yaw_from_gyro(gyro_data, dt, yaw_prev):
    # Integrate Z-axis gyro
    yaw = yaw_prev + gyro_data[2] * dt * (180.0 / np.pi)
    return yaw

def draw_text_column(img, text_lines, x_start, y_start_override=None, color=(255, 255, 255)):
    y_start = y_start_override if y_start_override else 25
    line_height = 25
    for i, line in enumerate(text_lines):
        cv2.putText(img, line, (x_start, y_start + (i * line_height)), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)

# ---------------------------------------------------------
# MAIN LOOP
# ---------------------------------------------------------

def main():
    pipeline, profile = init_realsense()
    orb = cv2.ORB_create(nfeatures=1000)
    
    yaw = 0
    last_time = None
    
    # Smoothers
    fps_raw_avg = Smoother(20)
    fps_deb_avg = Smoother(20)
    orb_raw_avg = Smoother(10)
    orb_deb_avg = Smoother(10)
    
    prev_time_raw = time.time()
    prev_time_deblur = time.time()

    frame_w, frame_h = 640, 480
    header_h = 50   
    footer_h = 100  
    total_h = frame_h + header_h + footer_h
    
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter("deblur_2modes.mp4", fourcc, 30, (frame_w * 2, total_h)) 
    
    print("Running with Threshold=200. Press 'q' to stop.")

    try:
        while True:
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            gyro_frame = frames.first_or_default(rs.stream.gyro)

            if not color_frame: continue

            frame = np.asanyarray(color_frame.get_data())
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # --- Gyro Logic ---
            gyro_vec = np.array([0.0, 0.0, 0.0])
            if last_time is None:
                last_time = gyro_frame.get_timestamp()
            else:
                new_time = gyro_frame.get_timestamp()
                dt = (new_time - last_time) / 1000.0
                last_time = new_time
                gyro = gyro_frame.as_motion_frame().get_motion_data()
                gyro_vec = np.array([gyro.x, gyro.y, gyro.z])
                yaw = imu_yaw_from_gyro(gyro_vec, dt, yaw)

            motion_status = get_motion_label(gyro_vec)

            # --- Blur Check (Threshold = 200) ---
            is_blur, score = is_blurry(gray, thresh=60.0)

            # ---------------------------
            # LEFT SIDE: RAW
            # ---------------------------
            t_raw = time.time()
            kp_raw = orb.detect(gray, None)
            kp_raw = sorted(kp_raw, key=lambda x: x.response, reverse=True)[:500]
            kp_raw, des_raw = orb.compute(gray, kp_raw)
            raw_out = cv2.drawKeypoints(gray, kp_raw, None, color=(255, 100, 100))
            
            current_fps_raw = 1.0 / (t_raw - prev_time_raw + 1e-9)
            smooth_fps_raw = fps_raw_avg.update(current_fps_raw)
            smooth_orb_raw = orb_raw_avg.update(len(kp_raw))
            prev_time_raw = t_raw

            # ---------------------------
            # RIGHT SIDE: DEBLUR
            # ---------------------------
            if is_blur:
                psf = motion_psf(length=9, angle=yaw)
                gray_deblur = wiener_deblur(gray, psf)
                status_str = "ACTIVE (Filtering)"
                status_color = (100, 255, 100) # Green
            else:
                gray_deblur = gray
                status_str = "INACTIVE (Sharp)"
                status_color = (150, 150, 150) # Gray

            t_de = time.time()
            kp_deb = orb.detect(gray_deblur, None)
            kp_deb = sorted(kp_deb, key=lambda x: x.response, reverse=True)[:500]
            kp_deb, des = orb.compute(gray_deblur, kp_deb)
            deb_out = cv2.drawKeypoints(gray_deblur, kp_deb, None, color=(0, 255, 0))
            
            current_fps_deb = 1.0 / (t_de - prev_time_deblur + 1e-9)
            smooth_fps_deb = fps_deb_avg.update(current_fps_deb)
            smooth_orb_deb = orb_deb_avg.update(len(kp_deb))
            prev_time_deblur = t_de

            # ---------------------------
            # UI CONSTRUCTION
            # ---------------------------
            
            # 1. TOP HEADER (Gyro Data)
            header = np.zeros((header_h, frame_w * 2, 3), dtype=np.uint8)
            
            # Simplified Header Text
            header_text = f"MOTION MODE: {motion_status}   |   Gyro Z (Horiz): {gyro_vec[2]:.2f}   |   Gyro X (Vert): {gyro_vec[0]:.2f}"
            cv2.putText(header, header_text, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # 2. BOTTOM FOOTER (Stats)
            footer = np.zeros((footer_h, frame_w * 2, 3), dtype=np.uint8)
            col_data = (200, 200, 200)
            
            # Footer Left (Raw)
            lines_left = [
                f"RAW INPUT",
                f"FPS: {int(smooth_fps_raw)}",
                f"Features: {int(smooth_orb_raw)}",
            ]
            draw_text_column(footer, lines_left, 20, y_start_override=25, color=col_data)

            # Footer Right (Deblur)
            lines_right = [
                f"DEBLURRED INPUT",
                f"FPS: {int(smooth_fps_deb)} | Features: {int(smooth_orb_deb)}",
                f"Blur Score: {int(score)} (Thresh: 60qq)",
            ]
            draw_text_column(footer, lines_right, 640 + 20, y_start_override=25, color=col_data)
            
            # STATUS INDICATOR
            cv2.putText(footer, f"DEBLUR STATUS: {status_str}", (640 + 300, 75), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)

            # 3. STACK EVERYTHING
            combined_imgs = np.hstack((raw_out, deb_out))
            final_display = np.vstack((header, combined_imgs, footer))
            
            # Separator line
            cv2.line(final_display, (640, header_h), (640, total_h), (100, 100, 100), 1)

            cv2.imshow("IMU Deblur - 2 Modes", final_display)
            writer.write(final_display)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()
        writer.release()
        print("Saved video to: deblur_2modes.mp4")

if __name__ == "__main__":
    main()