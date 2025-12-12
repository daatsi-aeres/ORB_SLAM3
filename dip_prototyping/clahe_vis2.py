import cv2
import numpy as np
import pyrealsense2 as rs
import time
import sys

def init_realsense():
    """Initialize the RealSense pipeline."""
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    
    # Start pipeline
    profile = pipeline.start(config)
    return pipeline, profile

def choose_clahe_params(gray):
    """
    Determine CLAHE parameters dynamically based on image variance 
    and dynamic range.
    """
    var = np.var(gray)
    range_ = np.max(gray) - np.min(gray)

    # Default initialization
    clipLimit = 4.0
    tileGridSize = (8, 8)
    status_msg = "Normal Indoor"
    do_clahe = False

    # Case 1: Extremely low contrast (Dark room) -> Aggressive CLAHE
    if var < 150 and range_ < 40:
        status_msg = "Low Light Boost"
        clipLimit = 6.0
        tileGridSize = (4, 4)
        do_clahe = True

    # Case 2: Medium low contrast -> Moderate CLAHE
    elif var < 300:
        clipLimit = 3.0
        tileGridSize = (8, 8)
        status_msg = "Low Light Boost"
        do_clahe = True

    # Case 3: High contrast -> Weak CLAHE (Prevent noise amplification)
    else:
        clipLimit = 2.0
        tileGridSize = (12, 12)
        status_msg = "HDR/ Glare Mode"
        do_clahe = True

    return {"clip": clipLimit, "grid": tileGridSize, "status": status_msg, "do_clahe": do_clahe}

def apply_clahe(gray):
    """Apply CLAHE using the calculated parameters."""
    params = choose_clahe_params(gray)
    clip = params["clip"]
    grid = params["grid"]
    
    # Create and apply CLAHE
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=grid)
    return clahe.apply(gray)

def guided_filter(gray, radius=16, eps=50):
    """
    Apply Edge-Preserving Guided Filter.
    Requires opencv-contrib-python.
    """
    try:
        return cv2.ximgproc.guidedFilter(
            guide=gray,
            src=gray,
            radius=radius,
            eps=eps
        )
    except AttributeError:
        # Fallback if contrib is not installed
        return gray 

def draw_info_box_auto(img, title, data_lines, x, y,
                       color_header=(0,255,255),
                       line_height=25, padding_left=10,
                       padding_right=20, padding_top=40, padding_bottom=20):
    """
    Draws a dynamic information box on the image frame.
    Returns the calculated width of the box to allow stacking.
    """

    # --- Compute required text width ---
    max_text_width = 0
    for key, value in data_lines:
        key_size = cv2.getTextSize(f"{key}:", cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0][0]
        val_size = cv2.getTextSize(str(value), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0][0]
        max_text_width = max(max_text_width, key_size + 30 + val_size)

    # Final box width
    width = padding_left + max_text_width + padding_right + 10

    # Dynamic height
    height = padding_top + padding_bottom + line_height * len(data_lines)

    # Background (Dark Grey with lighter border)
    cv2.rectangle(img, (x, y), (x + width, y + height), (30, 30, 30), -1)
    cv2.rectangle(img, (x, y), (x + width, y + height), (100, 100, 100), 1)

    # Header Title
    cv2.putText(img, title, (x + padding_left, y + 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color_header, 2)

    # Draw Data Lines
    y_offset = y + padding_top
    key_x = x + padding_left
    value_x = x + padding_left + 140   # Default spacing for values

    for key, value in data_lines:
        # Draw Key (Gray)
        cv2.putText(img, f"{key}:", (key_x, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1)
        # Draw Value (Green)
        cv2.putText(img, str(value), (value_x, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
        y_offset += line_height

    return width   # Return width so next box knows where to start

# ---------------------------------------------------------
# MAIN EXECUTION LOOP
# ---------------------------------------------------------
def main():
    pipeline, profile = init_realsense()
    
    # Guided Filter Constants
    GF_RADIUS = 16
    GF_EPS = 50

    # ORB Initialization (High limit as requested)
    # We use 10000 to effectively have "no limit" for this resolution
    orb = cv2.ORB_create(nfeatures=10000)
    show_orb = False  # Toggle state

    print("Pipeline started.")
    print(" Controls: 'W' = Show ORB Features | 'E' = Hide ORB Features | 'Q' = Quit")

    try:
        while True:
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue

            # Convert to numpy arrays
            frame = np.asanyarray(color_frame.get_data())
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Get Params
            clipandgrid = choose_clahe_params(gray)

            # ---------- 1. Apply CLAHE ----------
            clahe_out = apply_clahe(gray)

            # ---------- 2. Apply Guided Filter ----------
            gf_out = guided_filter(clahe_out, radius=GF_RADIUS, eps=GF_EPS)

            # ---------- Prepare Visualization ----------
            gray_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            clahe_bgr = cv2.cvtColor(clahe_out, cv2.COLOR_GRAY2BGR)
            gf_bgr = cv2.cvtColor(gf_out, cv2.COLOR_GRAY2BGR)

            # ---------- 3. ORB Feature Detection (Toggleable) ----------
            count_raw = "OFF"
            count_clahe = "OFF"
            count_gf = "OFF"

            if show_orb:
                # Detect on all 3
                kp_raw = orb.detect(gray, None)
                kp_clahe = orb.detect(clahe_out, None)
                kp_gf = orb.detect(gf_out, None)

                # Update Counts
                count_raw = len(kp_raw)
                count_clahe = len(kp_clahe)
                count_gf = len(kp_gf)

                # Draw Keypoints (Green)
                # Note: Drawing directly on the BGR images
                gray_bgr = cv2.drawKeypoints(gray_bgr, kp_raw, None, color=(0,255,0), flags=0)
                clahe_bgr = cv2.drawKeypoints(clahe_bgr, kp_clahe, None, color=(0,255,0), flags=0)
                gf_bgr = cv2.drawKeypoints(gf_bgr, kp_gf, None, color=(0,255,0), flags=0)

            # Add headers to images (Overlay text on top of video)
            cv2.putText(gray_bgr, "1. RAW GRAYSCALE", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(clahe_bgr, "2. ADAPTIVE CLAHE", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(gf_bgr, "3. GUIDED FILTER", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            # ---------- Expand frames vertically for info box ----------
            H, W = gray.shape
            extra_height = 200
            gray_bgr = cv2.copyMakeBorder(gray_bgr, 0, extra_height, 0, 0, cv2.BORDER_CONSTANT, value=(20,20,20))
            clahe_bgr = cv2.copyMakeBorder(clahe_bgr, 0, extra_height, 0, 0, cv2.BORDER_CONSTANT, value=(20,20,20))
            gf_bgr = cv2.copyMakeBorder(gf_bgr, 0, extra_height, 0, 0, cv2.BORDER_CONSTANT, value=(20,20,20))

            # ---------- Calculate Stats ----------
            metrics = {
                "L_mean": np.mean(gray)/255,
                "p10": np.percentile(gray, 10)/255,
                "p90": np.percentile(gray, 90)/255,
                "dark_frac": np.sum(gray < 50)/gray.size,
                "bright_frac": np.sum(gray > 200)/gray.size,
                "status": clipandgrid["status"],
                "do_clahe": clipandgrid["do_clahe"],
                "clip": clipandgrid["clip"],
                "grid": clipandgrid["grid"]
            }

            # --- Data for Common Metrics ---
            box_scene_data = [
                ("Mean Lum (0-1)", f"{metrics['L_mean']:.3f}"),
                ("P10 (Shadows)",  f"{metrics['p10']:.3f}"),
                ("P90 (Highlights)", f"{metrics['p90']:.3f}"),
                ("Dark Tiles %",   f"{metrics['dark_frac']*100:.1f}%"),
                ("Bright Tiles %", f"{metrics['bright_frac']*100:.1f}%")
            ]

            # --- Data for CLAHE Box ---
            box_clahe_data = [
                ("Algorithm State", metrics["status"]),
                ("Active?",        "YES" if metrics["do_clahe"] else "NO"),
                ("Clip Limit",     f"{metrics['clip']:.2f}"),
                ("Kernel Size",    f"{metrics['grid']} x {metrics['grid']}"),
                ("ORB Detected",   f"{count_clahe}")  # <--- ADDED HERE
            ]
            
            # --- Data for Guided Filter Box ---
            box_gf_data = [
                ("Radius (r)",     f"{GF_RADIUS}"),
                ("Epsilon (eps)",  f"{GF_EPS}"),
                ("Type",           "Edge Preserving"),
                ("Input",          "CLAHE Result"),
                ("ORB Detected",   f"{count_gf}")     # <--- ADDED HERE
            ]

            # ---------- Draw info boxes ----------
            
            # 1. Raw Feed: 
            # Manually print ORB count in the footer area (since no box exists)
            cv2.putText(gray_bgr, f"RAW ORB COUNT: {count_raw}", (20, H + 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # 2. CLAHE Feed: 
            # Scene Metrics + CLAHE Params
            box1_w = draw_info_box_auto(clahe_bgr, "SCENE METRICS", box_scene_data,
                                        10, H + 10, color_header=(255,191,0))
            draw_info_box_auto(clahe_bgr, "CLAHE PARAMS", box_clahe_data,
                               10 + box1_w + 20, H + 10, color_header=(0,191,255))

            # 3. Guided Filter Feed:
            # Scene Metrics + Guided Filter Params
            box1_w = draw_info_box_auto(gf_bgr, "SCENE METRICS", box_scene_data,
                                        10, H + 10, color_header=(255,191,0))
            draw_info_box_auto(gf_bgr, "FILTER PARAMS", box_gf_data,
                               10 + box1_w + 20, H + 10, color_header=(255,0,255)) 

            # ---------- Stack and Show ----------
            combined = np.hstack([gray_bgr, clahe_bgr, gf_bgr])
            cv2.imshow("Raw vs CLAHE vs GuidedFilter", combined)

            # Key Handling
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('w'):
                show_orb = True
                print("ORB ON")
            elif key == ord('e'):
                show_orb = False
                print("ORB OFF")

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()