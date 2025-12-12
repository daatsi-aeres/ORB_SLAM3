import pyrealsense2 as rs
import numpy as np
import cv2
import time

# ==============================================================================
# 1. ADAPTIVE LOGIC (Ported from your C++ Code)
# ==============================================================================
def analyze_and_get_clahe_params(gray_img):
    h, w = gray_img.shape
    sub_img = gray_img[::4, ::4] 
    
    # --- 1. Metrics ---
    L_mean = np.mean(sub_img) / 255.0
    
    # Use p99 instead of p90 to catch small bright lights/windows
    p99 = np.percentile(sub_img, 99) / 255.0
    p10 = np.percentile(sub_img, 10) / 255.0
    
    # Calculate "Saturation": Fraction of pixels that are blown out (near 255)
    # This is the truest indicator of glare
    saturated_pixels = np.sum(sub_img > 250)
    saturation_frac = saturated_pixels / sub_img.size

    # --- 2. Decision Logic ---
    
    # DEFAULT: Standard indoor setting
    do_clahe = True
    clip_limit = 2.0
    grid_size = 8 
    status_msg = "Normal Indoor"

    # CONDITION A: Low Light (Your existing logic - kept as is)
    # If generally dark and flat
    if (L_mean < 0.4) and (p99 < 0.6): 
        clip_limit = 5.3
        grid_size = 8
        status_msg = "Low Light Boost"

    # CONDITION B: GLARE / HIGH DYNAMIC RANGE (The Fix)
    # Trigger: If we have PURE WHITE pixels (p99 is high) but the room isn't washed out (p10 is low)
    elif (p99 > 0.95) or (saturation_frac > 0.02):
        # STRATEGY FOR WINDOWS:
        # We need HIGH clip limit to see the dark wall next to the window.
        # But we need SMALL tiles (8x8) to isolate the window. 
        # If we use big tiles (32x32), the window's brightness will darken the whole wall.
        
        do_clahe = True
        clip_limit = 4.0  # High contrast to bring out the wall details
        grid_size = 4     # VERY SMALL tiles. This creates a "cage" around the window.
        status_msg = "HDR/Window Glare Mode"

    return {
        "do_clahe": do_clahe,
        "clip": clip_limit,
        "grid": grid_size,
        "L_mean": L_mean,
        "p10": p10,
        "p90": p99, # Displaying p99 instead of p90 for your chart
        "dark_frac": saturation_frac, # Re-purposing this slot to show saturation %
        "bright_frac": 0.0,
        "status": status_msg
    }

# ==============================================================================
# 2. VISUALIZATION HELPERS
# ==============================================================================
def draw_info_box(img, title, data_lines, x, y, width, height, color_header=(0, 255, 255)):
    """ Draws a stylish data box on the dashboard """
    # Background
    cv2.rectangle(img, (x, y), (x + width, y + height), (30, 30, 30), -1)
    cv2.rectangle(img, (x, y), (x + width, y + height), (100, 100, 100), 1)
    
    # Header
    cv2.putText(img, title, (x + 10, y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color_header, 2)
    
    # Data Lines
    y_offset = y + 55
    for line in data_lines:
        key, value = line
        # Key text
        cv2.putText(img, f"{key}:", (x + 10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        # Value text (Colored based on value usually, but keeping white/green for simplicity)
        cv2.putText(img, str(value), (x + 140, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        y_offset += 25

# ==============================================================================
# 3. MAIN LOOP
# ==============================================================================
def main():
    # Configure RealSense Pipeline 
    # NOTE: For D435i Stereo SLAM, the standard is Infrared 1 with Emitter OFF.
    # This provides Global Shutter grayscale without the structured light dots.
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.infrared, 1, 640, 480, rs.format.y8, 30)
    
    print("Starting RealSense...")
    try:
        pipeline_profile = pipeline.start(config)
        
        # --- CRITICAL: DISABLE EMITTER TO REMOVE DOTS ---
        # This ensures we get the "plain grayscale" image SLAM uses.
        device = pipeline_profile.get_device()
        depth_sensor = device.first_depth_sensor()
        if depth_sensor.supports(rs.option.emitter_enabled):
            depth_sensor.set_option(rs.option.emitter_enabled, 0) # 0 = Off
            print("Emitter Disabled (Clean Grayscale Mode)")
            
    except RuntimeError:
        print("RealSense not found! Exiting.")
        return

    try:
        while True:
            # 1. Get Frame
            frames = pipeline.wait_for_frames()
            ir_frame = frames.get_infrared_frame(1)
            if not ir_frame: continue
            
            # Convert to numpy
            frame_data = np.asanyarray(ir_frame.get_data())
            
            # 2. Run Analysis
            metrics = analyze_and_get_clahe_params(frame_data)
            
            # 3. Apply CLAHE
            if metrics["do_clahe"]:
                clahe = cv2.createCLAHE(clipLimit=metrics["clip"], tileGridSize=(metrics["grid"], metrics["grid"]))
                processed_frame = clahe.apply(frame_data)
            else:
                # If CLAHE is off, just copy original but maybe add a text saying "Pass-through"
                processed_frame = frame_data.copy()
                cv2.putText(processed_frame, "PASSTHROUGH", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            # 4. Create Dashboard Layout
            # We will create a canvas of size (Width * 2, Height + Info_Box_Height)
            H, W = frame_data.shape
            info_H = 200
            dashboard = np.zeros((H + info_H, W * 2), dtype=np.uint8)
            
            # Convert to Color for the Dashboard (so we can use colored text)
            dashboard_color = cv2.cvtColor(dashboard, cv2.COLOR_GRAY2BGR)
            
            # Place Images (Convert grayscale frames to BGR for consistency)
            img_orig_bgr = cv2.cvtColor(frame_data, cv2.COLOR_GRAY2BGR)
            img_proc_bgr = cv2.cvtColor(processed_frame, cv2.COLOR_GRAY2BGR)
            
            # Add labels to images
            cv2.putText(img_orig_bgr, "RAW INPUT (Global Shutter)", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            cv2.putText(img_proc_bgr, "ADAPTIVE CLAHE", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            
            dashboard_color[0:H, 0:W] = img_orig_bgr
            dashboard_color[0:H, W:W*2] = img_proc_bgr
            
            # 5. Draw Data Boxes
            
            # BOX 1: Scene Characteristics (Luminescence)
            box1_data = [
                ("Mean Lum (0-1)", f"{metrics['L_mean']:.3f}"),
                ("P10 (Shadows)",  f"{metrics['p10']:.3f}"),
                ("P99 (Highlights)", f"{metrics['p90']:.3f}"),
                ("Dark Tiles %",   f"{metrics['dark_frac']*100:.1f}%"),
                ("Bright Tiles %", f"{metrics['bright_frac']*100:.1f}%")
            ]
            draw_info_box(dashboard_color, "SCENE METRICS", box1_data, 
                          x=50, y=H + 10, width=500, height=180, color_header=(255, 191, 0))
            
            # BOX 2: CLAHE Response (Parameters)
            box2_data = [
                ("Algorithm State", metrics["status"]),
                ("Active?",        "YES" if metrics["do_clahe"] else "NO"),
                ("Clip Limit",     f"{metrics['clip']:.2f}"),
                ("Kernel Size",    f"{metrics['grid']} x {metrics['grid']}"),
                ("Note",           "High Clip = More Detail")
            ]
            draw_info_box(dashboard_color, "ADAPTIVE PARAMETERS", box2_data, 
                          x=W + 50, y=H + 10, width=500, height=180, color_header=(0, 255, 0))

            # 6. Display
            cv2.imshow("DIP Pipeline Overview", dashboard_color)
            
            key = cv2.waitKey(1)
            if key & 0xFF == ord('q'):
                break
                
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()