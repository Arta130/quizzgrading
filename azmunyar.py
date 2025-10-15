
# Place Key Sheet in the cyan box.

# Press 'K' to capture the key.

# (Optional, but recommended) Press 'C' to define the ROI (bubble area) once.

# Press 'P' to switch to Grading Ready mode.

# Place Student Sheet in the cyan box.

# Press 'R' to see the final grade.
import cv2
import numpy as np
import json
import os
import time

# --- Configuration Constants ---
# Target dimensions for Portrait A4 (800 x 1100)
TARGET_W = 800
TARGET_H = 1100
TOLERANCE_FACTOR = 0.03  # 3% tolerance for bubble matching
MIN_KEYPOINT_SIZE = 12   # Minimum size (in pixels) for ORB keypoints used for alignment

# Target points for the warped image (800x1100 for portrait)
TARGET_POINTS = np.array([
    [0, 0],
    [TARGET_W - 1, 0],
    [TARGET_W - 1, TARGET_H - 1],
    [0, TARGET_H - 1]
], dtype="float32")

# --- Global State ---
roi_box = None # Stores (x, y, w, h) of the crop area on the 800x1100 key image
full_warped_key_a4 = None # Stores the full 800x1100 key image (used for ROI selection)
key_img = None # Stores the final CROPPED key image (used for alignment/display)
key_positions = None # Stores detected key bubble positions
combined_result_to_save = None

# Grading modes
KEY_MODE = 0
GRADE_MODE = 1
current_mode = KEY_MODE

# --- File Paths ---
# New paths to ensure we can always reference the full, uncropped A4 image
KEY_CROPPED_PATH = "key_cropped.jpg"
KEY_FULL_A4_PATH = "key_full_a4.jpg"
KEY_POS_PATH = "key_positions.json"
KEY_ROI_PATH = "key_roi.json"

def calculate_fixed_warp_points(width, height):
    """
    Calculates the fixed coordinates for the paper guide box.
    Uses a large scale factor (0.9) to make the guide frame nearly 
    full-screen, while enforcing the 8:11 (Portrait) aspect ratio.
    """
    
    # Scale factor for the guide box size (Set to 90% to cover most of the screen)
    TARGET_SCALE_PERCENT = 0.9 
    
    # Aspect ratio of the target A4 paper (800/1100 = 0.727, Portrait)
    ASPECT_RATIO = TARGET_W / TARGET_H
    
    # 1. Start by sizing based on the screen height (since the aspect ratio is < 1)
    box_h = height * TARGET_SCALE_PERCENT
    box_w = box_h * ASPECT_RATIO # box_w will be smaller than box_h
    
    # 2. Fallback check: If the calculated width is too large, scale based on width instead
    if box_w > width * TARGET_SCALE_PERCENT: 
        box_w = width * TARGET_SCALE_PERCENT
        box_h = box_w / ASPECT_RATIO
    
    # 3. Center the box
    margin_w = (width - box_w) / 2
    margin_h = (height - box_h) / 2
    
    # Points: Top-Left, Top-Right, Bottom-Right, Bottom-Left
    return np.array([
        [margin_w, margin_h],
        [width - margin_w - 1, margin_h],
        [width - margin_w - 1, height - margin_h - 1],
        [margin_w, height - margin_h - 1]
    ], dtype="float32").astype(np.int32).reshape((-1, 1, 2))

def warp_content_to_target(frame, fixed_source_points_array):
    """
    Warps the content of the camera frame defined by fixed_source_points_array 
    to the target 800x1100 size.
    """
    # Reshape contour back to 4x2 float array for perspective transform
    source_points = fixed_source_points_array.reshape(4, 2).astype("float32")
    
    M = cv2.getPerspectiveTransform(source_points, TARGET_POINTS)
    warped = cv2.warpPerspective(frame, M, (TARGET_W, TARGET_H))
    return warped


def select_and_crop_roi(img):
    """Allows user to manually select a Region of Interest (ROI) using the mouse."""
    cv2.namedWindow("Select ROI - Press ENTER or SPACE when done", cv2.WINDOW_AUTOSIZE)
    roi = cv2.selectROI("Select ROI - Press ENTER or SPACE when done", img, False, False)
    cv2.destroyWindow("Select ROI - Press ENTER or SPACE when done")
    
    if roi[2] > 0 and roi[3] > 0:
        x, y, w, h = roi
        cropped_img = img[y:y+h, x:x+w]
        print(f"✅ ROI selected and applied: x={x}, y={y}, w={w}, h={h}")
        return cropped_img, roi
    else:
        print("⚠️ ROI selection cancelled or invalid. Using full image.")
        h, w, _ = img.shape
        return img, (0, 0, w, h)


def detect_key_bubbles(warped_img):
    """Detects filled bubbles in the warped and cropped key image."""
    gray = cv2.cvtColor(warped_img, cv2.COLOR_BGR2GRAY)
    mean_brightness = np.mean(gray)
    adaptive_threshold = max(60, min(140, mean_brightness * 0.8))

    def is_dark_dynamic(x, y, r):
        mask = np.zeros(gray.shape, dtype="uint8")
        cv2.circle(mask, (x, y), r, 255, -1)
        mean_intensity = cv2.mean(gray, mask=mask)[0]
        return mean_intensity < adaptive_threshold

    blurred = cv2.medianBlur(gray, 5)
    circles = cv2.HoughCircles(
        blurred, cv2.HOUGH_GRADIENT, dp=1.2, minDist=20,
        param1=50, param2=20, minRadius=8, maxRadius=25
    )
    if circles is None:
        return warped_img.copy(), {}

    circles = np.round(circles[0, :]).astype("int")
    
    key_positions = {}
    key_preview = warped_img.copy()

    for i, (x, y, r) in enumerate(circles):
        if is_dark_dynamic(x, y, r):
            cv2.rectangle(key_preview, (x - r, y - r), (x + r, y + r), (0, 255, 0), 2)
            key_positions[str(i)] = (float(x), float(y), float(r))
        else:
            cv2.circle(key_preview, (x, y), r, (150, 150, 150), 1)

    return key_preview, key_positions


def align_paper_to_key(student_img, key_img):
    """Aligns the student image (cropped) to the key image (cropped) using ORB and Homography."""
    gray_student = cv2.cvtColor(student_img, cv2.COLOR_BGR2GRAY)
    gray_key = cv2.cvtColor(key_img, cv2.COLOR_BGR2GRAY)

    orb = cv2.ORB_create(2000)
    kp1_all, des1_all = orb.detectAndCompute(gray_student, None)
    kp2_all, des2_all = orb.detectAndCompute(gray_key, None)

    # Filter keypoints by size
    def filter_keypoints_and_descriptors(kp_all, des_all, min_size):
        if des_all is None: return [], None
        filtered_kp = []
        filtered_des_indices = []
        for i, kp in enumerate(kp_all):
            if kp.size >= min_size:
                filtered_kp.append(kp)
                filtered_des_indices.append(i)
        
        if not filtered_kp: return [], None
        return filtered_kp, des_all[filtered_des_indices]

    kp1, des1 = filter_keypoints_and_descriptors(kp1_all, des1_all, MIN_KEYPOINT_SIZE)
    kp2, des2 = filter_keypoints_and_descriptors(kp2_all, des2_all, MIN_KEYPOINT_SIZE)

    if des1 is None or des2 is None or len(kp1) < 15 or len(kp2) < 15:
        print("⚠️ Failed to detect sufficient large ORB descriptors for robust alignment.")
        return student_img, np.eye(3)

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = sorted(bf.match(des1, des2), key=lambda x: x.distance)[:100]

    if len(matches) < 15:
        print(f"⚠️ Only {len(matches)} keypoint matches found. Alignment may be poor.")
        return student_img, np.eye(3)

    src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    
    H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    
    if H is None:
        print("⚠️ Homography calculation failed.")
        return student_img, np.eye(3)

    aligned = cv2.warpPerspective(student_img, H, (key_img.shape[1], key_img.shape[0]))

    return aligned, H


def calculate_grade(aligned_student, key_positions):
    """Scores the aligned student sheet against the key positions."""
    gray = cv2.cvtColor(aligned_student, cv2.COLOR_BGR2GRAY)
    mean_brightness = np.mean(gray)
    adaptive_threshold = max(60, min(140, mean_brightness * 0.8))

    def is_dark_dynamic(x, y, r):
        mask = np.zeros(gray.shape, dtype="uint8")
        cv2.circle(mask, (x, y), r, 255, -1)
        mean_intensity = cv2.mean(gray, mask=mask)[0]
        return mean_intensity < adaptive_threshold

    blurred = cv2.medianBlur(gray, 5)
    circles = cv2.HoughCircles(
        blurred, cv2.HOUGH_GRADIENT, dp=1.2, minDist=20,
        param1=50, param2=20, minRadius=8, maxRadius=25
    )
    if circles is None:
        return aligned_student.copy(), 0, 0

    circles = np.round(circles[0, :]).astype("int")

    student_filled = [(x, y, r) for (x, y, r) in circles if is_dark_dynamic(x, y, r)]
    
    graded = aligned_student.copy()
    paper_width = gray.shape[1]
    TOLERANCE = int(TOLERANCE_FACTOR * paper_width) 

    correct = 0
    matched_key_indices = set()

    parsed_key_positions = {int(k): (int(v[0]), int(v[1]), int(v[2])) for k, v in key_positions.items()}
    
    for (sx, sy, sr) in student_filled:
        best_match = None
        min_dist = 1e9
        
        for idx, (kx, ky, kr) in parsed_key_positions.items():
            dx, dy = abs(kx - sx), abs(ky - sy)
            dist = np.sqrt(dx ** 2 + dy ** 2)
            
            if dist < TOLERANCE and dist < min_dist:
                best_match = idx
                min_dist = dist

        if best_match is not None and best_match not in matched_key_indices:
            cv2.rectangle(graded, (sx - sr, sy - sr), (sx + sr, sy + sr), (0, 255, 0), 3)
            matched_key_indices.add(best_match)
            correct += 1
        else:
            cv2.rectangle(graded, (sx - sr, sy - sr), (sx + sr, sy + sr), (0, 0, 255), 3)
    
    for idx, (kx, ky, kr) in parsed_key_positions.items():
        if idx not in matched_key_indices:
            # Orange for missing/unanswered key bubbles
            cv2.rectangle(graded, (kx - kr, ky - kr), (kx + kr, ky + kr), (0, 165, 255), 2)
            
    return graded, correct, len(student_filled)


# --- Orchestration Functions ---

def capture_key_from_camera(frame, fixed_source_points):
    """Handles the key capture process using the fixed warp points."""
    global full_warped_key_a4
    print("--- Capturing Key ---")
    
    warped_key_a4 = warp_content_to_target(frame, fixed_source_points)

    if warped_key_a4 is None:
        print("⚠️ Fixed warp failed.")
        return None

    # Save the full warped image (before cropping) for ROI selection
    cv2.imwrite(KEY_FULL_A4_PATH, warped_key_a4)
    full_warped_key_a4 = warped_key_a4

    # Print message updated to reflect 800x1100 size and Portrait orientation
    print("✅ Key paper captured (800x1100 - Portrait). Press 'C' to define the bubble area (ROI).")
    return warped_key_a4


def process_key_with_roi(full_warped_key):
    """
    Applies the ROI crop (from global roi_box) to the key, re-runs bubble detection, 
    and saves the final key data.
    """
    global key_img, key_positions, roi_box
    
    # Check if a valid ROI box is set globally
    if roi_box and roi_box[2] > 0 and roi_box[3] > 0:
        x, y, w, h = roi_box
        # Use full_warped_key which is the 800x1100 image
        cropped_key = full_warped_key[y:min(y+h, full_warped_key.shape[0]), x:min(x+w, full_warped_key.shape[1])]
    else:
        # If no ROI is set, use the full image and update roi_box to match the full image
        cropped_key = full_warped_key
        h, w, _ = full_warped_key.shape
        roi_box = (0, 0, w, h)
        # Save this 'full image' ROI as the default
        with open(KEY_ROI_PATH, "w") as f:
            json.dump(list(roi_box), f, indent=4)


    # 1. Bubble Detection on Cropped Image
    key_preview, new_key_positions = detect_key_bubbles(cropped_key)

    # 2. Save new key data
    cv2.imwrite(KEY_CROPPED_PATH, cropped_key) # Save the cropped version for fast alignment
    with open(KEY_POS_PATH, "w") as f:
        json.dump(new_key_positions, f, indent=4)
    cv2.imwrite("key_filled_preview.jpg", key_preview)

    key_img = cropped_key # Set the global key_img to the cropped/final version
    key_positions = new_key_positions
    print(f"✅ Key processed with {len(key_positions)} filled bubbles.")
    return key_preview


def grade_student_sheet(student_img, key_img, key_positions, roi_box, fixed_source_points):
    """Combines warping, alignment, and grading, applying the ROI crop."""
    
    # 1. Warp student image using the same fixed warp points (to 800x1100)
    student_warp = warp_content_to_target(student_img, fixed_source_points)
            
    if student_warp is None:
        print("⚠️ Fixed warp on student sheet failed.")
        return None, 0, 0
        
    # 2. Apply ROI Crop (using the global roi_box)
    if roi_box:
        x, y, w, h = roi_box
        # Crop the student sheet in the same way the key was cropped
        student_cropped = student_warp[y:min(y+h, student_warp.shape[0]), x:min(x+w, student_warp.shape[1])]
    else:
        student_cropped = student_warp

    if student_cropped.size == 0 or student_cropped.shape[0] < 50 or student_cropped.shape[1] < 50:
        print("⚠️ ROI crop resulted in an invalid/too small image. Alignment skipped.")
        return None, 0, 0
    
    # 3. Alignment (student_cropped to key_img)
    aligned_student, _ = align_paper_to_key(student_cropped, key_img)
    
    if aligned_student is None:
        return None, 0, 0
    
    # 4. Grading on Aligned Sheet
    graded_overlay, correct, total_filled = calculate_grade(aligned_student, key_positions)

    return graded_overlay, correct, total_filled


# --- Main Camera Loop and Interface ---

def run_grader():
    global roi_box, full_warped_key_a4, key_img, key_positions, combined_result_to_save, current_mode
    
    cap = cv2.VideoCapture(0)
    
    # Attempt to load saved data
    
    # 1. Load FULL A4 image (needed for 'C' to work correctly)
    if os.path.exists(KEY_FULL_A4_PATH):
        try:
            full_warped_key_a4 = cv2.imread(KEY_FULL_A4_PATH)
            print("💾 Loaded full A4 reference image for ROI editing.")
        except Exception as e:
            print(f"Error loading full A4 key file: {e}. Skipping.")
    
    # 2. Load ROI if it exists
    if os.path.exists(KEY_ROI_PATH):
        try:
            with open(KEY_ROI_PATH, "r") as f:
                roi_box = tuple(json.load(f))
            print(f"📐 Loaded saved ROI box: {roi_box}")
        except Exception as e:
            print(f"Error loading ROI file: {e}. Resetting ROI.")
            roi_box = None

    # 3. Load CROPPED key image (used for fast alignment)
    if os.path.exists(KEY_CROPPED_PATH) and os.path.exists(KEY_POS_PATH):
        try:
            key_img = cv2.imread(KEY_CROPPED_PATH)
            with open(KEY_POS_PATH, "r") as f:
                key_positions = json.load(f)
            print(f"💾 Loaded existing key with {len(key_positions)} answers.")
        except Exception as e:
            print(f"Error loading cropped key files: {e}")
            key_img = None
            key_positions = None


    print("\n--- Auto Grader Initialized ---")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        h, w, _ = frame.shape
        
        # 1. Calculate the fixed guide box points based on the current frame size
        fixed_contour = calculate_fixed_warp_points(w, h)
        
        # 2. Draw the fixed guide box (Cyan) on the live feed
        display = frame.copy()
        cv2.drawContours(display, [fixed_contour], -1, (255, 255, 0), 5) 

        # 3. Display Status
        cv2.putText(display, "BUBBLE SHEET GRADER", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2, cv2.LINE_AA)
        
        if key_img is None:
            status_text = "STATUS: PLACE KEY (Portrait) | Press 'K' to capture" 
            status_color = (0, 165, 255) # Orange
        elif current_mode == KEY_MODE:
            # Check key_img shape to confirm if a key has been loaded or processed
            key_width, key_height = key_img.shape[1], key_img.shape[0]
            roi_status = f"ROI: {roi_box[2]}x{roi_box[3]}" if roi_box and roi_box[2] > 0 else "ROI: None (Press 'C')"
            status_text = f"KEY MODE: Active ({len(key_positions)} ans) | {key_width}x{key_height} | {roi_status}"
            status_color = (0, 255, 0) # Green
        else: # GRADE_MODE
            status_text = "GRADING READY: Place Student Sheet (Portrait) | Press 'R' to grade"
            status_color = (255, 0, 255) # Magenta
        
        cv2.putText(display, status_text, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2, cv2.LINE_AA)

        cv2.imshow("Live Camera Feed (Original)", display)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('k'):
            # Key Capture Mode
            current_mode = KEY_MODE
            
            # 1. Capture and Warp to 800x1100 (Portrait), saving to full_warped_key_a4
            new_full_warped = capture_key_from_camera(frame, fixed_contour)
            
            if new_full_warped is not None:
                if roi_box and roi_box[2] > 0:
                    # If ROI is defined, automatically apply crop and process bubbles
                    process_key_with_roi(new_full_warped)
                else:
                    # If no ROI, set key_img to the full A4 image for immediate use
                    key_img = new_full_warped.copy() 
                    print("Hint: Press 'C' to set the Region of Interest now.")
            
        elif key == ord('c'):
            # Define Crop Area (ROI)
            
            # CRITICAL: Always use the full_warped_key_a4 for ROI selection
            img_for_roi = full_warped_key_a4 
            
            if img_for_roi is None:
                print("⚠️ Please capture the Key Sheet first (press 'K').")
            else:
                print("--- ROI Selection Started ---")
                
                # new_key_img here is TEMPORARILY the cropped preview of the selection
                new_key_img_preview, new_roi_box = select_and_crop_roi(img_for_roi.copy()) 
                
                if new_roi_box:
                    roi_box = new_roi_box
                    
                    # Save the new ROI box coordinates
                    with open(KEY_ROI_PATH, "w") as f:
                        json.dump(list(roi_box), f, indent=4)
                    
                    # Process the key image with the new ROI (uses the full A4 image in memory)
                    process_key_with_roi(img_for_roi)

        elif key == ord('p'):
            # Toggle to Grading Mode (as requested by user)
            if key_img is None:
                 print("⚠️ Cannot switch to Grading Mode. Capture the key first (press 'K').")
            else:
                 current_mode = GRADE_MODE
                 print("✅ Switched to Grading Ready Mode. Place student sheet in the guide box and press 'R'.")

        elif key == ord('r'):
            # Grading Mode
            if key_img is None or key_positions is None:
                print("⚠️ Capture and process a key first (press 'K' then 'C' if needed).")
                combined_result_to_save = None
            elif current_mode != GRADE_MODE:
                 print("⚠️ Please press 'P' to switch to Grading Ready Mode before pressing 'R'.")
            else:
                # Grade using the fixed warp points
                graded, correct, total_filled = grade_student_sheet(
                    frame, key_img, key_positions, roi_box, fixed_contour
                )
                
                if graded is not None:
                    print(f"✅ Grading Result: {correct}/{len(key_positions)} correct. Student filled {total_filled} bubbles.")
                    
                    # --- SCORE TEXT ---
                    # Positioned to fit the portrait output (800x1100)
                    score_text = f"Score: {correct}/{len(key_positions)}"
                    score_y = 70 
                    score_font_scale = 1.0
                    score_font_thickness = 3
                    score_color = (0, 0, 0) # Black
                    cv2.putText(graded, score_text, (10, score_y), cv2.FONT_HERSHEY_SIMPLEX, score_font_scale, score_color, score_font_thickness, cv2.LINE_AA)

                    # Stack Key with Graded Student
                    combined_img = np.hstack([key_img, graded])
                    
                    # --- ADD TITLES (Black, 2x smaller font) ---
                    key_width = key_img.shape[1]
                    title_y = 30
                    font_scale = 0.75 
                    font_thickness = 2
                    title_color = (0, 0, 0) # Black text
                    font_style = cv2.FONT_HERSHEY_SIMPLEX 

                    cv2.putText(combined_img, "KEY", 
                                (key_width // 2 - 40, title_y), 
                                font_style, font_scale, title_color, font_thickness, cv2.LINE_AA)
                    
                    cv2.putText(combined_img, "STUDENT", 
                                (key_width + key_width // 2 - 70, title_y), 
                                font_style, font_scale, title_color, font_thickness, cv2.LINE_AA)
                    # --- END TITLES ---
                    
                    combined_result_to_save = combined_img.copy()
                    cv2.imshow("FINAL GRADE PREVIEW (Key | Graded Student)", combined_result_to_save)
                
        elif key == ord('s'):
            if combined_result_to_save is not None:
                filename = f"graded_result_{int(time.time())}.jpg"
                cv2.imwrite(filename, combined_result_to_save)
                print(f"💾 Saved combined result (Key and Graded Student) as {filename}")
            else:
                print("⚠️ Please grade a sheet first (press 'R') before saving.")
        
        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_grader()
