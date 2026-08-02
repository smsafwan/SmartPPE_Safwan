import cv2
import yt_dlp
from ultralytics import RTDETR

# ====================================================================
# CONFIGURATION & THRESHOLDS
# ====================================================================
MODEL_PATH = 'C:/Users/hsmsa/runs/detect/ppe_training_results/rtdetr_frontview_800px-2/weights/best.pt'

# BGR Colors: 0:Bare_Head, 1:Hat_Not_OK, 2:NO_Safety_vest, 3:Safety_vest_OK, 4:helmet_OK
COLORS = {
    0: (0, 0, 255),    # Bare_Head -> RED
    1: (0, 165, 255),  # Hat_Not_OK -> ORANGE
    2: (0, 0, 255),    # NO_Safety_vest -> RED
    3: (0, 255, 0),    # Safety_vest_OK -> GREEN
    4: (0, 255, 0)     # helmet_OK -> GREEN
}

# Calibrated Thresholds: Strict cutoffs to kill background noise
CLASS_THRESHOLDS = {
    0: 0.55,  # Bare_Head
    1: 0.50,  # Hat_Not_OK (Raised from 0.35 to stop truck hallucinations)
    2: 0.55,  # NO_Safety_vest
    3: 0.55,  # Safety_vest_OK
    4: 0.55   # helmet_OK
}

def get_youtube_url(url):
    """Extracts direct video stream URL from a YouTube link."""
    ydl_opts = {'format': 'best[ext=mp4]/best', 'quiet': True}
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=False)
        return info['url']

def process_frame(frame, model):
    """Runs RT-DETR inference, applies IoU suppression & custom thresholds."""
    
    # agnostic_nms=True forces it to delete overlapping boxes even if they are different classes
    results = model(frame, stream=True, conf=0.45, iou=0.45, agnostic_nms=True)

    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])

            # Apply per-class confidence filter
            if conf < CLASS_THRESHOLDS.get(cls_id, 0.50):
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            color = COLORS.get(cls_id, (255, 255, 255))
            label = f"{model.names[cls_id]} {conf:.2f}"

            # Render box and text overlay
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
            cv2.rectangle(frame, (x1, y1 - 25), (x1 + len(label)*11, y1), color, -1)
            cv2.putText(frame, label, (x1 + 5, y1 - 8), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
    return frame

def run_detection(source):
    print("===========================================")
    print("   Smart PPE Detection - System Active     ")
    print("===========================================")

    model = RTDETR(MODEL_PATH)

    # 1. Handle Static Image File
    if isinstance(source, str) and source.endswith(('.jpg', '.jpeg', '.png')):
        frame = cv2.imread(source)
        if frame is None:
            print(f"[ERROR] Image not found at {source}")
            return
        annotated = process_frame(frame, model)
        cv2.imshow("Smart PPE Detection - Image Result", annotated)
        cv2.imwrite("ppe_output_result.jpg", annotated)
        print("[INFO] Saved output as 'ppe_output_result.jpg'. Press any key to close.")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return

    # 2. Handle YouTube Stream
    if isinstance(source, str) and ("youtube.com" in source or "youtu.be" in source):
        print("[INFO] Extracting YouTube stream URL...")
        source = get_youtube_url(source)

    # 3. Handle Video File or Webcam
    cap = cv2.VideoCapture(source)
    print("[INFO] Live feed active. Press 'Q' to quit.")

    # --- SPEED CONTROL ---
    # Set to 1 for normal speed. 
    # Set to 2 for 2x speed (skips every other frame).
    # Set to 3 for 3x speed, etc.
    speed_multiplier = 2 
    frame_counter = 0

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        frame_counter += 1
        
        # Skip frames to increase playback speed
        if frame_counter % speed_multiplier != 0:
            continue

        annotated = process_frame(frame, model)
        cv2.imshow("Smart PPE Detection System", annotated)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    # EXAMPLES OF SOURCES YOU CAN PASS TO run_detection():
    # -----------------------------------------------------
    # 0                                         -> Live Webcam
    # "test_image.jpg"                          -> Static Image
    # "datasets/simulation_video.mp4"           -> Local Video
    # "https://www.youtube.com/watch?v=..."     -> YouTube Stream

    run_detection("https://www.youtube.com/watch?v=7M8_pZborlw")  # Change 0 to your image path or video URL