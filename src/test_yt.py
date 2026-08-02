import cv2
import yt_dlp
from ultralytics import RTDETR

def get_youtube_stream_url(youtube_url):
    """Extracts direct mp4 stream link from YouTube URL."""
    ydl_opts = {
        'format': 'best[ext=mp4]/best',
        'quiet': True
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(youtube_url, download=False)
        return info['url']

def run_youtube_inference(youtube_url):
    print("===========================================")
    print("    Initializing YouTube PPE Inference     ")
    print("===========================================")

    model_path = 'C:/Users/hsmsa/runs/detect/ppe_training_results/rtdetr_frontview_800px/weights/best.pt'
    model = RTDETR(model_path)

    COLORS = {
        0: (0, 0, 255),    # Bare_Head -> RED
        1: (0, 165, 255),  # Hat_Not_OK -> ORANGE
        2: (0, 0, 255),    # NO_Safety_vest -> RED
        3: (0, 255, 0),    # Safety_vest_OK -> GREEN
        4: (0, 255, 0)     # helmet_OK -> GREEN
    }

    print("[INFO] Extracting stream URL from YouTube...")
    try:
        stream_url = get_youtube_stream_url(youtube_url)
    except Exception as e:
        print(f"[ERROR] Failed to extract YouTube stream: {e}")
        return

    cap = cv2.VideoCapture(stream_url)
    print("[INFO] Video feed active. Press 'Q' to exit.")

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("[INFO] End of video stream or connection lost.")
            break

        # Run inference (conf=0.35 helps capture lower-confidence hat classes)
        results = model(frame, stream=True, conf=0.35)

        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])

                color = COLORS.get(cls_id, (255, 255, 255))
                label = f"{model.names[cls_id]} {conf:.2f}"

                # Draw bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
                cv2.rectangle(frame, (x1, y1 - 25), (x1 + len(label)*11, y1), color, -1)
                cv2.putText(frame, label, (x1 + 5, y1 - 8), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

        # Show frame
        cv2.imshow("Smart PPE System - YouTube Feed", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("\n[INFO] Exiting...")
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    # Paste any construction site / factory YouTube video link here
    sample_youtube_link = "https://www.youtube.com/watch?v=9sjAuUdldyo" # Replace with your video
    run_youtube_inference(sample_youtube_link)