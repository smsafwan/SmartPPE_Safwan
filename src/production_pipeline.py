import cv2
import numpy as np
import threading
import time
import os
import winsound
import requests
from ultralytics import RTDETR

# ==========================================
# CONFIGURATION & HYPERPARAMETERS
# ==========================================
# Pointing directly to PyTorch weights
MODEL_PATH = "weights/best.pt"
INPUT_WIDTH = 640
INPUT_HEIGHT = 640

CLASS_NAMES = ["Bare_Head", "NO_Safety_vest", "Non_Compliant_Hat", "Safety_vest", "helmet"]
# Colors in BGR: Red for violations, Green for compliance
CLASS_COLORS = [(0, 0, 255), (0, 0, 255), (0, 0, 255), (0, 255, 0), (0, 255, 0)]
CLASS_THRESHOLDS = [0.50, 0.50, 0.50, 0.55, 0.55]

# Telegram & Audio Configuration
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "CHAT_ID")
CUSTOM_ALARM_PATH = "assets/alarm.wav"

# ==========================================
# ASYNCHRONOUS ALERT & MEDIA MANAGER
# ==========================================
class AlertManager:
    def __init__(self):
        self.last_alert_time = 0.0
        self.cooldown_seconds = 5.0
        self.first_alert = True

    def trigger_violation_actions(self, frame, violation_type):
        current_time = time.time()
        if self.first_alert or (current_time - self.last_alert_time >= self.cooldown_seconds):
            self.last_alert_time = current_time
            self.first_alert = False
            
            # 1. Play local floor audio alarm asynchronously
            threading.Thread(target=self._play_custom_alarm, daemon=True).start()
            
            # 2. Compress frame and upload snapshot to Telegram API
            threading.Thread(target=self._send_telegram_photo, args=(frame.copy(), violation_type), daemon=True).start()

    def _play_custom_alarm(self):
        try:
            if os.path.exists(CUSTOM_ALARM_PATH):
                winsound.PlaySound(CUSTOM_ALARM_PATH, winsound.SND_FILENAME | winsound.SND_ASYNC)
            else:
                # Fallback system beep if custom file is missing
                winsound.Beep(2500, 400)
        except Exception as e:
            print(f"[AUDIO ERROR]: {e}")

    def _send_telegram_photo(self, frame, violation_type):
        try:
            success, encoded_image = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if not success:
                return

            photo_bytes = encoded_image.tobytes()
            caption = f"⚠️ SMART-PPE ALARM: Violation Detected -> {violation_type}"

            if TELEGRAM_BOT_TOKEN == "BOT_TOKEN":
                print(f"[SIMULATED TELEGRAM DISPATCH]: {caption}")
                return

            url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendPhoto"
            files = {'photo': ('violation.jpg', photo_bytes, 'image/jpeg')}
            data = {'chat_id': TELEGRAM_CHAT_ID, 'caption': caption}

            requests.post(url, data=data, files=files, timeout=5)
        except Exception as e:
            print(f"[TELEGRAM EXCEPTION]: {e}")

# ==========================================
# THREADED INFERENCE ENGINE (NATIVE PYTORCH)
# ==========================================
class SmartPPEInferenceEngine:
    def __init__(self, model_path):
        print("[INFO] Initializing PyTorch Inference Engine with CUDA...")
        
        # Load the native PyTorch model
        self.model = RTDETR(model_path)
        
        # Quick dummy warmup to allocate GPU memory before the webcam starts
        print("[INFO] Warming up GPU...")
        dummy_frame = np.zeros((INPUT_HEIGHT, INPUT_WIDTH, 3), dtype=np.uint8)
        self.model.predict(dummy_frame, imgsz=INPUT_WIDTH, verbose=False)
        
        self.lock = threading.Lock()
        self.latest_frame = None
        self.latest_detections = []
        self.running = True
        
        # Inference FPS Tracking
        self.inference_fps = 0.0
        self.infer_counter = 0
        self.infer_timer = time.time()
        
        self.alert_manager = AlertManager()
        
        self.thread = threading.Thread(target=self._inference_loop, daemon=True)
        self.thread.start()

    def submit_frame(self, frame):
        with self.lock:
            self.latest_frame = frame.copy()

    def _inference_loop(self):
        while self.running:
            frame_to_process = None
            with self.lock:
                if self.latest_frame is not None:
                    frame_to_process = self.latest_frame
                    self.latest_frame = None
            
            if frame_to_process is None:
                time.sleep(0.005)
                continue
                
            # Native PyTorch Forward Pass
            results = self.model.predict(
                frame_to_process, 
                imgsz=INPUT_WIDTH,  
                verbose=False,
                conf=0.55 # Pre-filter low confidence boxes internally
            )
            
            # Update Inference FPS
            self.infer_counter += 1
            elapsed = time.time() - self.infer_timer
            if elapsed >= 1.0:
                with self.lock:
                    self.inference_fps = self.infer_counter / elapsed
                self.infer_counter = 0
                self.infer_timer = time.time()
            
            detections = []
            frame_has_violation = False
            violation_category = "Missing PPE"

            # Parse Ultralytics Results Object
            boxes = results[0].boxes
            for box in boxes:
                confidence = float(box.conf[0])
                class_id = int(box.cls[0])
                
                if class_id < 0 or class_id >= len(CLASS_THRESHOLDS):
                    continue
                    
                if confidence >= CLASS_THRESHOLDS[class_id]:
                    # Extract bounding box coordinates (xyxy format)
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    
                    detections.append((x1, y1, x2, y2, class_id, confidence))
                    
                    if class_id in [0, 1, 2]:  # Violation classes
                        frame_has_violation = True
                        violation_category = CLASS_NAMES[class_id]

            if frame_has_violation:
                self.alert_manager.trigger_violation_actions(frame_to_process, violation_category)

            with self.lock:
                self.latest_detections = detections

    def get_results(self):
        with self.lock:
            return self.latest_detections, self.inference_fps

    def stop(self):
        self.running = False

# ==========================================
# MAIN EXECUTION ROUTINE
# ==========================================
def main():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    
    if not cap.isOpened():
        print("[ERROR] Could not open video capture stream.")
        return

    engine = SmartPPEInferenceEngine(MODEL_PATH)
    print("[INFO] Production monitoring pipeline active. Press 'ESC' to exit.")

    display_fps = 0.0
    frame_counter = 0
    fps_timer = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        engine.submit_frame(frame)
        detections, infer_fps = engine.get_results()
        
        # Calculate Camera UI Display FPS
        frame_counter += 1
        elapsed = time.time() - fps_timer
        if elapsed >= 1.0:
            display_fps = frame_counter / elapsed
            frame_counter = 0
            fps_timer = time.time()
        
        # Render Detections
        has_active_violation = False
        for (x1, y1, x2, y2, class_id, confidence) in detections:
            color = CLASS_COLORS[class_id]
            label = f"{CLASS_NAMES[class_id]} {confidence:.2f}"
            
            if class_id in [0, 1, 2]:
                has_active_violation = True
            
            # Draw Bounding Box & Label Tag
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(frame, (x1, max(y1 - 20, 0)), (x1 + w, max(y1, 20)), color, -1)
            cv2.putText(frame, label, (x1, max(y1 - 5, 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Overlay System HUD & Diagnostic Metrics
        status_text = "STATUS: VIOLATION DETECTED" if has_active_violation else "STATUS: COMPLIANT / NOMINAL"
        status_color = (0, 0, 255) if has_active_violation else (0, 255, 0)
        
        cv2.putText(frame, status_text, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, status_color, 2)
        cv2.putText(frame, f"Display FPS: {display_fps:.1f}", (20, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, f"Inference FPS: {infer_fps:.2f}", (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            
        cv2.imshow("SmartPPE Industrial Compliance Monitor", frame)
        
        if cv2.waitKey(1) == 27:
            break

    engine.stop()
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()