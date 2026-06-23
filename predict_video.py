import cv2
from PIL import Image
from predict import predict_image
import sys

if len(sys.argv) < 2:
    print("사용법: python predict_video.py <영상.mp4>  또는  0(웹캠)")
    sys.exit()

src = sys.argv[1]
cap = cv2.VideoCapture(0 if src == "0" else src)

if not cap.isOpened():
    print(f"열 수 없음: {src}")
    sys.exit()

frame_count = 0
label, conf = "ground", 0.0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    if frame_count % 5 == 0:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)
        label, conf = predict_image(pil_img)

    color = (0, 0, 255) if label == "stair" else (0, 200, 0)
    text = f"{label.upper()} ({conf*100:.0f}%)"
    cv2.putText(frame, text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)

    if label == "stair":
        cv2.putText(frame, "WARNING: STAIRS", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

    cv2.imshow("SafeStep", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    frame_count += 1

cap.release()
cv2.destroyAllWindows()
print("종료")
