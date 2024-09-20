import threading
from modules.processors.face_analyser import get_many_faces, draw_on, verify
from modules.processors.face_aniti_spoof import verify_face_real
import cv2

# cap = cv2.VideoCapture('rtsp://admin:Hicas%402022@10.0.10.119')
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

frames = 0
faces_verify = []

def check_frame(frame):
    global faces_verify
    try:
        faces = get_many_faces(frame)
        faces_verify_temp = []
        for face in faces:
            label, value = verify_face_real(frame, face)
            if label == 1 and value >= 0.8:
                faces_verify_temp.append(face)
        faces_verify = faces_verify_temp
        faces_verify = verify(faces_verify, "database")
    except ValueError as error:
        print("Error verify:", error)

if not cap.isOpened():
    print("Không thể mở camera")
    exit()

while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)

    if not ret:
        print("Không thể nhận khung hình (Có thể là do kết nối bị lỗi). Đang thoát...")
        break
    
    if frames % 30 == 0:
        try: 
            threading.Thread(target=check_frame, args=(frame.copy(),)).start()
        except ValueError:
            print("Error")


    rimg = draw_on(frame, faces_verify)
    cv2.imshow('Camera', rimg)

    if cv2.waitKey(1) == ord('q'):
        break

    frames += 1

cap.release()
cv2.destroyAllWindows()