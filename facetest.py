import numpy as np
import cv2

xml = 'C:\\aa\\haarcascade_frontalface_default.xml' # 얼굴 검출
face_cascade = cv2.CascadeClassifier(xml) # Cascade 분류기를 생성
# Haar Cascade : 머신러닝기반 오브젝트 검출 알고리즘
# CascadeClassifier : 객체 인식을 위한 Haar 기능 기반 OpenCV API.
path = "C:\\aa\\test1.jpg"
cap = cv2.VideoCapture("C:\\aa\\test1.jpg") # 0이면 노트북 웹캠을 카메라로 사용, 파일명으로 입력해서 사진이나 동영상 불러오기 가능
cap.set(3,600) # 너비
cap.set(4,440) # 높이

# 3번에 사용할 이미지 불러오기
img = cv2.imread('C:\\aa\\test1.jpg', cv2.IMREAD_UNCHANGED) # 원본 사용
if img is None:
    print('이미지 파일을 로드할 수 없습니다.')

while(True):
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1) # 1 : 좌우 대칭, 2. 상하 대칭
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # 얼굴 검출을 위해 회색조로 영상 로드
    faces = face_cascade.detectMultiScale(gray,1.05,5) # Cascade 분류기를 사용하여 얼굴 검출하기
    #detectMultiScale : #입력 이미지에서 다양한 크기의 객체를 감지하는 함수. 감지된 개체는 사각형 목록으로 반환
    # image.shape(height, width, channel(색상 정보))

    print("Number of faces detected: " + str(len(faces))) # 발견한 얼굴 수

    if len(faces):
        for (x,y,w,h) in faces:
            face_img = frame[y:y+h, x:x+w] # 인식된 얼굴 영역의 이미지를 크롭해서 face_img에 저장
            
            # 1. 얼굴 모자이크 시키기
            #face_img = cv2.resize(face_img, dsize=(0, 0), fx=0.04, fy=0.04) # face_img의 높이와 너비를 0.04배로 축소
            #face_img = cv2.resize(face_img, (w, h), interpolation=cv2.INTER_AREA) # face_img를 원래 비율로. 이 과정에서 이미지가 깨짐
            #frame[y:y+h, x:x+w] = face_img # 인식된 얼굴 영역 face_img로 변경
        
            # 2. 얼굴 블러 처리하기
            #Gaussian blur : 중심에 있는 픽셀에 높은 가중치를 부여하는 블러링 방식
            #face_img = cv2.GaussianBlur(face_img,(99,99), 30)
            #frame[y:y+h, x:x+w] = face_img

            # 3. 얼굴에 이미지 씌우기
            #t = cv2.resize(img, dsize=(h, w), interpolation=cv2.INTER_LINEAR)
            #frame[y:y+h, x:x+w] = t

            # 4. 얼굴에 사각형 프레임 씌우기
            cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)

            # 5. 얼굴에 원 프레임 씌우기
            # 중심과 반지름은 정수형으로 표기. /(나누기)를 사용하면 실수형이 되어 오류 발생
            # cv2.circle(frame,(x+w//2,y+h//2),w//2,(0,255,255),3)
            
    cv2.imshow('result', frame)
        
    k = cv2.waitKey(30) & 0xff
    if k == 27: # Esc 키를 누르면 종료
        break

cap.release()
cv2.destroyAllWindows()