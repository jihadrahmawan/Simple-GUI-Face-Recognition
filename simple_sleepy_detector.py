# -*- coding: utf-8 -*-

import dlib
import cv2
import numpy as np
import PySimpleGUI as sg
import time

camera_Width  = 640 # 480 # 640 # 1024 # 1280
camera_Heigth = 480 # 320 # 480 # 780  # 960
frameSize = (camera_Width, camera_Heigth)
video_capture = cv2.VideoCapture(1)

def landmarks_to_np(landmarks, dtype="int"):
    num = landmarks.num_parts
    coords = np.zeros((num, 2), dtype=dtype)
    for i in range(0, num):
        coords[i] = (landmarks.part(i).x, landmarks.part(i).y)
    return coords

sg.theme("Black")
predictor_path = r"./shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)


queue = np.zeros(30,dtype=int)
queue = queue.tolist()

# def webcam col
colwebcam1_layout = [[sg.Text("Camera View", size=(60, 1), justification="center", text_color='white')],
                        [sg.Image(filename="", key="cam1")]]
colwebcam1 = sg.Column(colwebcam1_layout, element_justification='center')

colwebcam2_layout = [[sg.Text("Face Recognition", size=(60, 1), justification="center",text_color='white')],
                        [sg.Image(filename="", key="cam1detect")]]
colwebcam2 = sg.Column(colwebcam2_layout, element_justification='center')
colslayout = [colwebcam1, colwebcam2]

rowfooter = [sg.Image(filename="./background.png", key="-IMAGEBOTTOM-")]
layout = [colslayout, rowfooter]

window    = sg.Window("Smart Coffe Break V 1.0 ", layout, 
                    no_titlebar=False, alpha_channel=1, grab_anywhere=False, 
                    return_keyboard_events=True, location=(100, 100), background_color='black')




while True:
    start_time = time.time()
    event, values = window.read(timeout=20)

    if event == sg.WIN_CLOSED:
        break

    # get camera frame
    ret, frameOrig = video_capture.read()
    frame = cv2.resize(frameOrig, frameSize)
    newframe = frameOrig.copy()
    # if (time.time() – start_time ) > 0:
    #     fpsInfo = "FPS: " + str(1.0 / (time.time() – start_time)) # FPS = 1 / time to process loop
    #     font = cv2.FONT_HERSHEY_DUPLEX
    #     cv2.putText(frame, fpsInfo, (10, 20), font, 0.4, (255, 255, 255), 1)

    # # update webcam1
    imgbytes = cv2.imencode(".png", frame)[1].tobytes()
    window["cam1"].update(data=imgbytes)
    
    # # transform frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    rects = detector(gray, 1)
    for i, rect in enumerate(rects):
 
        x = rect.left()
        y = rect.top()
        w = rect.right() - x
        h = rect.bottom() - y

        cv2.rectangle(newframe, (x,y), (x+w,y+h), (0,255,0), 1)
        #cv2.putText(img, "Face #{}".format(i + 1), (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
    
        landmarks = predictor(gray, rect)
        landmarks = landmarks_to_np(landmarks)

        for (x, y) in landmarks:
            cv2.circle(newframe, (x, y), 1, (0, 0, 255), -1)
 
        d1 =  np.linalg.norm(landmarks[37]-landmarks[41])
        d2 =  np.linalg.norm(landmarks[38]-landmarks[40])
        d3 =  np.linalg.norm(landmarks[43]-landmarks[47])
        d4 =  np.linalg.norm(landmarks[44]-landmarks[46])
        d_mean = (d1+d2+d3+d4)/4
        d5 =np.linalg.norm(landmarks[36]-landmarks[39])
        d6 =np.linalg.norm(landmarks[42]-landmarks[45])
        d_reference = (d5+d6)/2
        d_judge = d_mean/d_reference
        #print(d_judge)
        
        flag = int(d_judge<0.25)

        queue = queue[1:len(queue)] + [flag]
        
        
        if sum(queue) > len(queue)/2 :
            cv2.putText(newframe, "SLEEPY!", (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
        else:
            cv2.putText(newframe, "WORKING", (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)


   
    imgbytes = cv2.imencode(".png", newframe)[1].tobytes()
    window["cam1detect"].update(data=imgbytes)

video_capture.release()
cv2.destroyAllWindows()
window.close()

