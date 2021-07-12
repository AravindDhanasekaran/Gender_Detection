from flask import Flask, render_template, Response
import cv2
import numpy as np


from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np

import os
import cvlib as cv

app=Flask(__name__)
camera = cv2.VideoCapture(0)

model=load_model('gender_detection.model')
classes=['man','woman']


def gen_frames():  
    while True:
        success, frame = camera.read()  # read the camera frame
        if not success:
            break
        else:
            
            face,confidence=cv.detect_face(frame)
            for idx,f in enumerate(face):
                startX,startY=f[0],f[1]
                endX,endY=f[2],f[3]
                cv2.rectangle(frame,(startX,startY),(endX,endY),(0,255,0),2)
                face_crop=np.copy(frame[startY:endY,startX:endX])
                face_crop= cv2.resize(face_crop,(96,96))
                face_crop=face_crop.astype("float")/255.0
                face_crop=img_to_array(face_crop)
                face_crop=np.expand_dims(face_crop,axis=0)
                conf=model.predict(face_crop)[0]
                idx=np.argmax(conf)
                label=classes[idx]
                label="{}:{:.2f}%".format(label,conf[idx]*100)
                y=startY-10 if startY-10>10 else startY+10
                cv2.putText(frame,label,(startX,y),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,0),2)
            

            ret, buffer = cv2.imencode('.jpg', frame)
            if not ret:
                continue
            frame = buffer.tobytes()
    yield (b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video')
def video():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__=='__main__':
    app.run(debug=True)












     