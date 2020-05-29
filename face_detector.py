#start
from Trait.face_blurring import face_blurring
import os.path
import numpy as np
import cv2
import json
from flask import Flask,request,Response
import uuid
#detect face in an image
def faceDetect(img):
    face_cascade = cv2.CascadeClassifier('face_detctor_cascaade.xml')#load the detector
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #load the image in gray scale mode
    faces = face_cascade.detectMultiScale(gray,1.05,6)
    #use it to find faces in the image,If faces are found, it returns the positions of detected faces as Rect(x,y,w,h).


    eyes_cascade = cv2.CascadeClassifier('model/eyes_detector_cascade.xml')
    eyes = eyes_cascade.detectMultiScale(gray,1.01,2)
    for (ex, ey, ew, eh) in eyes:
        #cv2.rectangle(img, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
        sub_face = img[ey:ey + eh, ex:ex + ew]
        sub_face = face_blurring(sub_face, factor=2.0)
        img[ey:ey + sub_face.shape[0], ex:ex + sub_face.shape[1]] = sub_face

    for(x,y,w,h) in faces :
        #cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0))

        sub_face = img[y:y + h, x:x + w]
        sub_face = face_blurring(sub_face,factor = 3.0)
        img[y:y + sub_face.shape[0], x:x + sub_face.shape[1]] = sub_face

        roi_gray = gray[y:y + h, x:x + w]
        roi_color = img[y:y + h, x:x + w]
        #for (ex, ey, ew, eh) in eyes:
            #cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0),1)


    #save file
    path_file = ('static/%s.jpg' %uuid.uuid4().hex)
    cv2.imwrite(path_file,img)
    return json.dumps(path_file) #return image file

#API
app = Flask(__name__)
#app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
#route http post to this method
@app.route('/api/take',methods=['post'])# Tells the flask server on which url path does it trigger
def take():
    img = cv2.imdecode(np.fromstring(request.files['image'].read(),np.uint8),cv2.IMREAD_UNCHANGED)
    #process image
    img_processed = faceDetect(img)
    #response
    return Response (response=img_processed, status=200,mimetype="application/json") #return json string
#start server
app.run(host="0.0.0.0",port=5000,debug= False,ssl_context='adhoc')
#ssl_context='adhoc'