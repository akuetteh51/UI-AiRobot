import cv2
from flask import Flask, render_template, Response
app = Flask(__name__)
img=cv2.imread('nk.jpg')
img = cv2.resize(img, (0,0), fx=0.9, fy=0.5)
@app.route('/')
def index():
    """Video streaming home page."""
    return render_template('app.html')



def gen():
    classNames=[]
    classfile='coco.names'
    with open(classfile,'rt') as f:
        classNames=f.read().rstrip('\n').split('\n')
    #print(classNames)    
    configPath='ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
    weightsPath='frozen_inference_graph.pb'
    net=cv2.dnn_DetectionModel(weightsPath,configPath)
    net.setInputSize(320,320)
    net.setInputScale(1.0/127.5)
    net.setInputMean((127.5,127.5,127.7))
    net.setInputSwapRB(True)
    
        
    classIds,confs,bbox=net.detect(img,confThreshold=0.5)
    print(classIds,bbox)
    
    
    for classId,confidence,box in zip(classIds.flatten(),confs.flatten(),bbox):
        cv2.rectangle(img,box,color=(0,255,0),thickness=2)
        cv2.putText(img,classNames[classId-1].upper(),(box[0]+10,box[1]+30),
                    cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
        cv2.putText(img,str(round(confidence*100,2)),(box[0]+200,box[1]+30),
                    cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
#cv2.imshow("Output",img)
#cv2.waitKey(0)
    
    frame = cv2.imencode('.jpg', img)[1].tobytes()
    yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    """Video streaming route. Put this in the src attribute of an img tag."""
    return Response(gen(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


