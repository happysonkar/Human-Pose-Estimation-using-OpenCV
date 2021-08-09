import cv2
import numpy as np
import time

proto_file="pose_deploy_linevec_faster_4_stages.prototxt"
weights_file="pose_iter_160000.caffemodel"
npoints=15
POSE_PAIRS=[[0,1],[1,2],[2,3],[3,4],[1,5],[5,6],[6,7],[1,14],[14,8],[8,9],[9,10],[14,11],[11,12],[12,13]]


img=cv2.imread("person.jpeg")
img_copy=np.copy(img)

img_width=img.shape[1]
img_height=img.shape[0]
print(img_width,img_height)

threshold=0.1
net=cv2.dnn.readNetFromCaffe(proto_file,weights_file)

t=time.time()

blob=cv2.dnn.blobFromImage(img,1.0/255,(328,328),(0,0,0),swapRB=False,crop=False)
#print(blob)

net.setInput(blob)

pred=net.forward()
print(pred.shape)
print("Time taken by network : {:.3f}".format(time.time()-t))

h=pred.shape[2]
w=pred.shape[3]
#print(w,h)

detection=[]

for i in range(npoints):
    confidence_map=pred[0,i,:,:]
    
    # Find global maxima of the probMap.
    minVal,prob,minLoc,point=cv2.minMaxLoc(confidence_map)

    # Scale the point to fit on the original image
    x=(img_width * point[0]) / w
    y=(img_height * point[1]) / h
    #print(x,y)

    if prob>threshold:
        cv2.circle(img_copy,(int(x),int(y)),8,(0, 255, 255),thickness=-1,lineType=cv2.FILLED)
        cv2.putText(img_copy, "{}".format(i), (int(x),int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255),2 ,lineType=cv2.LINE_AA)

        detection.append((int(x),int(y)))
    else:
        detection.append(None)

print(detection)

for pair in POSE_PAIRS:
    partA=pair[0]
    partB=pair[1]

    cv2.line(img,(detection[partA]),detection[partB],(0,255,255),2,cv2.LINE_AA)
    cv2.circle(img, detection[partA], 8, (0,0,255),-1)

print("Total time taken :{:.3f}".format(time.time()-t))

cv2.imshow("Out", img_copy)
cv2.imshow("Out1", img)
cv2.waitKey(0)