import cv2
import numpy as np
import time

fps_start_time=time.time()
fps=0
total_frames=0

cap=cv2.VideoCapture("sample_video.mp4")
threshold=0.1

proto_file="pose_deploy_linevec_faster_4_stages.prototxt"
weights_file="pose_iter_160000.caffemodel"
npoints=15
POSE_PAIRS=[[0,1],[1,2],[2,3],[3,4],[1,5],[5,6],[6,7],[1,14],[14,8],[8,9],[9,10],[14,11],[11,12],[12,13]]

while cv2.waitKey(1) < 0:
    t=time.time()
    total_frames+=1
    hasFrame,frame=cap.read()
    frame_copy=np.copy(frame)
    #print(frame.shape)
    frame_width=frame.shape[1]
    frame_height=frame.shape[0]
    #print(frame_width,frame_height)
    video_writer=cv2.VideoWriter("Output_Video.avi",cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'),10,(frame.shape[1],frame.shape[0]))
    net=cv2.dnn.readNetFromCaffe(proto_file,weights_file)


    blob=cv2.dnn.blobFromImage(frame,1/255,(328,328),(0,0,0),swapRB=False,crop=False)
    #print(blob)

    net.setInput(blob)

    pred=net.forward()
    #print(pred.shape)
    #print("Time taken by network : {:.3f}".format(time.time()-t))

    h=pred.shape[2]
    w=pred.shape[3]
    #print(w,h)

    detection=[]

    for i in range(npoints):
        confidence_map=pred[0,i,:,:]
        
        # Find global maxima of the probMap.
        minVal,prob,minLoc,point=cv2.minMaxLoc(confidence_map)

        # Scale the point to fit on the original image
        x=(frame_width * point[0]) / w
        y=(frame_height * point[1]) / h
        #print(x,y)

        if prob>threshold:
            cv2.circle(frame_copy,(int(x),int(y)),8,(0, 255, 255),thickness=-1,lineType=cv2.FILLED)
            cv2.putText(frame_copy, "{}".format(i), (int(x),int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255),2 ,lineType=cv2.LINE_AA)

            detection.append((int(x),int(y)))
        else:
            detection.append(None)

    #print(detection)

    for pair in POSE_PAIRS:
        partA=pair[0]
        partB=pair[1]
        if detection[partA] and detection[partB]:
            cv2.line(frame,(detection[partA]),detection[partB],(0,255,255),2,cv2.LINE_AA)
            cv2.circle(frame, detection[partA], 8, (0,0,255),-1,cv2.FILLED)
            cv2.circle(frame, detection[partB], 8, (0, 0, 255), thickness=-1, lineType=cv2.FILLED)
    cv2.putText(frame, "Time taken :{:.3f} sec".format(time.time()-t),(50,50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,0,0),2)
    #print("Total time taken :{:.3f}".format(time.time()-t))

    fps_end_time=time.time()
    diff_time=fps_end_time-fps_start_time
    if diff_time==0:
        fps=0.0
    else:
        fps=(total_frames/diff_time)     #Calculationg the FPS
    fps_text="Fps : {:.2f}".format(fps)
    cv2.putText(frame,fps_text,(50,20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (150,0,255),2)
    cv2.imshow("Out1", frame)
    video_writer.write(frame)
    if cv2.waitKey(1) & 0XFF==ord('q'):
        break
video_writer.release()
cap.release()
cv2.destroyAllWindows()
