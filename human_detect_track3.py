'''
Version3
tracking + reidentification.
Tracking is done using last k=15 frames.

Change from human_detect_track3: some changes in find()
Used matchTemplate() to check the result of tracking.
Sometimes, 2 people very close to each other are tracked as the same person, so this makes sure that this error does not occur.
'''
import numpy as np
import tensorflow as tf
import cv2
import time
import os
from run import Reid
#from run import main2
from importlib import import_module

class DetectorAPI:
    def __init__(self, path_to_ckpt):
        self.reid = Reid()
        self.path_to_ckpt = path_to_ckpt
        #self.module = import_module('run')
        self.detection_graph = tf.Graph()
        with self.detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(self.path_to_ckpt, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')

        self.default_graph = self.detection_graph.as_default()    
        self.sess = tf.Session(graph=self.detection_graph)

        # Definite input and output Tensors for detection_graph
        self.image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
        # Each box represents a part of the image where a particular object was detected.
        self.detection_boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
        # Each score represent how level of confidence for each of the objects.
        # Score is shown on the result image, together with the class label.
        self.detection_scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
        self.detection_classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
        self.num_detections = self.detection_graph.get_tensor_by_name('num_detections:0')

    def processFrame(self, image):
        # Expand dimensions since the trained_model expects images to have shape: [1, None, None, 3]
        image_np_expanded = np.expand_dims(image, axis=0)
        # Actual detection.
        start_time = time.time()
        (boxes, scores, classes, num) = self.sess.run(
            [self.detection_boxes, self.detection_scores, self.detection_classes, self.num_detections],
            feed_dict={self.image_tensor: image_np_expanded})
        end_time = time.time()

        im_height, im_width,_ = image.shape
        boxes_list = [None for i in range(boxes.shape[1])]
        for i in range(boxes.shape[1]):
            boxes_list[i] = (int(boxes[0,i,0] * im_height),
                        int(boxes[0,i,1]*im_width),
                        int(boxes[0,i,2] * im_height),
                        int(boxes[0,i,3]*im_width))

        return boxes_list, scores[0].tolist(), [int(x) for x in classes[0].tolist()], int(num[0])

    def close(self):
        self.sess.close()
        self.default_graph.close()  

    def find(self, img, boxes_cur, boxes_prev, box):
        #print('## Find called')
        cv2.imwrite('./temporaryImg.jpg',img)

        past_ppl = './past_ppl'
        folders = os.listdir(past_ppl)

        for folder in folders:
            files = os.listdir(past_ppl + '/' + folder)
            same = 0
            diff = 0
            numOfFiles = len(files)
            for f in range(numOfFiles):
                if(f%10 is not 0):
                    continue
                ret = self.reid.compare('./temporaryImg.jpg'  ,    './past_ppl/' + folder + '/' + str(f+1) + '.jpg')
                
                if(ret == True):
                    same += 1
                else:
                    diff += 1
                
            p = 100 * float(same) / float(same + diff)        
            if( p > 70 ):
                person_no = len(files) + 1
                cv2.imwrite(past_ppl + '/' + folder + '/' + str(person_no) + '.jpg',img)   
                boxes_cur[ int(folder) ][0] = box   
                boxes_prev[ int(folder) ] = -1 
                return
        
        l = len(folders)
        os.makedirs(past_ppl + '/' + str( l )  )
        cv2.imwrite(past_ppl + '/' + str( l ) + '/1.jpg',img)
        boxes_cur.append( [box] )
        
        return


def iou(box1, box2):
    xa = max( box1[1] , box2[1] )
    ya = max( box1[0] , box2[0] )
    xb = min( box1[3] , box2[3] )
    yb = min( box1[2] , box2[2] )
    
    interArea = max(0, xb - xa ) * max(0, yb - ya )

    box1Area = (box1[2] - box1[0]) * (box1[3] - box1[1] )
    box2Area = (box2[2] - box2[0]) * (box2[3] - box2[1] )
 
    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = float(interArea) / float(box1Area + box2Area - interArea)

    # return the intersection over union value
    return iou
 

if __name__ == "__main__":
#   model_path = '/path/to/faster_rcnn_inception_v2_coco_2017_11_08/frozen_inference_graph.pb'
    model_path = './model/frozen_inference_graph.pb'

    past_ppl = './past_ppl'

    odapi = DetectorAPI(path_to_ckpt=model_path)
    threshold = 0.8
    iou_threshold = 0.6
    cap = cv2.VideoCapture('./video3.mp4')
    
    #maximum number of previous frames to check iou with
    k = 15
    
    #this will store the bounding boxes detected in the previous frame.
    boxes_prev = []
    framenum = 1
    start_time =  time.time();     #seconds
    time240 = 0
    #iterate over frames
    while True: 
        r, img = cap.read()
        img = cv2.resize(img, (1280, 720))

        boxes, scores, classes, num = odapi.processFrame(img)
        boxes_cur = []
        for l in range(len(boxes_prev)):
            if( len(boxes_prev[l]) < k ):
                boxes_cur.append(  [-1] + boxes_prev[l]  )
            else:
                boxes_cur.append(  [-1] + boxes_prev[l][0:k-1]  )
                
        for i in range(len(boxes)):
            # Class 1 represents human
            if classes[i] == 1 and scores[i] > threshold:
                box = boxes[i]
                
                #draw the bounding box on the image
                cv2.rectangle(img,(box[1],box[0]),(box[3],box[2]),(255,0,0),2)
                
                cropped_img = img[ box[0]:box[2] , box[1]:box[3] ]
                
                maxthreshold = -1
                maxindex = 101     #the index in boxes_prev indicating the matching person from the previous k frames. 
                maxframe = -1
                
                for j in range( len(boxes_prev) ):
                    #Every boxes_prev[j] denotes a person. It is a list of the last k positions of the person j.
                    
                    if( boxes_prev[j] == -1 ):   #This previous person has already been alloted to another person in the current frame 
                        continue
                    
                    for kk in range( len(boxes_prev[j]) ):
                        if(boxes_prev[j][kk] == -1):      #person was not detected in frame kk
                            continue
                        r = iou( boxes_prev[j][kk] ,box)
                        if(  r > maxthreshold  and  r > iou_threshold):
                            maxthreshold = r
                            maxindex = j     
                            maxframe = kk       
                
                if( maxthreshold != -1 ):
                    #Was the tracking correct?
                    b = boxes_prev[ maxindex ][ maxframe ]
                    old_img = img[ b[0]:b[2] , b[1]:b[3] ]
                    cur_img = cropped_img
                    
                    r,c,d = cur_img.shape
                    old_img = cv2.resize(old_img ,(c,r) )
                    
                    res = cv2.matchTemplate(old_img,cur_img,cv2.TM_CCOEFF_NORMED)
                    if(res[0][0] < 0.45):
                        #Tracking was incorrect
                        maxthreshold = -1
       
                    
                #maxthreshold != -1 at this point means this person is the same as prevbox in the last frame. 
                if( maxthreshold != -1 ):
                    #print('tracked ###########')
                    boxes_cur[ maxindex ][0] = box
                    boxes_prev[ maxindex ] = -1
                    
                    #also add this image of the person to his previous images
                    person_no = len( os.listdir( past_ppl + '/' + str(maxindex) ) ) + 1
                    cv2.imwrite(past_ppl + '/' + str(maxindex) + '/' + str(person_no) + '.jpg',cropped_img) 
                else:
                    #The person was not present in the previous frame. Add him to a new folder. 
                    #The folder name should be equal to the index of the person in box_cur.
                    odapi.find(img, boxes_cur,boxes_prev, box )
                    
                
        num_ppl = len(os.listdir(past_ppl))
        #print('#People:   ' + str(num_ppl))        
        print('Time for '+ str(framenum) + ' frames: (seconds)')
        print( time.time() - start_time )
        print('\n')
        
            
        framenum += 1  
        boxes_prev =  boxes_cur

        if(framenum == 240):
            time240 = time.time() - start_time

        if(framenum > 240):
            print('Time for 240 frame:' + str(time240))
            print('\n')
            
        cv2.imshow("preview", img)
        key = cv2.waitKey(1)
        if key & 0xFF == ord('q'):
            break   
            
           
    
            
        
        
        
       
                  
