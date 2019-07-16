#Test code for human_detect.py
#Runs version0 which returns a list of identities, which serves as ground truth
#Then runs version1 which returns another list of indentities
#Then compares both lists

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

        #print("Elapsed Time:", end_time-start_time)

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

    '''
Version0 serves as ground truth .It finds a match for the person and returns the matching folder number. If no match is found, it creates a new folder and returns the match for  that.
    '''
    def version0(self, img):
        cv2.imwrite('./temporaryImg_test0.jpg',img)

        past_ppl = './past_ppl_test_version0'
        folders = os.listdir(past_ppl)

        for folder in folders:
            files = os.listdir(past_ppl + '/' + folder)
            for f in files:
                ret = self.reid.compare('./temporaryImg_test0.jpg'  ,    './past_ppl_test_version0/' + folder + '/' + f)
                
                if(ret == True):
                    person_no = len(files) + 1
                    cv2.imwrite(past_ppl + '/' + folder + '/' + str(person_no) + '.jpg',img)      
                    return int(folder)
        
        l = len(folders)
        os.makedirs(past_ppl + '/' + str( l )  )
        cv2.imwrite(past_ppl + '/' + str( l ) + '/1.jpg',img)
        return l
     
    '''
Version0 serves as ground truth 
    '''
    def isSamePerson(self, img1, img2):
        cv2.imwrite('./temporaryImg_test0.jpg',img1)
        cv2.imwrite('./temporaryImg_test1.jpg',img2)
        ret = self.reid.compare('./temporaryImg_test0.jpg'  ,  './temporaryImg_test1.jpg')        
        return ret

    def find(self, img, boxes_cur, box):
        cv2.imwrite('./temporaryImg_test1.jpg',img)

        past_ppl = './past_ppl_test_version1'
        folders = os.listdir(past_ppl)

        for folder in folders:
            files = os.listdir(past_ppl + '/' + folder)
            for f in files:
                ret = self.reid.compare('./temporaryImg_test1.jpg'  ,    './past_ppl_test_version1/' + folder + '/' + f)
                
                #ret = run(past_ppl + '/' + folder + '/' + f , './temporaryImg.jpg')
                if(ret == True):
                    person_no = len(files) + 1
                    cv2.imwrite(past_ppl + '/' + folder + '/' + str(person_no) + '.jpg',img)   
                    boxes_cur[ int(folder) ] = box    
                    return int(folder)
        
        l = len(folders)
        os.makedirs(past_ppl + '/' + str( l )  )
        cv2.imwrite(past_ppl + '/' + str( l ) + '/1.jpg',img)
        boxes_cur.append( box )
        return -1
        
    '''
    This function is called when the tracking mechanism has tracked a person.
    This code tests if the track was correct or not
    '''    
    def tester(self, img_cur, img_prev):
        cv2.imwrite('./temporaryImg_cur.jpg',img_cur)
        cv2.imwrite('./temporaryImg_prev.jpg',img_prev)
        return self.reid.compare('./temporaryImg_cur.jpg'  , './temporaryImg_prev.jpg')


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
    iou_threshold = 0.7
    cap = cv2.VideoCapture('./video.avi')
    
    #this will store the bounding boxes detected in the previous frame.
    boxes_prev = []
    
    start_time =  time.time();     #seconds
    
    framenum = 0
    
    
    num_bbox_correct = 0
    num_bbox_total = 0
    prev_frame = None
    prev_box = None
    
    #iterate over frames
    while True:         
        r, img = cap.read()
        img = cv2.resize(img, (1280, 720))

        boxes, scores, classes, num = odapi.processFrame(img)
        boxes_cur = [-1] * len(boxes_prev)

        for i in range(len(boxes)):
            if classes[i] == 1 and scores[i] > threshold:
                box = boxes[i]
                num_bbox_total += 1  
                  
                #draw the bounding box on the image
                cv2.rectangle(img,(box[1],box[0]),(box[3],box[2]),(255,0,0),2)
                cropped_img = img[ box[0]:box[2] , box[1]:box[3] ]

                maxthreshold = -1
                maxindex = 101
                
                for j in range( len(boxes_prev) ):
                    if( boxes_prev[j] == -1 ):
                        continue
                    r = iou( boxes_prev[j] ,box)
                    
                    if(r < iou_threshold):
                        continue
                    if(  r > maxthreshold  ):
                        maxthreshold = r
                        maxindex = j     
                        prev_box = boxes_prev[j]        
                        
                #maxthreshold != -1 at this point means this person is the same as prevbox in the last frame. 
                if( maxthreshold != -1 ):
                    print('### TRACKED ###')
                    box_prev = boxes_prev[maxindex]
                    cropped_img_prev = img[ box_prev[0]:box_prev[2] , box_prev[1]:box_prev[3] ]
                    #correct = odapi.tester(cropped_img, cropped_img_prev)
                    
                    #test
                    if(prev_frame is not None):
                        prev_cropped_img = prev_frame[ prev_box[0]:prev_box[2] , prev_box[1]:prev_box[3] ]
                        if( odapi.isSamePerson(cropped_img, prev_cropped_img) ):
                            num_bbox_correct += 1
                    
                    
                    boxes_cur[ maxindex ] = box
                    boxes_prev[ maxindex ] = -1
                    
                    #also add this image of the person to his previous images
                    person_no = len( os.listdir( past_ppl + '/' + str(maxindex) ) ) + 1
                    cv2.imwrite(past_ppl + '/' + str(maxindex) + '/' + str(person_no) + '.jpg',cropped_img) 

                    if(maxindex == ground_truth):
                        num_bbox_correct += 1
                '''
                #maxthreshold == -1 at this point means this person was not present in last frame and needs to be re-identified
                else:
                    prev_per = odapi.find(cropped_img, boxes_cur, box)
                    
                    if(prev_per == ground_truth):
                        num_bbox_correct += 1

                    try:
                        if(prev_per != -1):
                            if(boxes_prev[prev_per] != -1):
                                r = iou( boxes_prev[prev_per] ,box)
                                if(r > iou_threshold  and r > maxthreshold ):
                                    #it should have tracked, but it re-id instead
                                    reidShouldveTracked += 1
                        else:
                            reidNew += 1        
                    except IndexError:
                        print('Index error in boxes_prev list')
                '''
        if(framenum % 1 == 0):    
            num_ppl = len(os.listdir(past_ppl))
            print('\nFrame number: '+str(framenum))
            print('#People:   ' + str(num_ppl))
            print('#bbox tracked correct: '+str(num_bbox_correct))
            print('#bbox : '+str(num_bbox_total))
            pos = 100*(float(num_bbox_correct) / float(num_bbox_total) )
            print('%bbox tracked correct: '+str( pos  ) + '\n--------------\n')
            
        framenum += 1  
        prev_frame = img
        boxes_prev =  boxes_cur
        cv2.imshow("preview", img)
        key = cv2.waitKey(1)
        if key & 0xFF == ord('q'):
            break   
            
           
