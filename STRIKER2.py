#!/usr/bin/env python
import cv2
import numpy as np
import matplotlib.pyplot as plt

#f=open('result.csv','a')
Src=cv2.VideoCapture("/home/smriti/Desktop/8th_Semester/7th_Semester/ProjectIII/PROGRAMS_v2/2018-10-16/video1.mp4") #input video
#Src=cv2.VideoCapture("/home/smriti/Desktop/7th_Semester/ProjectIII/Carrom_VIDS/video6.mp4")
#print(str("frame number")+","+str("black")+","+str("white"),file=open('/home/smriti/Desktop/7th_Semester/ProjectIII/PROGRAMS_V3/result.csv','a'))
def getFrame(sec):
        Src.set(cv2.CAP_PROP_POS_MSEC,sec*1000)
        hasFrames,image = Src.read()
        if hasFrames:
#                cv2.imwrite("frame "+str(sec)+" sec.jpg", image)     # save frame as JPG file
                return hasFrames,image
def Camera():
        i=0
        radius1=0.0
        radius2=0.0
        radius3=0.0
        radius4=0.0
        sec = 0
        frameRate = 1#it will capture image in each 0.5 second
        success = getFrame(sec)
        while success:
                sec = sec + frameRate
                sec = round(sec, 2)
                success,frame = getFrame(sec)               
                

                frame_resize = cv2.resize(frame, (400,300)) # frame resizing 
                
                gray1=cv2.cvtColor(frame_resize,cv2.COLOR_BGR2GRAY)
                luv=cv2.cvtColor(frame_resize,cv2.COLOR_BGR2LUV)
                gray2=cv2.cvtColor(luv,cv2.COLOR_BGR2GRAY)
#                lab=cv2.cvtColor(frame_resize,cv2.COLOR_BGR2Lab)
#                gray4=cv2.cvtColor(lab,cv2.COLOR_BGR2GRAY)
                
                def color_selection(color):
                        if color=='black':
                                low=np.array([0])
                                high=np.array([40])
                        if color=='white':
                                low=np.array([170])
                                high=np.array([255])
                        if color=='pink':
                                low=np.array([130])
                                high=np.array([134])
                        if color=='brown':
                                low=np.array([104])
                                high=np.array([104])
                        return low,high
                                
                black_low,black_high=color_selection('black')
                white_low,white_high=color_selection('white')
                brown_low,brown_high=color_selection('brown')
                pink_low,pink_high=color_selection('pink')

                mask_b=cv2.inRange(gray1,black_low,black_high)#masking for black dice
                #cv2.imshow('mask_b',mask_b)
                        
                ## Removal of corner pixels
                mask_b[0:30, 0:30] = 0 # removing Top-left corner pixels
                mask_b[275:300, 0:30] = 0 # removing Bottom-left corner pixels
                mask_b[275:300, 370:400] = 0 # removing Bottom-right corner pixels
                mask_b[0:30, 370:400] = 0 # removing Top-right corner pixels

                mask_w=cv2.inRange(gray1,white_low,white_high)#masking for white dice
                mask_p=cv2.inRange(gray2,pink_low,pink_high) #For pink dice
                mask_br=cv2.inRange(gray2,brown_low,brown_high)

                
                kernel = np.ones((3,3),np.uint8)#defining kernel for noise reduction
                kernel2 = np.ones((2,2),np.uint8)#defining kernel for noise reduction

                erosion_b = cv2.erode(mask_b,kernel,iterations = 1)#erosion for black dice
                erosion_w = cv2.erode(mask_w,kernel,iterations = 1)#erosion for white dice
                dilate_p = cv2.dilate(mask_p,kernel,iterations = 4)
                erosion_p = cv2.erode(dilate_p,kernel,iterations = 2)
                erosion_br = cv2.erode(mask_br,kernel2,iterations = 2)
                erosion_br = cv2.erode(mask_br,kernel,iterations = 2)
                dilate_br = cv2.dilate(erosion_br,kernel,iterations = 5)


                image1, contours1, hier1 = cv2.findContours(erosion_b, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
                image2, contours2, hier2 = cv2.findContours(erosion_w, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
                image3, contours3, hier3 = cv2.findContours(erosion_p, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
                image4, contours4, hier3 = cv2.findContours(dilate_br, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

                
                for c1 in contours1:
                        xb, yb, wb, hb = cv2.boundingRect(c1)
                        rect = cv2.minAreaRect(c1)
                        box = cv2.boxPoints(rect)
                        box = np.int0(box)
                        (xb, yb), radius1 = cv2.minEnclosingCircle(c1)
                        center = (int(xb), int(yb))
                        radius1 = int(radius1)
                        frame_resize = cv2.circle(frame_resize, center, radius1, (0, 255, 0), 2)
                for c2 in contours2:        
                        x, y, w, h = cv2.boundingRect(c2)
                        rect = cv2.minAreaRect(c2)
                        box = cv2.boxPoints(rect)
                        box = np.int0(box)
                        (x, y), radius2 = cv2.minEnclosingCircle(c2)
                        center = (int(x), int(y))
                        radius2 = int(radius2)
                        frame_resize = cv2.circle(frame_resize, center, radius2, (255, 0, 0), 2)
                for c3 in contours3:        
                        x, y, w, h = cv2.boundingRect(c3)
                        rect = cv2.minAreaRect(c3)
                        box = cv2.boxPoints(rect)
                        box = np.int0(box)
                        (x, y), radius3 = cv2.minEnclosingCircle(c3)
                        center = (int(x), int(y))
                        radius3 = int(radius3)
                        frame_resize = cv2.circle(frame_resize, center, radius3, (0, 0, 0), 2)
                for c4 in contours4:        
                        x, y, w, h = cv2.boundingRect(c4)
                        rect = cv2.minAreaRect(c4)
                        box = cv2.boxPoints(rect)
                        box = np.int0(box)
                        (x, y), radius4 = cv2.minEnclosingCircle(c4)
                        center = (int(x), int(y))
                        radius4 = int(radius4)
                        frame_resize = cv2.circle(frame_resize, center, radius4, (255, 153, 255), 2)
        #        print(radius4)
                i=i+1
                if radius4>radius1 and radius4>0.0:
                        if not(len(contours1)>9) and not(len(contours2)>9):
                                print(str("frame number :")+str(i)+","+str("black : ")+str(len(contours1))+","+str("white : ")+str(len(contours2)))
                if radius4>radius1 and radius4>0.0:
                        if not(len(contours1)>9) and not(len(contours2)>9):
#                                print(str("frame number :")+str(i)+","+str("black : ")+str(len(contours1))+","+str("white : ")+str(len(contours2)),file=open('/home/smriti/Desktop/7th_Semester/ProjectIII/PROGRAMS_V3/result.csv','a'))
                                print(str(i)+","+str(len(contours1))+","+str(len(contours2)),file=open('/home/smriti/Desktop/8th_Semester/7th_Semester/ProjectIII/PROGRAMS_V3/result.csv','a'))
                

                        
                        
                
                cv2.imshow('contours',frame_resize) # final window with dices detected


                k = cv2.waitKey(5) & 0xFF
                if k == 27:
                    break
        cv2.destroyAllWindows()
Camera()






















