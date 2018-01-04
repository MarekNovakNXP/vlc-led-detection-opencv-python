import numpy as np
import cv2
import time
import video
import os
import random
from math import sin, cos, pi, sqrt
import matplotlib.pyplot as plt
import random


RED = (255,0,0)
GREEN = (0,255,0)
BLUE = (0,0,255)

class TransmitterDetector(object):

    def __init__(self, width, height, markerColor, HISTORY = 20):
        self.height = height
        self.width = width
        self.HISTORY = HISTORY
        # color of marker in the image in BGR format
        self.color = (markerColor[2], markerColor[1], markerColor[0])
        self.inventory=[]
        for i in range(HISTORY):
            self.inventory.append((random.randint(1,width), random.randint(1,height)))
        self.weightHistory = []
        self.likelyhoodHistory = []
        self.errorEvolution = []

    def rejectOutliers(self, data, m=10):
        mean = np.mean(data)
        std = np.std(data)
        treshold = std*m
        out = []

        for d in data:
            if abs(d - mean) <= treshold:
                out.append(d)

        return out

    def addInventory(self, coordinates):

        #add detected coordinates to the inventory
        self.inventory.append(coordinates)

        #drop oldest, keep only HISTORY latest
        while len(self.inventory) > self.HISTORY:
            self.inventory.pop(0)

    def getInventoryCoordinate(self):
        if len(self.inventory) > 0:
            self.xArr = [coord[0] for coord in self.inventory]
            self.yArr = [coord[1] for coord in self.inventory]
            xRejected = self.rejectOutliers(self.xArr)
            yRejected = self.rejectOutliers(self.yArr)

            return (int(np.mean(xRejected)), int(np.mean(yRejected))) 
        else:
            #in case of error, return middle
            self.xArr = []
            self.yArr = []
            return (int(self.width/2), int(self.height/2))

class TemporalMaximumDetector(TransmitterDetector):

    def __init__(self, width=640, height=480, Q = 0.9, markerColor = (255,0,0)):
        super(self.__class__, self).__init__(width, height, markerColor)
        self.lastGray = np.zeros((height,width, 1), np.uint8)
        self.differenceAccumulator = np.zeros((height,width, 1), np.uint8)
        self.Q = Q

        #set label
        self.label = "Temporal maximum detector"

    def run(self, inputFrame):

        #get gray-scale image
        gray = cv2.cvtColor(inputFrame, cv2.COLOR_BGR2GRAY)

        #calculate the difference between this and last gray-scale image
        difference = cv2.absdiff(gray, self.lastGray)

        #save the gray image for next iteration
        self.lastGray = gray.copy()

        #add the result to the accumulator
        self.differenceAccumulator = cv2.add(self.differenceAccumulator, difference)

        #multiply by quotient for exponential filtration
        self.differenceAccumulator = np.uint8(self.differenceAccumulator*self.Q)

        ##cv2.imshow('acc', self.differenceAccumulator)

        #blur to remove the noise
        blurred = cv2.blur(self.differenceAccumulator,(10,10))

        ##cv2.imshow('blurred', blurred)

        #get coordinates of maximum
        mini,maxi,minLoc,maxLoc = cv2.minMaxLoc(blurred)

        #add maximum coordinates to inventory
        self.addInventory(maxLoc)

        return self.getInventoryCoordinate()

class ColorDetector(TransmitterDetector):

    def __init__(self, width=640, height=480,
                L=(171,118,51), H=(5,255,255),
                CLOSING_SIZE = 10, OPENING_SIZE=2,
                markerColor = (0,255,0)):
        super(self.__class__, self).__init__(width, height, markerColor)

        # color boundaries in HSV format
        self.L = L
        self.H = H

        # set closing size in pixels, for removing noise
        self.CLOSING_SIZE = CLOSING_SIZE

        # set opening size in pixels, for removing noise
        self.OPENING_SIZE = OPENING_SIZE

        #set label
        self.label = "Color-based detector"

    def run(self, inputFrame):

        hsvFrame = cv2.cvtColor(inputFrame, cv2.COLOR_BGR2HSV)

        # find the colors within the specified boundaries
        if self.L[0] > self.H[0]: # we are crossing 0degrees
            L1 = np.array([self.L[0], self.L[1], self.L[2]], dtype = "uint8")
            H1 = np.array([179, self.H[1], self.H[2]], dtype = "uint8")
            mask1 = cv2.inRange(hsvFrame, L1, H1)
            L2 = np.array([0, self.L[1], self.L[2]], dtype = "uint8")
            H2 = np.array([self.H[0], self.H[1], self.H[2]], dtype = "uint8")
            mask2 = cv2.inRange(hsvFrame, L2, H2)
            mask = cv2.add(mask1, mask2)
        else:
            L1 = np.array([self.L[0], self.L[1], self.L[2]], dtype = "uint8")
            H1 = np.array([self.H[0], self.H[1], self.H[2]], dtype = "uint8")
            mask = cv2.inRange(hsvFrame, L1, H1)

        ##cv2.imshow('mask', mask)
        
        # apply color mask to original frame
        intersection = cv2.bitwise_and(inputFrame, inputFrame, mask = mask)

        # removes noise from image
        kernel = np.ones((self.OPENING_SIZE, self.OPENING_SIZE), np.uint8)
        opened = cv2.morphologyEx(intersection, cv2.MORPH_OPEN, kernel)

        # make the spot round and filled
        kernel = np.ones((self.CLOSING_SIZE, self.CLOSING_SIZE), np.uint8)
        closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)

        ##cv2.imshow('closed', closed)
        ##cv2.imshow('opened', opened)

        # convert to grayscale
        gray = cv2.cvtColor(closed, cv2.COLOR_BGR2GRAY)

        # apply Hough circles detection
        circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT,
                                    # accumulator resolution
                                   dp=1,
                                   # minimum distance between 2 detected circles
                                   minDist = self.width/8,
                                   # canny edge detector parameters, experimentally
                                   param1=250, param2=5,
                                   # minimum and maximum radii of circles
                                   minRadius=2, maxRadius=self.width/64)

        if not (circles is None):
            for coord in circles[0,:]:
                #add detected spot coordinates to inventory
                self.addInventory(coord)

        return self.getInventoryCoordinate()



class TemporalShapeDetector(TransmitterDetector):

    def __init__(self, width=640, height=480,
                Q = 0.8,
                CLOSING_SIZE = 10, OPENING_SIZE=6,
                markerColor = (0,0,255)):
        super(self.__class__, self).__init__(width, height, markerColor)
        self.lastGray = np.zeros((height,width, 1), np.uint8)
        self.differenceAccumulator = np.zeros((height,width, 1), np.uint8)
        self.Q = Q
        # set closing size in pixels, for removing noise
        self.CLOSING_SIZE = CLOSING_SIZE

        # set opening size in pixels, for removing noise
        self.OPENING_SIZE = OPENING_SIZE

        #set label
        self.label = "Temporal shape detector"

    def run(self, inputFrame):

        #get gray-scale image
        gray = cv2.cvtColor(inputFrame, cv2.COLOR_BGR2GRAY)

        #calculate the difference between this and last gray-scale image
        difference = cv2.absdiff(gray, self.lastGray)

        #save the gray image for next iteration
        self.lastGray = gray.copy()

        #add the result to the accumulator
        self.differenceAccumulator = cv2.add(self.differenceAccumulator, difference)

        #multiply by quotient for exponential filtration
        self.differenceAccumulator = np.uint8(self.differenceAccumulator*self.Q)


        # removes noise from image
        kernel = np.ones((self.OPENING_SIZE, self.OPENING_SIZE), np.uint8)
        opened = cv2.morphologyEx(self.differenceAccumulator, cv2.MORPH_OPEN, kernel)

        # make the spot round and filled
        kernel = np.ones((self.CLOSING_SIZE, self.CLOSING_SIZE), np.uint8)
        closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)

        ##cv2.imshow('closed', closed)
        ##cv2.imshow('opened', opened)

        # apply Hough circles detection
        circles = cv2.HoughCircles(closed, cv2.HOUGH_GRADIENT,
                                    # accumulator resolution
                                   dp=1,
                                   # minimum distance between 2 detected circles
                                   minDist = self.width/8,
                                   # canny edge detector parameters, experimentally
                                   param1=250, param2=6,
                                   # minimum and maximum radii of circles
                                   minRadius=2, maxRadius=self.width/64)

        if not (circles is None):
            for coord in circles[0,:]:
                #add detected spot coordinates to inventory
                self.addInventory(coord)

        return self.getInventoryCoordinate()



class TransmitterTracker(object):

    def __init__(self):
        self.detectors = []
        self.tick = 1

    def addDetector(self, detector):
        self.detectors.append(detector)

    def getCoordinateFusion(self, frame, drawMarkers, showWeightEvolution):
        xFusion = 0
        yFusion = 0
        likelyhoodSum = 0
        detectorsNumber = 0
        detectorIndex = 0
        unmergedCoordinates = []
        initialTransient = 400
        plotPeriod = 2000

        plotNow = False
        if self.tick > initialTransient: # skip the initial settling
            if self.tick % plotPeriod == 0:
                plotNow = True

        if drawMarkers:
            copyFrame = frame.copy()

        for detector in self.detectors:

            detectorsNumber += 1

            #execute the detector
            detector.coordinates = detector.run(frame)
            unmergedCoordinates.append(detector.coordinates)

            std = np.std(detector.xArr)+np.std(detector.yArr)
            #std = np.std(np.diff(detector.xArr, n=2))+np.std(np.diff(detector.yArr, n=2))
            std += 1 # to avoid division by 0
            if (std != std):
                #high value
                std = detector.width+detector.height
            else:
                # draw circles when enabled and when std is reasonable
                if drawMarkers:
                    #draw marker according to std
                    cv2.circle(copyFrame, detector.coordinates, int(10*std), detector.color,2) 

            likelyhood = 1.0/std
            likelyhoodSum += likelyhood
            detector.std = std
            detector.likelyhood = likelyhood

        if plotNow == True:
            plt.figure(1)

        colors = ['r', 'g', 'b', 'k']
        for detector in self.detectors:
            weight = detector.likelyhood/likelyhoodSum
            if self.tick > initialTransient:  # skip the initial settling
                detector.weightHistory.append(weight)
                detector.likelyhoodHistory.append(detector.likelyhood)
            xFusion += weight*detector.coordinates[0]
            yFusion += weight*detector.coordinates[1]

            if plotNow == True:
                plt.plot(detector.likelyhoodHistory, colors[detectorIndex], label=detector.label)
            detectorIndex += 1

        if plotNow == True:
            plt.legend(loc='upper center', shadow=True)
            plt.ylabel('Likelihood P')
            plt.xlabel('Frame number')
            axes = plt.gca()
            axes.set_ylim([0,1])
            plt.show()
        self.tick += 1

        return (int(xFusion), int(yFusion), copyFrame, unmergedCoordinates)


def simulation(noiseLevel = 0.1, onvalue = 255, offvalue = 0, hue=0, lim = 9999):
    mergedErrorEvolution = []

    tracker = TransmitterTracker()
    tracker.addDetector(TemporalMaximumDetector(markerColor=RED))
    tracker.addDetector(ColorDetector(markerColor=GREEN))
    tracker.addDetector(TemporalShapeDetector(markerColor=BLUE))

    counter = 0
    noise = np.zeros((480,640, 3), np.uint8)
    step = 0
    gx = 0
    gy = 0
    h = 2
    s = 255
    v = 255

    while counter < lim:
        start = cv2.getTickCount()
        
        noise = cv2.randn(noise,np.zeros(3),np.ones(3)*255*noiseLevel)  
        counter = counter + 1

        #generate test image
        frame = np.zeros((480,640, 3), np.uint8)

        gx = 300+step*int(100*sin(2*pi*(counter%1000)/1000))
        gy = 300+step*int(100*sin(2*pi*((counter/2)%1000)/1000))
        s = 255

        if counter % 2 == 0:
            v = offvalue
        else:
            v = onvalue

        h = hue

        hsv  = np.uint8([[[h,s,v]]])
        bgr = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)
        color = (int(bgr[0][0][0]), int(bgr[0][0][1]), int(bgr[0][0][2]))
        cv2.circle(frame, (gx, gy), 3, color, 6) 

        v = 255
        h = 20
        for i in range(0):
            gx += 10*i
            gy += 5*i
            h += 5*i
            hsv  = np.uint8([[[h,s,v]]])
            bgr = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)
            color = (int(bgr[0][0][0]), int(bgr[0][0][1]), int(bgr[0][0][2]))
            cv2.circle(frame, (gx, gy), 4, color, 6) 



        #add noise and generated frame
        frame = cv2.add(frame, noise)

        (mx,my,processedFrame, unmergedCoordinates) = tracker.getCoordinateFusion(frame, True, True)

        
        tracker.detectors[0].errorEvolution.append(sqrt((unmergedCoordinates[0][0]-gx)**2+(unmergedCoordinates[0][1]-gy)**2))
        tracker.detectors[1].errorEvolution.append(sqrt((unmergedCoordinates[1][0]-gx)**2+(unmergedCoordinates[1][1]-gy)**2))
        tracker.detectors[2].errorEvolution.append(sqrt((unmergedCoordinates[2][0]-gx)**2+(unmergedCoordinates[2][1]-gy)**2))
        mergedErrorEvolution.append(sqrt((mx-gx)**2+(my-gy)**2))

        stop = cv2.getTickCount()
        elapsed = (stop - start) / cv2.getTickFrequency()
        print("Cycle time: %d ms" % int(1000*elapsed))

        cv2.imshow('frame', processedFrame)
        cv2.waitKey(1)

        #noiseLevel = counter/2000.0
        print counter

    plt.figure(1)
    plt.plot(tracker.detectors[0].errorEvolution, 'r', label="Temporal Maximum Detector")
    plt.plot(tracker.detectors[1].errorEvolution, 'g', label="Color Detector")
    plt.plot(tracker.detectors[2].errorEvolution, 'b', label="Temporal Shape Detector")
    plt.plot(mergedErrorEvolution, 'k', label="Merged Detector")
    plt.show()
    cv2.destroyAllWindows()

def webcameraTest():
    tracker = TransmitterTracker()
    tracker.addDetector(TemporalMaximumDetector(markerColor=RED)) #width=1280, height = 720))
    tracker.addDetector(ColorDetector(markerColor=GREEN)) #width=1280, height = 720))
    tracker.addDetector(TemporalShapeDetector(markerColor=BLUE)) #width=1280, height = 720))

    cap = cv2.VideoCapture(0)

    while(1):
        start = cv2.getTickCount()
        _, frame = cap.read()

        (x,y,processedFrame) = tracker.getCoordinateFusion(frame, True, True)

        stop = cv2.getTickCount()
        elapsed = (stop - start) / cv2.getTickFrequency()
        print("Cycle time: %d ms" % int(1000*elapsed))

        cv2.imshow('frame', processedFrame)
        cv2.waitKey(1)

    cap.release()
    cv2.destroyAllWindows()


simulation(lim=1000)
simulation(hue=50, lim=1000)
simulation(noiseLevel=0.5, onvalue=128, offvalue=(128+25), lim=1000)
#webcameraTest()

