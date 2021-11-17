import cv2
import mediapipe as mp
import numpy as np
import time
import autopy
import math

from pprint import pprint

def getDistance(p1,p2):
	return math.sqrt(10 * pow(p1.x-p2.x,2) + 10 * pow(p1.y-p2.y,2))

wCam, hCam = 780,720

cap = cv2.VideoCapture(0)
cap.set(3,wCam)
cap.set(4,hCam)

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
img = "Test Data/20211003_154324.jpg"
img = cv2.imread(img)
draw = True
fingerStatus = []
x1,y1,x2,y2 = -1,-1,-1,-1
imageDrawned = []
colorArray = [[255,0,0],[0,255,0],[0,0,255]]
colorIndex = 0
changeColor = False

with mp_hands.Hands(min_detection_confidence = 0.8, min_tracking_confidence = 0.5, static_image_mode = False) as hands:
	while True:
		success, img = cap.read()
	
		#Detect Hands
		h, w, c = img.shape
		img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
		img.flags.writeable = False
		result = hands.process(img)
		img.flags.writeable = True
		img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

	#Render Image
		if result.multi_hand_landmarks:
			for num, hand in enumerate(result.multi_hand_landmarks):
				mp_drawing.draw_landmarks(img, hand, mp_hands.HAND_CONNECTIONS)
				#img = cv2.rectangle(img, (int(hand.landmark[3].x * 640), int(hand.landmark[3].y * 640)),(int(hand.landmark[4].x * 640), int(hand.landmark[4].y * 640)),(0,255,0),3) 
				finger = []
				for i in range(1,6):
					finger.append(hand.landmark[4*i])
				knuckle = [hand.landmark[5], hand.landmark[5], hand.landmark[9], hand.landmark[13], hand.landmark[17]]
				pip = [hand.landmark[2], hand.landmark[6], hand.landmark[10], hand.landmark[14], hand.landmark[18]]
				#Right Thumbs
				img = cv2.rectangle(img, (int(hand.landmark[3].x * w), int(hand.landmark[3].y * h)), (int(hand.landmark[4].x * w),int(hand.landmark[4].y * h)),(0,255,0),3) 
				#Right Index
				img = cv2.rectangle(img, (int(hand.landmark[7].x * w), int(hand.landmark[7].y * h)), (int(hand.landmark[8].x * w),int(hand.landmark[8].y * h)),(0,255,0),3) 
				#Right Middle
				img = cv2.rectangle(img, (int(hand.landmark[11].x * w), int(hand.landmark[11].y * h)), (int(hand.landmark[12].x * w),int(hand.landmark[12].y * h)),(0,255,0),3) 
				#Right Ring
				img = cv2.rectangle(img, (int(hand.landmark[15].x * w), int(hand.landmark[15].y * h)), (int(hand.landmark[16].x * w),int(hand.landmark[16].y * h)),(0,255,0),3) 
				#Right Pingky
				img = cv2.rectangle(img, (int(hand.landmark[19].x * w), int(hand.landmark[19].y * h)), (int(hand.landmark[20].x * w),int(hand.landmark[20].y * h)),(0,255,0),3) 
				
				Pivot = hand.landmark[0]
			distanceFinger = []
			distanceKnuckle = []
			for i in range(5):
				distanceFinger.append(getDistance(finger[i],Pivot))
				distanceKnuckle.append(getDistance(knuckle[i],Pivot))

			fingerStatus = []


			for i in range(5):
				status = False
				text = "False"
				if(finger[i].y < pip[i].y):
					status = True
					text = "True"
				fingerStatus.append(status)
				#if draw == True:
				img = cv2.putText(img,text, (int(finger[i].x * w), int(finger[i].y * h)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)
			
			#Hapus Gambar
			if fingerStatus[0] == True and fingerStatus[1] == False:
				imageDrawned = []

			#Mulai Gambar
			if fingerStatus[1] == True:
				#print("test")
				x1 = int(finger[1].x * w)
				y1 = int(finger[1].y * h)
				if x2 == -1:
					x2,y2 = x1,y1
				imageDrawned.append([(x1,y1),(x2,y2)])
				x2,y2 = x1+1,y1+1
			else:
				x1,x2,y1,y2 = -1,-1,-1,-1	

			if fingerStatus[4] == True and fingerStatus[1] == True and changeColor == False:
				changeColor = True
				if colorIndex >= 2:
					colorIndex = 0
				else:
					colorIndex += 1

			if fingerStatus[4] == False and fingerStatus[1] == True and changeColor == True:
				changeColor = False

			color = colorArray[colorIndex]

		for i in imageDrawned:
			img = cv2.line(img, i[0],i[1], color, 6)

		#img = cv2.rectangle(img, (int(hand.landmark[i].x * w), int(hand.landmark[i].y * h)), (int(hand.landmark[i+1].x * w),int(hand.landmark[i+1].y * h)),(0,255,0),3) 
		#print(hand.landmark[0].x)
		#for i in range(len(hand.landmark)-1):
		#	img = cv2.rectangle(img, (int(hand.landmark[i].x * 640), int(hand.landmark[i].y * 640)), (int(hand.landmark[i+1].x * 640),int(hand.landmark[i+1].y * 640)),(0,255,0),3) 
		#print(f"Thumbs : {Thumbs}\nIndex : {Index}\nMiddle : {Middle}\nRing : {Ring}\nPingky : {Pingky}")
		
		if draw == True:
			pprint(vars(result))
			draw = False
		
		img = cv2.resize(img, (wCam,hCam))
		img = cv2.flip(img, 1)
		#while True:
		cv2.imshow("image", img)
		if cv2.waitKey(10) & 0xFF == ord('q'):
			print(distanceFinger)
			print("Knucle : ")
			print(distanceKnuckle)
			print(fingerStatus)
			break

'''
	for id, lm in enumerate(results.multi_hand_landmarks[hand_no].landmark):
                    for index in indexes:
                        if id == index:
                            h, w, c = img.shape
                            x, y = int(lm.x*w), int(lm.y*h)
                            lst.append((x,y))
'''