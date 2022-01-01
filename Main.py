import cv2
import mediapipe as mp
import numpy as np
import time
import autopy
import math
import tensorflow as tf
import matplotlib.pyplot as plt

from pprint import pprint

OD = tf.keras.models.load_model('ocr_test')
OD.summary()	

class LetterObject:
	letter = np.array([])
	letterName = ""
	letterConfident = 0
	letterFrame = np.array([])

class OCR:
	def __init__(self):
		self.labels = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z','a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z']

	def squareCorner(self, img):
		Gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
		filtered = cv2.bilateralFilter(Gray,9,255,255)
		Dial,thresh = cv2.threshold(filtered, np.mean(filtered), 255, cv2.THRESH_BINARY)
		contours, hierarchy = cv2.findContours(image=thresh, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_NONE)
		square = []
		for c in contours:
			x,y,w,h = cv2.boundingRect(c)
			square.append([x,y,w,h])
	
		return square,Dial,thresh

	def predictLetter(self, img):
		letterDrawned = cv2.flip(img, 1)
		square,Dial2,Thres2 = self.squareCorner(letterDrawned)
		letterList = []
		letterObjectList = []
		for n,i in enumerate(square):
			crop_img = letterDrawned[i[1]:i[1]+i[3], i[0]:i[0]+i[2]]
			crop_img = cv2.cvtColor(crop_img,cv2.COLOR_BGR2GRAY)
			crop_img = cv2.resize(crop_img, (28,28), interpolation=cv2.INTER_LINEAR)
			crop_img = crop_img[..., np.newaxis]
			crop_img = crop_img.astype('float32')
			crop_img = crop_img/255.0
			letterList.append(crop_img)
			letterObject = LetterObject()
			letterObject.letter = crop_img
			letterObject.letterFrame = i
			letterObjectList.append(letterObject)

		if len(letterList) != 0:
			crop_img = np.array(letterList)
			predictions = OD.predict(crop_img)
			for i, prediction in enumerate(predictions):
				top_k_values, top_k_indices = tf.nn.top_k(prediction)
				an_array = [top_k_indices.numpy(), top_k_values.numpy()]
				letterObjectList[i].letterName = self.labels[an_array[0][0]]
				letterObjectList[i].letterConfident = an_array[1][0]

		return letterObjectList

wCam, hCam = 1280, 720

cap = cv2.VideoCapture(0)
cap.set(3,wCam)
cap.set(4,hCam)

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
draw = True
fingerStatus = []
x1,y1,x2,y2 = -1,-1,-1,-1
imageDrawned = []
colorArray = [[255,0,0],[0,255,0],[0,0,255]]
colorIndex = 0
changeColor = False
OCR = OCR()
letterDetected = []

with mp_hands.Hands(min_detection_confidence = 0.7, min_tracking_confidence = 0.5, static_image_mode = False) as hands:
	while True:
		success, img = cap.read()
		
		#Detect Hands
		h, w, c = img.shape
		img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
		img.flags.writeable = False
		result = hands.process(img)
		img.flags.writeable = True
		img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
		letterDrawned = np.zeros_like(img)

		empty = True

		for i in imageDrawned:
			img = cv2.line(img, i[0],i[1], color, 6)
			letterDrawned = cv2.line(letterDrawned, i[0],i[1], color, 15)
		
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

				fingerStatus = []
	
				for i in range(5):
					status = False
					text = "False"
					if(finger[i].y < pip[i].y):
						status = True
						text = "True"
					fingerStatus.append(status)
	
					img = cv2.putText(img,text, (int(finger[i].x * w), int(finger[i].y * h)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)
				
				#Hapus Gambar
				#if fingerStatus[0] == True and fingerStatus[1] == False:
				#	imageDrawned = []
				#	empty = True
	
				#Mulai Gambar
				if fingerStatus[1] == True:
					#print("test")
					x1 = int(finger[1].x * w)
					y1 = int(finger[1].y * h)
					if x2 == -1:
						x2,y2 = x1,y1
					imageDrawned.append([(x1,y1),(x2,y2)])
					x2,y2 = x1+1,y1+1
					empty = False
				else: 
					x1,x2,y1,y2 = -1, -1, -1, -1
	
				if fingerStatus[4] == True and fingerStatus[1] == True and changeColor == False:
					changeColor = True
					if colorIndex >= 2:
						colorIndex = 0
					else:
						colorIndex += 1
	
				if fingerStatus[4] == False and fingerStatus[1] == True and changeColor == True:
					changeColor = False
	
				color = colorArray[colorIndex]

		if cv2.waitKey(10) & 0xFF == ord('c'):
			imageDrawned = []
			letterDetected = []
			empty = True
		
		if cv2.waitKey(10) & 0xFF == ord('v'):
			print("test")
			letterDetected = OCR.predictLetter(letterDrawned)

		img = cv2.flip(img, 1)
		img = cv2.resize(img, (wCam,hCam))

		letterDrawned = cv2.flip(letterDrawned, 1)
		if letterDetected != 0:
			for letter in letterDetected:
				i = letter.letterFrame
				label = f"{letter.letterName} | {round(letter.letterConfident*100, 2)} %"
				cv2.rectangle(letterDrawned,(i[0],i[1]),(i[0]+i[2],i[1]+i[3]),(0,255,0),2)
				cv2.putText(letterDrawned, label, (i[0]-20, i[1]-20), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0),2)
		letterDrawned = cv2.resize(letterDrawned, (wCam-420,hCam))
		
		numpy_vertical_concat = np.concatenate((img, letterDrawned), axis=1)
		cv2.imshow("image", numpy_vertical_concat)
		if cv2.waitKey(10) & 0xFF == ord('q'):
			break
