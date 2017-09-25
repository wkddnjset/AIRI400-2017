import cv2
cap = cv2.VideoCapture(0)

while(cap.isOpened()):
	ret,frame = cap.read()
	if ret == True:
		rec_point1 = (50, 50)
		rec_point2 = (100, 100)
		cv2.rectangle(frame, rec_point1, rec_point2, (0,255,0))

		cv2.putText(frame, "Input Text", (text_start_point_x, text_start_point_y),
			cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0))
		cv2.imshow(frame)
cap.release()