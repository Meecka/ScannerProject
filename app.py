import cv2 as cv

cam = cv.VideoCapture(0)

cv.namedWindow("Python Webcam Screenshot")

img_counter = 0

while True:
  ret,frame = cam.read()

  if not ret:
    print("failed to grab frame")
    break

  cv.imshow("test", frame)

  k = cv.waitKey(1)

  if k%256 == 27:
    print("Escape hit, closing...")
    break

  elif k%256 == 32:
    img_name = "opencv_frame_{}.png".format(img_counter)
    cv.imwrite(img_name, frame)
    print("screenshot taken")
    img_counter += 1

cam.release()

cam.destroyAllWindows()