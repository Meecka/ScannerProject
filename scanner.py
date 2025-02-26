import cv2
import numpy
import mapper

img = cv2.imread("output.png")
orig=img.copy()
cv2.imshow('Original', img)
cv2.waitKey(0)

kernel = numpy.ones((5,5),numpy.uint8)
img_blank = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel, iterations= 3)
cv2.imshow('Blank', img_blank)
cv2.waitKey(0)

img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# cv2.imshow('Gray', img_gray)
img_blur = cv2.GaussianBlur(img_gray, (3,3), 0) 
# cv2.imshow('Blur', img_blur)

sobelx = cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=1, dy=0, ksize=5)
sobely = cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=0, dy=1, ksize=5)
sobelxy = cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=1, dy=1, ksize=5)

cv2.imshow('Sobel X', sobelx)
cv2.waitKey(0)

cv2.imshow('Sobel Y', sobely)
cv2.waitKey(0)

cv2.imshow('Sobel X Y using Sobel() function', sobelxy)
cv2.waitKey(0)

edged = cv2.Canny(image=img_blur, threshold1=100, threshold2=200) # Canny Edge Detection

cv2.imshow('Canny Edge Detection', edged)
cv2.waitKey(0)

contours,hierarchy=cv2.findContours(edged,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
contours=sorted(contours,key=cv2.contourArea,reverse=True)

for c in contours:
    p=cv2.arcLength(c,True)
    approx=cv2.approxPolyDP(c,0.02*p,True)

    if len(approx)==4:
        target=approx
        break
approx=mapper.mapp(target)

pts=numpy.float32([[0,0],[800,0],[800,800],[0,800]]) 

op=cv2.getPerspectiveTransform(approx,pts)
dst=cv2.warpPerspective(orig,op,(800,800))

cv2.imshow('test', dst)
cv2.waitKey(0)

cv2.destroyAllWindows()

# image=cv2.imread("output.jpg")   #read in the image
# image=cv2.resize(image,(1300,800)) #resizing because opencv does not work well with bigger images
# orig=image.copy()

# gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)  #RGB To Gray Scale
# cv2.imshow("Title",gray)

# blurred=cv2.GaussianBlur(gray,(5,5),0)  #(5,5) is the kernel size and 0 is sigma that determines the amount of blur
# cv2.imshow("Blur",blurred)

# edged=cv2.Canny(blurred,30,50)  #30 MinThreshold and 50 is the MaxThreshold
# cv2.imshow("Canny",edged)
# cv2.imwrite("edged.png", image)


# cv2.imshow("Scanned",dst)
# press q or Esc to close
# cv2.waitKey(0)
# cv2.destroyAllWindows()

