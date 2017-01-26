import cv2

def HSVColor(img):

   img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
   #print 'working'
   return img_hsv