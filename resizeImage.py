
import cv2

def resizeIm(image,size):
	r = float(size) / image.shape[1]
	dim = (size, int(image.shape[0] * r))
	# perform the actual resizing of the image and show it
	resized = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
	return resized