import os
import numpy as np
#from scipy import ndimage
import sys
from PIL import Image
from skimage.feature import local_binary_pattern
from sklearn.preprocessing import normalize
from scipy.stats import itemfreq
import matplotlib.pyplot as plt
from sklearn import svm

def main():
    
    root = 'C:\\Users\\aratr\\Desktop\\MACHINE LEARNING PROJECT\\589\\Pandora_V1\\TESTING DATA'
    print 'Printing from the Test class'
    image_paths = []
    X = []
    for path, subdirs, files in os.walk(root):
        for name in files:
            image_paths.append(os.path.join(path, name))
            im = Image.open(os.path.join(path,name))
            im = im.convert("L")
            X.append(np.asarray(im, dtype=np.uint8))
            #print len(X)

    print len(X)

    labels = []
    paintings_name = []
    for i in range(len(image_paths)):
        x = image_paths[i]
        y = x.split('\\')
        labels.append(y[4])
        paintings_name.append(y[5])

    #Computation of LBP features

    X_feat = []
    radius = 3
    no_points = 8*radius

    for img in X:
        #plt.imshow(img)
        lbp = local_binary_pattern(img, no_points, radius, method='uniform')
        x = itemfreq(lbp.ravel())
        hist = x[:,1]/sum(x[:,1])
        X_feat.append(hist)


    print len(X_feat)
    return X_feat,labels
    
if __name__ == "__main__":
    main();
    