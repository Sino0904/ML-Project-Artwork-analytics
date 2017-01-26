import os
import numpy as np
#from scipy import ndimage
import sys
from PIL import Image
from skimage.feature import local_binary_pattern
from sklearn.preprocessing import normalize
from scipy.stats import itemfreq
import matplotlib.pyplot as plt
from Color_descriptor import ColorDescriptor
from HSVColor import HSVColor
from resizeImage import resizeIm
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from sklearn.externals import joblib
from sklearn.metrics import accuracy_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import label_binarize
def main():
    
    #Testing
    #model_path = 'C:\\Users\\aratr\\Desktop\MACHINE LEARNING PROJECT\\clf1_ml.pkl'

    
    root_t = 'C:\\Python27\\ml_proj\\TESTING DATA'
    #root_t = 'C:\\Python27\\ml_proj\\Artist_Testing'
    #root_t = 'C:\\Users\\aratr\\Desktop\\MACHINE LEARNING PROJECT\\589\\Pandora_V1\\VALIDATION DATA'
    image_paths_test = []
    X1 = []
    X2 = []
    size_new = 256
    for path_t, subdirs_t, files_t in os.walk(root_t):
        for name_t in files_t:
            image_paths_test.append(os.path.join(path_t, name_t))
            im = Image.open(os.path.join(path_t,name_t))
            #im.thumbnail(size, Image.ANTIALIAS)
            im_gray = im.convert("L")
            im_hsv = HSVColor(resizeIm(np.asarray(im, dtype=np.uint8),size_new))
            X1.append(resizeIm(np.asarray(im_gray, dtype=np.uint8),size_new))
            X2.append(im_hsv)
            #print len(X)

    print len(X1)
    print X1[0].shape
    print len(X2)
    print X2[0].shape

    labels_names = []
    labels_number = []
    paintings_name = []
    for i in range(len(image_paths_test)):
        x = image_paths_test[i]
        y = x.split('\\')
        labels_names.append(y[4])
        if(y[4] == 'Cubism'):
            labels_number.append(1)
        if(y[4] =='Impressionalism'):
            labels_number.append(2)
        if(y[4] =='Romanticism'):
            labels_number.append(3)
        paintings_name.append(y[5])

    labels_number2 = label_binarize(labels_number, classes=[1,2,3])
    
    
    #Computation of LBP features

    LBP_feat = []
    radius = 3
    no_points = 8*radius

    for img in X1:
        #plt.imshow(img)
        lbp = local_binary_pattern(img, no_points, radius, method='uniform')
        x = itemfreq(lbp.ravel())
        hist = x[:,1]/sum(x[:,1])
        LBP_feat.append(hist)


    print len(LBP_feat)
    print type(LBP_feat[0])
    print LBP_feat[0].shape
    #print LBP_feat[0]

    #Computation for color descriptor
    CSD_feat = []
    # initialize the color descriptor
    cd = ColorDescriptor((8, 12, 3))

    for img in X2:
        csd_features = cd.describe(img)
        CSD_feat.append(np.array(csd_features))

    print len(CSD_feat)
    print type(CSD_feat[0])
    print CSD_feat[0].shape
    #print CSD_feat[0]

    final_set = np.array([]).reshape(0,LBP_feat[0].shape[0] + CSD_feat[0].shape[0])
    for i in range(len(LBP_feat)):
        features_lbp = LBP_feat[i]
        features_csd = CSD_feat[i]
        comb_feat = np.concatenate((features_lbp,features_csd))
        final_set = np.vstack((final_set,comb_feat))

    print final_set.shape

    
    return final_set,labels_number2,labels_number
if __name__ == "__main__":
    main();