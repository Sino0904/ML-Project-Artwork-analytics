from PIL import Image, ImageDraw
import os
import numpy as np
import sys
import matplotlib.pyplot as plt

def get_colors(img, numcolors=10, swatchsize=20, resize=150):

    
    #EVALUATION OF COLOR FEATURE
    result = image.convert('P', palette=Image.ADAPTIVE, colors=numcolors)
    result.putalpha(0)
    colors = result.getcolors(resize*resize)

    

