""" A Take on - One Shot Frequency Dominant Neighborhood Structure(https://www.sciencedirect.com/science/article/pii/S0262885621001505) : 
An improvement on novel Perceptual Hasing technique. The scheme is as follows :
Classical Way : 

    #Step 1: Image Preprocessing - The first step after detecting a face(MTCNN) is to crop it.
    The cropped face images is resized to 120 x 120 pixels(using a bi-linear interpolation). It
    is then converted to grayscale and smoothed using a Gaussian low-pass filter. 

    #Step 2: Feature Extraction - Features are extracted from the image using DCT/DWT/DFT and the
    Dominant Neighborhood Structure(Khellah) :
        First, the Image is divided into 4 non-overlapping blocks of 60 x 60 pixels, then DCT is 
        applied to each block to obtain four frequency blocks(complete energy distribution of image
        is frequency domain)

        Secondly, Dominant Neighborhood Structure is applied to each frequency blocks to extract 
        features from its texture(Use 3 x 3 neighborhood window).FRom each DCT transformed block,
        a single DNS map is extracted. Therefore, 4 OSF-DNS maps are extracted from the input face 
        image.


    #Step 3: Hash Generation - The global texture energy features are computed by summing up the
    four extracted OSF-DNS maps. As a result, OSF-GNS(... Global Neighborhood Structure). The 
    final image hash is composed by the coefficients of the  OSF-GNS map except the first row 
    and column. OSF-DNS hash code of each image is composed of 64 real values.    

New Approach :     
"""
import os 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import cv2
import functools
from operator import xor, lshift
from mtcnn import MTCNN
import numpy
import tensorflow as tf

def DCT(gs_part_img):
    imf = numpy.float32(gs_part_img)/255.0
    dct_img = cv2.dct(imf)
    return numpy.uint8(dct_img*255)


img = cv2.cvtColor(cv2.imread('Self_New_Photo.jpg'), cv2.COLOR_BGR2RGB)
detecteor = MTCNN()

jobj = detecteor.detect_faces(img)
# print(tf.config.list_physical_devices('GPU'))

x, y, w, h = jobj[0]['box']

# w = x1-x0
# h = y1-y0

cv2.rectangle(img, (x,y),(x+w,y+h), color=(0,0,255) )

new_img = img[y:y+h , x:x+w]  # After cropping the image
# cv2.imshow('Img', new_img)

red_img = cv2.resize(new_img, (120, 120), interpolation=cv2.INTER_LINEAR, )
# cv2.imshow('Img', red_img)

gs_img = cv2.cvtColor(red_img, cv2.COLOR_BGR2GRAY)
# cv2.imshow('Img', gs_img)

bl_img = cv2.GaussianBlur(gs_img, (3,3), 0.1)
# cv2.imshow('Img', bl_img)

fin_img = bl_img

a1, a3, a2, a4 = fin_img[0:60, 0:60], fin_img[60:120, 0:60], fin_img[0:60, 60:120], fin_img[60:120, 60:120]
# cv2.imshow('Img', a4) 

f_a1 = DCT(a1)
# cv2.imshow('Img', f_a1)

f_a2 = DCT(a2)
# cv2.imshow('Img', f_a2)

f_a3 = DCT(a3)
# cv2.imshow('Img', f_a3)

f_a4 = DCT(a4)
# cv2.imshow('Img', f_a4)
sift = cv2.SIFT_create()
kp_f_a1, des_f_a1 = map(list, sift.detectAndCompute(f_a1, None))

kp_f_a2, des_f_a2 = map(list, sift.detectAndCompute(f_a2, None))

kp_f_a3, des_f_a3 = map(list, sift.detectAndCompute(f_a3, None))

kp_f_a4, des_f_a4 = map(list, sift.detectAndCompute(f_a4, None))

numpy.random.seed(10)

numpy.random.shuffle(kp_f_a1)

numpy.random.shuffle(kp_f_a2)

numpy.random.shuffle(kp_f_a3)

numpy.random.shuffle(kp_f_a4)


loc_kp_f_a1 = cv2.KeyPoint_convert(kp_f_a1)
loc_kp_f_a2 = cv2.KeyPoint_convert(kp_f_a2)
loc_kp_f_a3 = cv2.KeyPoint_convert(kp_f_a3)
loc_kp_f_a4 = cv2.KeyPoint_convert(kp_f_a4)

fingerprint_1 = [functools.reduce(lshift, bytes(i)) for i in loc_kp_f_a1]
fingerprint_2 = [functools.reduce(lshift, bytes(i)) for i in loc_kp_f_a2]
fingerprint_3 = [functools.reduce(lshift, bytes(i)) for i in loc_kp_f_a3]
fingerprint_4 = [functools.reduce(lshift, bytes(i)) for i in loc_kp_f_a4]

numpy.random.shuffle(fingerprint_1)

numpy.random.shuffle(fingerprint_2)

numpy.random.shuffle(fingerprint_3)

numpy.random.shuffle(fingerprint_4)

nd_arr = fingerprint_1 + fingerprint_2 + fingerprint_3 + fingerprint_4
# print(nd_arr)

bit_gen = numpy.random.MT19937(numpy.random.SeedSequence(nd_arr))

one_fg = functools.reduce(xor,bit_gen.random_raw(10))

print(one_fg)
# cv2.waitKey()
# cv2.destroyAllWindows()