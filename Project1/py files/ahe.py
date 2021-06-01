import cv2
import numpy as np
import sys
import math

# read arguments
if(len(sys.argv) != 4) :
    print(sys.argv[0], ": takes 3 arguments. Not ", len(sys.argv)-1)
    print("Expecting arguments: w ImageIn ImageOut.")
    print("Example:", sys.argv[0], "10 fruits.jpg out.png")
    sys.exit()

w = int(sys.argv[1])
name_input = sys.argv[2]
name_output = sys.argv[3]


# read image
inputImage = cv2.imread(name_input, cv2.IMREAD_COLOR)
if(inputImage is None) :
    print(sys.argv[0], ": Failed to read image from: ", name_input)
    sys.exit()
cv2.imshow("input image: " + name_input, inputImage)

luvimg = cv2.cvtColor(inputImage, cv2.COLOR_BGR2LUV)

rows, cols, bands = inputImage.shape
if(bands != 3) :
    print("Input image is not a standard color image:", inputImage)
    sys.exit()

outputImage = np.zeros([rows, cols, bands],dtype=np.uint8)

# output pixel is computed from a window of size 2w+1 x 2w+1 in the input image

for i in range(w, rows-w) :
    for j in range(w, cols-w) :
        eq_L = cv2.equalizeHist ( luvimg [i-w:i+w, j-w:j+w,0]) 
        l, u, v = inputImage[i, j]
        m,n = eq_L.shape
        l = eq_L[math.floor((m+1) / 2)][math.floor((n+1) / 2)]
        outputImage[i, j] = [l, u, v]

new_img = cv2.cvtColor(outputImage , cv2.COLOR_LUV2BGR)

cv2.namedWindow('color_original', cv2.WINDOW_AUTOSIZE)
cv2.imshow('color_original', inputImage)
cv2.namedWindow('color result', cv2.WINDOW_AUTOSIZE)
cv2.imshow('color result', new_img)



# wait for key to exit
cv2.waitKey(0)
cv2.destroyAllWindows()
