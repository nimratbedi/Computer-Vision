import cv2
import numpy as np
import sys

# read arguments
if(len(sys.argv) != 3) :
    print(sys.argv[0], ": takes 2 arguments. Not ", len(sys.argv)-1)
    print("Expecting arguments: ImageIn ImageOut.")
    print("Example:", sys.argv[0], "fruits.jpg out.png")
    sys.exit()

name_input = sys.argv[1]
name_output = sys.argv[2]

# read image
inputImage = cv2.imread(name_input, cv2.IMREAD_COLOR)
if(inputImage is None) :
    print(sys.argv[0], ": Failed to read image from: ", name_input)
    sys.exit()
cv2.imshow("input image: " + name_input, inputImage)

rows, cols, bands = inputImage.shape
if(bands != 3) :
    print("Input image is not a standard color image:", inputImage)
    sys.exit()

outputImage = np.zeros([rows, cols, bands],dtype=np.uint8)


luvimg = cv2.cvtColor(inputImage, cv2.COLOR_BGR2LUV)
print(np.amax(luvimg[: , : , 0]))
print(np.amin(luvimg[: , : , 0]))
print(luvimg[: , : , 0])
eq_L = cv2.equalizeHist(luvimg[:, :, 0])
print(eq_L[0, 0])
luvimg[:, :, 0] = eq_L
print(luvimg[: , : , 0])
print(np.amax(luvimg[: , : , 0]))
print(np.amin(luvimg[: , : , 0]))
new_img = cv2.cvtColor(luvimg, cv2.COLOR_LUV2BGR)

cv2.namedWindow('color_original', cv2.WINDOW_AUTOSIZE)
cv2.imshow('color_original', inputImage)
cv2.namedWindow('color result', cv2.WINDOW_AUTOSIZE)
cv2.imshow('color result', new_img)

# wait for key to exit

cv2.waitKey(0)
cv2.destroyAllWindows()