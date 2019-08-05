#import the libraries
import cv2


#plt.rcParams['figure.figsize'] = [15, 12]
print(cv2.__version__)

resize_factor=1

image_name = '12H28052019_angioE02-tr'
img = cv2.imread(image_name+'.jpg',0)

sku.process_frame(img,resize_factor,646.35, True,True)

