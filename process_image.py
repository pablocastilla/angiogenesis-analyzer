import cv2
import argparse
import glob

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
	help="path to input dataset of images")
ap.add_argument("-i", "--store", required=False,
	help="path to input dataset of images")
ap.add_argument("-i", "--store", required=False,
	help="path to input dataset of images")
args = vars(ap.parse_args())

# loop over the images
for imagePath in glob.glob(args["images"] + "/*.jpg"):
    img = cv2.imread(image_name+'.jpg',0)

    result = sku.process_frame(img,resize_factor,646.35)

    if(show_image):
        plt.imshow(img, cmap='Greys',  interpolation='nearest')   
        plt.show()

    if(store_processed_image):
        cv2.imwrite('image_processed.jpg',img)