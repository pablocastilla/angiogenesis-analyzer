#pip install TiffCapture
import tiffcapture as tc
import skeletonize_utils as sku
import cv2 as cv2
import csv
import os
import matplotlib.pyplot as plt
import numpy as np
import skimage as skimage


RESIZE_FACTOR = 5
DISTANCE_PER_PIXEL = 646.35

#ap = argparse.ArgumentParser()
#ap.add_argument("-i", "--tiff", required=True,	help="path to input dataset of images")

#args = vars(ap.parse_args())

file_name = '28052019_angioE02.tif'
tiff = tc.opentiff('videos\\'+file_name)

log_folder_name='analysis\\'+file_name+'_imagelog'
if not os.path.exists(log_folder_name):
    os.mkdir(log_folder_name)
    
with open('analysis\\'+file_name+'_result.csv', 'w') as csvfile:
    filewriter = csv.writer(csvfile, delimiter=';',lineterminator='\n')
    filewriter.writerow(['Frame', 'number_of_joints','number_of_meshes','total_meshes_area','average_meshes_area','number_of_segments','total_segments_length'])

    index=0
    for img in tiff:
        print(index)

        #if(index!=40):
        #    index=index+1
        #    continue

        max_region_factor = 1+(0.5*int(index/10))
        print('frame:'+str(index) + ' factor:'+str(max_region_factor))
        result = sku.process_frame(img,RESIZE_FACTOR,DISTANCE_PER_PIXEL,max_region_factor)
      
        if(True):
            
            cv2.imwrite(log_folder_name+'\\frame'+str(index)+'_processed.jpg',result[0])                 
            final_image_bit_aux = np.uint8(skimage.img_as_bool(result[7]))*255
            cv2.imwrite(log_folder_name+'\\frame'+str(index)+'_processed_bits.jpg',final_image_bit_aux)
            

        #plt.imshow(result[7], cmap='Greys',  interpolation='nearest')   
        #plt.show()

        filewriter.writerow([index, result[1],result[2], result[3],result[4],result[5],result[6]])

        index=index+1