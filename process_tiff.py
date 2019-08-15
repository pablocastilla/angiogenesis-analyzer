import tiffcapture as tc
import skeletonize_utils as sku
import cv2 as cv2
import csv
import os
import numpy as np
import skimage as skimage
import time
from PIL import Image
from PIL import BmpImagePlugin,GifImagePlugin,Jpeg2KImagePlugin,JpegImagePlugin,PngImagePlugin,TiffImagePlugin,WmfImagePlugin # added this line for pyinstaller

RESIZE_FACTOR = 5
DISTANCE_PER_PIXEL = 646.35

file_names = []

for file in os.listdir("./videos"):
    if file.endswith(".tif"):
        file_names.append(file)

for file_name in file_names:

    tiff = tc.opentiff('videos\\'+file_name)

    log_folder_name='analysis\\'+file_name+'_imagelog'
    if not os.path.exists(log_folder_name):
        os.mkdir(log_folder_name)

    with open('analysis\\'+file_name+'_result.csv', 'w') as csvfile:
        filewriter = csv.writer(csvfile, delimiter=';',lineterminator='\n')
        filewriter.writerow(['Frame', 'number_of_joints','number_of_meshes','total_meshes_area','average_meshes_area','number_of_segments','total_segments_length'])

        index=0

        for img in tiff:       
            start_time = time.time()
            print('start frame:'+str(index) )
            

            #if(index%10!=0):
            #    index=index+1
            #    continue       

            result = sku.process_frame(img,RESIZE_FACTOR,DISTANCE_PER_PIXEL)
                   
            cv2.imwrite(log_folder_name+'\\frame'+str(index)+'_processed.jpg',result[0])                 
            final_image_bit_aux = np.uint8(skimage.img_as_bool(result[7]))*255
            cv2.imwrite(log_folder_name+'\\frame'+str(index)+'_processed_bits.jpg',final_image_bit_aux)

            filewriter.writerow([index, result[1],result[2], result[3],result[4],result[5],result[6]])

            index=index+1
            
            elapsed_time = time.time() - start_time
            print ('end frame:'+str(int(elapsed_time))+' seconds')