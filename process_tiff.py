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
REAL_DISTANCE_X =  3805708.8
REAL_DISTANCE_Y = 3044954.85

file_names = []

print('starting' )

#create videos and analysis folder
if not os.path.exists('videos'):
    os.mkdir('videos')

if not os.path.exists('analyses'):
    os.mkdir('analyses')

#look for videos (tifs)
for file in os.listdir("./videos"):
    if file.endswith(".tif"):
        file_names.append(file)

#for each tif file create the result folder and csv
for file_name in file_names:
    #if(file_name == '12_24_48_angioE02.tif'):
    #    continue

    print ('processing '+ file_name)

    tiff = tc.opentiff('videos\\'+file_name)

    make_circle = True

    if(tiff.length==0):
        tiffaux=[]
        gray_image = cv2.cvtColor(tiff.retrieve()[1], cv2.COLOR_BGR2GRAY)
        tiffaux.append(gray_image)
        tiff=tiffaux
        make_circle = False

    log_folder_name='analyses\\'+file_name+'_imagelog'
    if not os.path.exists(log_folder_name):
        os.mkdir(log_folder_name)

    with open('analyses\\'+file_name+'_result.csv', 'w') as csvfile:
        filewriter = csv.writer(csvfile, delimiter=';',lineterminator='\n')
        filewriter.writerow(['frame', 'number_of_joints','number_of_meshes','total_meshes_area_pixels','total_meshes_area_nm','average_meshes_area_pixels','average_meshes_area_nm',
                                'number_of_segments','total_segments_length_pixels','total_segments_length_nm'])

        index=1
        #for each frame process it and save the result and the log images
        for img in tiff:       
            start_time = time.time()
            print('start frame:'+str(index) )
                
            result = sku.process_frame(img,RESIZE_FACTOR,REAL_DISTANCE_X,REAL_DISTANCE_Y,make_circle)
                   
            cv2.imwrite(log_folder_name+'\\frame'+str(index)+'_processed.jpg',result[0])                 
            final_image_bit_aux = np.uint8(skimage.img_as_bool(result[10]))*255
            cv2.imwrite(log_folder_name+'\\frame'+str(index)+'_processed_bits.jpg',final_image_bit_aux)

            filewriter.writerow([index, int(result[1]),int(result[2]), int(result[3]),int(result[4]),int(result[5]),int(result[6]),int(result[7]),int(result[8]),int(result[9])])

            index=index+1
            
            elapsed_time = time.time() - start_time
            print ('end frame:'+str(int(elapsed_time))+' seconds')

_ = input("Finish, press any key to exist")