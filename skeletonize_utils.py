import skimage.morphology as m
import numpy as np
import skimage.draw as draw
import cv2 as cv2
import skimage as skimage
import skimage.io as io
import matplotlib.pyplot as plt
from statistics import mean 
import random
import math

#parameters
NUMBER_OF_DILATIONS = 5
MIN_CONTOUR_AREA = 800
MAX_CONTOUR_AREA = 500000
ADAPTATIVE_THRESHOLD_BLOCK_SIZE = 3
ADAPTATIVE_THRESHOLD_C = 1
CANNY_THRESHOLD = 50
SECTIONS_FOR_FINDING_BRIGHTEDGES=7
VARIANCE_IN_COLORS_THRESHOLD = 18

#process a single frame
def process_frame(img, resize_factor, distance_per_pixel):
    image_width=int(img.shape[1]/resize_factor)
    image_height=int(img.shape[0]/resize_factor)
    img = cv2.resize(img,(image_width,image_height))

    final_image_bit,final_contours,final_image_meshes = image_with_sections_contounered_in_cicle(img)

    skeleton = skeletonize(final_image_bit)

    final_joints =  find_joints(skeleton,final_image_meshes)    
    final_distances = find_distances(skeleton,final_joints,final_image_meshes)
    final_meshes = find_meshes(final_contours)

    paint_graph(img,final_distances,distance_per_pixel)
    paint_areas(img,final_contours)

    number_of_joints = len(final_joints)    
    
    number_of_meshes = len(final_meshes)
    total_meshes_area = sum(final_meshes)
    average_meshes_area = mean(final_meshes)
    number_of_segments = len(final_distances)
    total_segments_length = sum([item[3] for item in final_distances]) 

    return (img,number_of_joints,number_of_meshes,total_meshes_area,average_meshes_area,number_of_segments,total_segments_length, final_image_meshes)


    
#takes the image, finds the circle containing the experiment and the contours
def image_with_sections_contounered_in_cicle(img):     

    circle_image_mask = create_internal_circle_mask(img)
    inverted_circle_image_mask = np.logical_not(circle_image_mask)    

    img = cv2.multiply(img, circle_image_mask)

    edges = cv2.Canny(img,CANNY_THRESHOLD,CANNY_THRESHOLD)
    final_sobely = np.uint8(edges)
    final_image = np.zeros([img.shape[0],img.shape[1],1], dtype=np.uint8)
    thresh_gaussian = cv2.adaptiveThreshold(final_sobely,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,ADAPTATIVE_THRESHOLD_BLOCK_SIZE,ADAPTATIVE_THRESHOLD_C)
    (contours,hierarchy) = cv2.findContours(thresh_gaussian,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)

    contours = [c for c in contours if cv2.contourArea(c) > MIN_CONTOUR_AREA ] #remove smalls
       
    final_contours = [contour_validation(img,idx,c,inverted_circle_image_mask) for idx,c in enumerate(contours)]

    inner_holes = [c[0] for c in final_contours if c[1]]
    cv2.drawContours(final_image, inner_holes, -1, (255), 1)
    cv2.fillPoly(final_image,  inner_holes, color=(255))

    vessels = [c[0] for c in final_contours if not c[1]]
    cv2.drawContours(final_image, vessels, -1, (255), 1)
    cv2.fillPoly(final_image,  vessels, color=(255))
    
    final_image_bit = skimage.img_as_bool(cv2.bitwise_not(final_image))    
    final_image_bit = np.bitwise_and(final_image_bit,circle_image_mask)

    return final_image_bit,inner_holes,final_image

#http://opencvpython.blogspot.com/2012/06/contours-3-extraction.html
#false if it is not "microscope background", if so it has more son contours because of the strange forms
def contour_validation(img,idx,contour,inverted_circle_image_mask):

    validated = True
    area = cv2.contourArea(contour)

    if(area < MIN_CONTOUR_AREA):
        validated = False
    
    if(area > MAX_CONTOUR_AREA):
        validated = False
        return (contour,validated)

    #out of the circle
    for p in contour:
        if(inverted_circle_image_mask[p[0][1],p[0][0]]>0):           
            validated = False    
            

    x,y,width,height = cv2.boundingRect(contour)
    colors = []

    for w in range(width):
        for h in range(height):
            y_aux = y+h
            x_aux = x+w
            if cv2.pointPolygonTest(contour,(x_aux,y_aux),False)>0:
                colors.append(img[y_aux,x_aux])

    variance = math.sqrt(np.var(colors))

    print('variance: '+str(variance))
    if(len(colors)>0):
        print('mean: '+str(np.mean(colors)))
    print('area: '+str(area))
    print('x0:'+str(x)+',y0:'+str(y)+',x1:'+str(x+width)+',y1:'+str(y+height))

    if(variance>VARIANCE_IN_COLORS_THRESHOLD):       
        print('discarted!!')
        validated = False  
        
    return (contour,validated)

#finds a bright point in each side of the image in the middle row and creates a circle max with them
def create_internal_circle_mask(img):
    #take pixels in the middle and find brightest in the first quarter
    image_width=int(img.shape[1])
    image_height=int(img.shape[0])
    sections= SECTIONS_FOR_FINDING_BRIGHTEDGES
    middle_height = int(img.shape[0]/2)
    section_width = int(img.shape[1]/sections)
    left_middle_row = img[middle_height:middle_height+1,0:section_width]
    (_, _, _, max_loc_left) = cv2.minMaxLoc(left_middle_row)
    max_loc_left=max_loc_left[0]

    right_middle_row = img[middle_height:middle_height+1,(image_width-section_width):image_width]
    (_, _, _, max_loc_right) = cv2.minMaxLoc(right_middle_row)
    max_loc_right=max_loc_right[0]+section_width*(sections-1)

    radious=int((max_loc_right-max_loc_left)/2)
    center_x=radious+max_loc_left   
    
    circle_image_mask = np.zeros((image_height,image_width ), dtype=np.uint8)

    rr, cc = draw.circle(center_x, middle_height, radious)

    final_circle_rr = []
    final_circle_cc = []

    for i in range(len(rr)):
        if(rr[i]>=0 and rr[i]<image_width and cc[i]>=0 and cc[i]<image_height):
            final_circle_rr.append(rr[i])
            final_circle_cc.append(cc[i])

    circle_image_mask[ final_circle_cc,final_circle_rr] = 1

    return circle_image_mask


#first it makes dilations to remove some "hair" and later skeletonize the image
def skeletonize(img):
    skeleton = img
    for _ in range(NUMBER_OF_DILATIONS):
        skeleton = m.binary_dilation(skeleton)

    skeleton = m.skeletonize(skeleton)

    return skeleton

#find the joints in a skeleton looking for pixels that are sorounded by 3 or more pixels
def find_joints(skeleton_image,final_image_meshes):
    # Find row and column locations that are non-zero
    (rows,cols) = np.nonzero(skeleton_image)

    # Initialize empty list of co-ordinates
    skel_coords = []

    # For each non-zero pixel...
    for (r,c) in zip(rows,cols):

        number_of_neighbours = len(list(find_neighbours(skeleton_image, (r,c),[],final_image_meshes)))

        # If the number of non-zero locations equals 2, add this to 
        # our list of co-ordinates
        if number_of_neighbours > 2:
            skel_coords.append((r,c,number_of_neighbours))
    
    items_to_remove=[]

    for i in range(len(skel_coords)):
        for j in range(len(skel_coords)):
            if(i<j and euclidean_distance(skel_coords[i],skel_coords[j])<=pow(2,0.5) and skel_coords[i][2]>=skel_coords[j][2]):
                items_to_remove.append(skel_coords[j])    


    return list(((x[0],x[1]) for x in skel_coords if x not in items_to_remove))
  
#find the distantes of the joints, return a matrix of [jointA, jointB,[pixelesbetweenthem], distance in pixel]
def find_distances(skeleton, joints,final_image_meshes):    
    distances = []
           
    for j in joints:              
        #look for next points
        next_joints_points = find_neighbours(skeleton, j, [],final_image_meshes)
        
        for first_point_in_branch in next_joints_points:
            this_branch_points=[j]
            next_point = first_point_in_branch
            final_point = None
            joint_reached = False
            distance_in_pixels = 0

            while(len(find_neighbours(skeleton,next_point,this_branch_points,final_image_meshes))>0 and not joint_reached):
                old_next_point = next_point
                next_point = find_neighbours(skeleton,next_point,this_branch_points,final_image_meshes)[0]
                this_branch_points.append(old_next_point)               
                    
                if next_point in joints:
                    joint_reached = True
                
                distance_in_pixels = distance_in_pixels + euclidean_distance(old_next_point,next_point)
                final_point = next_point
               

            #not add if the oposite relation already exists       
            if(final_point is not None and not any(x[1] == j and x[0]==final_point for x in distances)):
                distances.append([j,final_point,this_branch_points,distance_in_pixels])                   
                    
    return distances

#find the neighbours of a pixel, neartests (not diagonal) have priority
def find_neighbours(skeleton, point, excluded_points,image_meshes):
    neighbours = []
    point_y,point_x=point
   
    avoid_top = False
    avoid_left = False
    avoid_right = False
    avoid_botton = False
    
    #find top 
    offset_y=-1
    offset_x=0
    if(evaluate_neighbour(skeleton, point, offset_y, offset_x)):
        neighbours.append((point_y+offset_y,point_x+offset_x))
        avoid_top = True
        
    #find left
    offset_y=0
    offset_x=-1
    if(evaluate_neighbour(skeleton, point, offset_y, offset_x)):
        neighbours.append((point_y+offset_y,point_x+offset_x))
        avoid_left = True
        
    #find right
    offset_y=0
    offset_x=1
    if(evaluate_neighbour(skeleton, point, offset_y, offset_x)):
        neighbours.append((point_y+offset_y,point_x+offset_x))
        avoid_right = True
                   
    #find botton
    offset_y=1
    offset_x=0
    if(evaluate_neighbour(skeleton, point, offset_y, offset_x)):
        neighbours.append((point_y+offset_y,point_x+offset_x))
        avoid_botton = True
        
    #find botton left
    offset_y=1
    offset_x=-1
    if(evaluate_neighbour(skeleton, point, offset_y, offset_x) and not avoid_botton and not avoid_left):
        neighbours.append((point_y+offset_y,point_x+offset_x))
    
    #find botton right
    offset_y=1
    offset_x=1
    if(evaluate_neighbour(skeleton, point, offset_y, offset_x) and not avoid_botton and not avoid_right):
        neighbours.append((point_y+offset_y,point_x+offset_x))

    #find top right
    offset_y=-1
    offset_x=1
    if(evaluate_neighbour(skeleton, point, offset_y, offset_x) and not avoid_top and not avoid_right):
        neighbours.append((point_y+offset_y,point_x+offset_x))
   
        
    #find top left
    offset_y=-1
    offset_x=-1
    if(evaluate_neighbour(skeleton, point, offset_y, offset_x) and not avoid_top and not avoid_left):
        neighbours.append((point_y+offset_y,point_x+offset_x))
        
    return list((x for x in neighbours if x not in excluded_points and image_meshes[x[0],x[1]]==0))

def evaluate_neighbour(skeleton, point, offset_y,offset_x):
    image_width=int(skeleton.shape[1])
    image_height=int(skeleton.shape[0])
    point_y,point_x=point    
    
    if((0 <= point_y+offset_y < image_height) and (0 <= point_x+offset_x < image_width) and skeleton[point_y+offset_y,point_x+offset_x]):
        return True
    else:
        return False



def find_meshes(final_contours):
    meshes = []

    for c in final_contours:             
        area = cv2.contourArea(c)
        meshes.append(area)

    return meshes


def paint_graph(img,graph,distance_per_pixel):
    for line in graph:
        #paint the lenght in the middle        
        for p in line[2]:
            img[p[0],p[1]] = 0

       
        #middle_y = int((line[0][0]+line[1][0])/2)
        #middle_x = int((line[0][1]+line[1][1])/2) -100 #trying to move text size to the left in order to center it
        #cv2.putText(img, "{:.1f}".format(len(line[2])),(int(middle_x), int(middle_y)), cv2.FONT_HERSHEY_SIMPLEX,1, (0, 255, 255), 3)

        #draw a circle in each node
        cv2.circle(img, (line[0][1],line[0][0]), 5, (0, 0, 0))
    

def paint_areas(img,contours):    
    
    for c in contours:              
        br = cv2.boundingRect(c)
        
        xMin=0
        yMin=0
        xMax=img.shape[0]
        yMax=img.shape[1]
    
        #if( br[0] <= xMin or br[1] <= yMin or (br[0]+br[2]) >= xMax or  (br[1]+br[3]) >= yMax):
        #        continue
            
        box = cv2.minAreaRect(c)
        box = cv2.boxPoints(box)
        box = np.array(box, dtype="int")    

        random_color = random.randint(220, 255)
        
        color = (random_color,random_color,random_color)

        cv2.drawContours(img, c, -1, color, 3)
    
        m = cv2.moments(c)
    
        divisor=m['m00']
        if(m['m00']==0):
            divisor=1
        cx = int(m['m10']/divisor)
        cy = int(m['m01']/divisor)    
           
        cv2.putText(img, "{:.1f}".format(cv2.contourArea(c)),(cx,cy ), cv2.FONT_HERSHEY_SIMPLEX,1, (255,255,255), 1)

def euclidean_distance(coordinate1, coordinate2):
    return pow(pow(coordinate1[0] - coordinate2[0], 2) + pow(coordinate1[1] - coordinate2[1], 2), .5)
        
