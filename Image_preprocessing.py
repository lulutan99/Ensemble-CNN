import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import cv2 #OPEN-CV? I hope
import os



train_images = pd.read_pickle('/train_images.pkl')
train_labels = pd.read_csv('/train_labels.csv')
test_images = pd.read_pickle('/test_images.pkl')

train_images.shape
labels = train_labels.loc[:,'Category'].values
labels = labels[np.newaxis]
labels = labels.T

def threshold (image, value):
    for i in range (len(image)):
        for j in range (len(image)):
            if(image[i,j]>=value):
                image[i,j]=255 #white
            else:
                image[i,j]=0   #black
    return image
    

def show_image(i, image1_int, images_original):
    print("\ni=",i)
    image_big = cv2.resize(images_original[i,:,:], (700,700))
    cv2.imshow('image',image_big)
    cv2.waitKey(0)
    image_big = cv2.resize(image1_int[i,:,:], (700,700))
    cv2.imshow('image',image_big)
    cv2.waitKey(0)
    
def process_image(i, image1_int, images_original):
    if(i%100==0):
        print("\ni= ",i)
    
    if(np.mean(image1_int[i,:,:])>=190):
        image1_int[i,:,:] = np.uint8(image1_int[i,:,:]/1.2) #darken (1.15) worked well
           
    blur = cv2.blur(image1_int[i,:,:], (5,5), 20.0)
    image1_int[i,:,:] = cv2.addWeighted(image1_int[i,:,:], 1.5, blur, -0.5, 0, image1_int[i,:,:])
   
    blur = cv2.blur(image1_int[i,:,:], (5,5), 20.0)
    image1_int[i,:,:] = cv2.addWeighted(image1_int[i,:,:], 1.5, blur, -0.5, 0, image1_int[i,:,:])
   
    image1_int[i,:,:] = cv2.blur(image1_int[i,:,:], (5,5)) #blur 

    image1_int[i,:,:] = np.uint8(image1_int[i,:,:]/1.1) #darken

    image1_int[i,:,:] = threshold(image1_int[i,:,:], 185) #threshold -185

    contours, hierarchy = cv2.findContours(image1_int[i,:,:], cv2.RETR_EXTERNAL, 2)
    
    rectParam= np.zeros((len(contours),4))
    j=0
    for c in contours:
        rect = cv2.boundingRect(c)
        x,y,w,h = rect
        if(w>h): #CREATING SQUARE BOUNDING BOXES BASED ON LARGEST DIMENSION
            h =w
        else:
            w=h
            
        rectParam[j,0]=x; rectParam[j,1]=y;rectParam[j,2]=w;rectParam[j,3]=h;
        j+=1
        cv2.rectangle(image1_int[i,:,:],(x,y),(x+w,y+h),(100,100,100),1)
   
    if(len(contours)):#EXTRACTING BOUNDING BOXES
        max_w=rectParam[0,2];
        max_index = 0
        for j in range(1,len(rectParam)):
            if (rectParam[j,2]>max_w):
                max_w = rectParam[j,2]
                max_index = j;
                
        for j in range(0,len(rectParam)):
            if (rectParam[j,2]<max_w):
                ax = rectParam[j,0]; bx = ax + rectParam[j,2]; ay = rectParam[j,1];by = ay + rectParam[j,2];
                ax =np.int(ax);bx =np.int(bx);ay =np.int(ay);by =np.int(by);
                if(ax>=64):ax = 63
                if(bx>=64):bx = 63
                if(ay>=64):ay = 63
                if(by>=64):by = 63 
                wx = np.int(bx-ax)
                wy = np.int(by-ay)
                zero_int = np.uint8(np.zeros((wy,wx)))
                image1_int[i,ay:by,ax:bx]= zero_int
        
        j = max_index #EXTRACTING LARGEST BOUNDING BOX
        offset = 5;
        ax = rectParam[j,0];
        bx = ax + rectParam[j,2]
        ax -= offset
        bx += offset;
        if(ax<0):ax = 0
        if(bx>=64):bx = 63
        ay = rectParam[j,1]
        by = ay + rectParam[j,2]
        ay -= offset;
        by += offset;
        if(ay<0):ay = 0
        if(by>=64):by = 63
        
        ax =np.int(ax);
        bx =np.int(bx);
        ay =np.int(ay);
        by =np.int(by);
        
        image1_int[i,:,:] = cv2.resize(images_original[i,ay:by,ax:bx], (64,64))

    return image1_int[i,:,:]
      
image1 = train_images
image1_int=np.uint8(image1)
images_original = np.uint8(train_images)

test1 = test_images
test1_original=np.uint8(test1)
test1_int = np.uint8(test1)

for i in range(len(image1_int)):
    image1_int[i,:,:] = process_image(i,image1_int, images_original)
    
for i in range(len(test1_int)):
    test1_int[i,:,:] = process_image(i,test1_int, test1_original)
    

for i in range(len(image1_int)):
    if(i<25):
        print("\nlabel =", labels[i])
        show_image(i,image1_int, images_original)
     
cv2.destroyAllWindows()

for i in range(len(test1_int)):
    if(i<25):
        show_image(i,test1_int, test1_original)
    
cv2.destroyAllWindows()
