# -*- coding: utf-8 -*-
"""
Created on Thu Oct 10 11:10:27 2019

@author: Ashwin
"""


#Kindly check the git hub link to import the required to input the required images
#git hub link:https://github.com/ASHWIN2605/Histogram_CV
#importing the required libraries
import cv2
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D 
from scipy import ndimage
from matplotlib import rcParams



#Function to read an image and display it 
#Source ref:https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_gui/py_image_display/py_image_display.html#display-image
def Read_image():
    im_rgb1=plt.imread('image.jpg',1)
    plt.figure()
    plt.title('Image')
    plt.imshow(im_rgb1,cmap='Greys_r')
    return im_rgb1


    
#Convert the RGB image to L,A,B format and display it
#Source ref:https://www.learnopencv.com/color-spaces-in-opencv-cpp-python/
def Convert_image(image):
    lab_image = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    #Split the image into L,A,B Components
    l_channel,a_channel,b_channel = cv2.split(lab_image)
    #Plot the L_Component
    plt.figure()
    plt.title('L_Component')
    plt.imshow(l_channel,cmap='gray')
    #Plot A Component
    plt.figure()
    plt.title('A_Component')
    plt.imshow(a_channel,cmap='gray')
    #Plot B component
    plt.figure()
    plt.title('B_Component')
    plt.imshow(b_channel,cmap='gray')
    #return all the components
    return l_channel,a_channel,b_channel


#Create derivatives of X and Y component and display it
#Source ref:https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.gaussian_filter.html
#Souce ref:https://docs.scipy.org/doc/numpy-1.10.1/reference/generated/numpy.diff.html
def Derivative_Component(l_Component):
    #Applying gaussian filter to the image
    filtered_image = ndimage.gaussian_filter(l_Component,sigma=6)
    
    #Taking the derivative across x-axis and display it
    l_dx = np.diff(filtered_image,axis=0)
    plt.figure()
    plt.title('L_dx')
    plt.imshow(l_dx,cmap='gray')
    
    #Taking the derivative along y-axis and display it
    l_dy = np.diff(filtered_image,axis=1)
    plt.figure()
    plt.title('L_dy')
    plt.imshow(l_dy,cmap='gray')
    
#Cal_2D_Histogram for the input image
#Source ref:https://matplotlib.org/3.1.1/gallery/mplot3d/hist3d.html
#Source ref:
def Cal_2D_Histogram(component_1,component_2,image):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    #calculate the 2d histogram and display
    hist, xbins, ybins = np.histogram2d(component_1.ravel(),component_2.ravel(),bins=100)
    plt.figure()
    plt.title('2D Plot')
    plt.imshow(hist,cmap='gray',interpolation = 'nearest',vmin=0,vmax=255)
    
    # Construct arrays for the anchor positions of the 16 bars.
    xpos, ypos = np.meshgrid(xbins[:-1] + 0.25, ybins[:-1] + 0.25, indexing="ij")
    xpos = xpos.ravel()
    ypos = ypos.ravel()
    zpos = 0

    # Construct arrays with the dimensions for the 16 bars.
    dx = dy = 0.5 * np.ones_like(zpos)
    dz = hist.ravel()

    ax.bar3d(xpos, ypos, zpos, dx, dy, dz, color='Blue',zsort='average')
    
    return hist
    

#BackProjection
#Source ref:https://www.youtube.com/watch?v=0rYtZtY5ML4
def Cal_Back_Projection(hist,Color_image):
    target = cv2.imread('target.png')
    target1=cv2.cvtColor(target, cv2.COLOR_BGR2RGB)
    plt.figure()
    plt.title('Target Imgae')
    plt.imshow(target1)
    lab_image = cv2.cvtColor(Color_image, cv2.COLOR_RGB2LAB)
    target_image = cv2.cvtColor(target, cv2.COLOR_BGR2LAB)
    target_hist = cv2.calcHist([target_image], [1, 2], None, [256, 256], [0, 256, 0, 256])
    dst = cv2.calcBackProject([lab_image],[1,2],target_hist,[0,256,0,256],1)
   
    
    # Now convolute with circular disc
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
    dst = cv2.filter2D(dst,-1,kernel)
    
    # threshold and binary AND
    ret,dst = cv2.threshold(dst,0,255,cv2.THRESH_BINARY)
    dst = cv2.merge((dst,dst,dst))
    #doing bit_wise operation on image
    res = cv2.bitwise_and(Color_image,dst)
    plt.figure()
    plt.title('Back_Projected_image')
    res=cv2.cvtColor(res, cv2.COLOR_RGB2GRAY)
    plt.imshow(res,cmap='gray')
    #storing the back-projection as res.jpg
    cv2.imwrite('res.jpg',res)
    
  
#Perform Histogram equalisation
#Source ref:http://programmingcomputervision.com/downloads/ProgrammingComputerVision_CCdraft.pdf,Histogram equalization chapter
def Histogram_equalisation(l_component):
    #calulating the gistogram equalisation
    hist,bins = np.histogram(l_component.flatten(),256)
    cdf = hist.cumsum() # cumulative distribution function
    cdf = 255 * cdf / cdf[-1] # normalize
    # use linear interpolation of cdf to find new pixel values
    im2 = np.interp(l_component.flatten(),bins[:-1],cdf)
    
    #Plotting the histogram equalisation graph
    plt.figure()
    plt.title('Histogram Equalisation' )
    plt.plot(cdf, color = 'b')
    plt.hist(im2.flatten(),256,[0,256], color = 'r')
    plt.xlim([0,256])
    plt.legend(('cdf','histogram'), loc = 'upper left')
    plt.show()
   
    #Plotting the Histogram comparision
    image= im2.reshape(l_component.shape)
    plt.figure()
    rcParams['figure.figsize'] = 11 ,8
    fig,ax = plt.subplots(1,2)
    ax[0].imshow(l_component,cmap="gray")
    ax[0].title.set_text('Original L')
    ax[1].imshow(image,cmap="gray")
    ax[1].title.set_text('Equalized L')

#Function to run all the call       
def Run():
    #1.a
    #To Read an RGB image and dispaly it.
    Color_image=Read_image()
    
    #1.b
    #To Convert the RGB image to L,A,B format
    l_Component,a_Component,b_Component=Convert_image(Color_image)
    
    #1.c
    #Derivative of x and Y component
    Derivative_Component(l_Component)
    
    #1.d
    #To calculate the 2D_Histogram of a and b component
    Histogram = Cal_2D_Histogram(a_Component,b_Component,Color_image)
    
    #1.e
    #BackProjection
    Cal_Back_Projection(Histogram,Color_image)
    
    #1.f
    #Histogram Equalisation
    Histogram_equalisation (l_Component)
    
    
def main():
    #function call to execute the run
    Run()
if __name__ == '__main__':
    main()