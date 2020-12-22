#!/usr/bin/env python
# coding: utf-8

# In[60]:


import matplotlib.pyplot as plt
import pylab
import cv2
import numpy as np

img = plt.imread("2007_000256.jpg")             #read image



fil_smooth = np.ones((9,9),np.float32)/(9*9)    #smooth filter

fil_original = np.array([[ 1,1,-1],             #without dilated         
                        [ 0,0,0],
                        [ -1,-1,-1]])

fil_Dilated = np.array([[ 1,0,0,0,1,0,0,0,1],   #dilated with space 3
                        [ 0,0,0,0,0,0,0,0,0],     
                        [ 0,0,0,0,0,0,0,0,0], 
                        [ 0,0,0,0,0,0,0,0,0], 
                        [ 0,0,0,0,0,0,0,0,0],
                        [ 0,0,0,0,0,0,0,0,0], 
                        [ 0,0,0,0,0,0,0,0,0], 
                        [ 0,0,0,0,0,0,0,0,0], 
                        [ -1,0,0,0,-1,0,0,0,-1]])
plt.imshow(img)                                    
pylab.show()
plt.savefig("original_image")
res_original = cv2.filter2D(img,-1,fil_original)

plt.title('nodilated image')
plt.imshow(res_original) 
plt.savefig("nodilated_image")        
pylab.show()

#dircvtly apply dilated convolution
res_dir_dilated = cv2.filter2D(img,-1,fil_Dilated)                     
plt.title('dir_dilated')
plt.imshow(res_dir_dilated)                                     
plt.savefig("dir_dilated_image")
pylab.show()

#apply a smooth convolution before apply dilated convolution
res_smooth = cv2.filter2D(img,-1,fil_smooth )
res_smooth_dilated = cv2.filter2D(res_smooth,-1,fil_Dilated)
plt.title('smooth_dilated')
plt.imshow(res_smooth_dilated)                                     
plt.savefig("smooth_dilated_image")
pylab.show()



# In[ ]:





# In[ ]:




