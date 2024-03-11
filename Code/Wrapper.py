"""
CMSC733 Spring 2019: Classical and Deep Learning Approaches for
Geometric Computer Vision
Homework 0: Alohomora: Phase 1 Starter Code


Author(s): 
Yi-Chung Chen (ychen921@umd.edu)
MEng in Robotics,
University of Maryland, College Park
"""

# Code starts here:

import numpy as np
import cv2
import os
from tqdm import tqdm
from utils import*


def main():
    
	'''Generate Difference of Gaussian Filter Bank: (DoG)
	Display all the filters in this filter bank and save image as DoG.png, 
	use command "cv2.imwrite(...)"'''
	DoG_filters = create_DoG_filters(scales=[6, 10], orientations=16, size=59)
	DoG_show(DoG_filters)

	"""Generate Leung-Malik Filter Bank: (LM)
	Display all the filters in this filter bank and save image as LM.png,
	use command "cv2.imwrite(...)"""
	LM_filters = create_LM_filters(img_size=49)
	LM_show(LM_filters)

	"""Generate Gabor Filter Bank: (Gabor)
	Display all the filters in this filter bank and save image as Gabor.png,
	use command "cv2.imwrite(...)"""
	Gabor_filters = create_gabor_filters()
	Gb_show(Gabor_filters)

	LM_filters = save_as_list(LM_filters)
 		
	"""
	Generate Half-disk masks
	Display all the Half-disk masks and save image as HDMasks.png,
	use command "cv2.imwrite(...)"
	"""
	HD_masks = create_HD_masks()
	HD_show(HD_masks)
 
	# Images path
	img_path = r'../BSDS500/Images/'
	canny_path = r'../BSDS500/CannyBaseline/'
	sobel_path = r'../BSDS500/SobelBaseline/'
	save_map_path = r'../Results/map/'
	
	if not os.path.exists(save_map_path):
		os.mkdir(save_map_path)
  
	for file_name in tqdm(os.listdir(img_path)):
		if os.path.isfile(os.path.join(img_path, file_name)):
			name = file_name.split('.')[0]
   
			if not os.path.exists(save_map_path+name+'/'):
				os.mkdir(save_map_path+name+'/')
    
			img = cv2.imread(img_path+file_name)

			"""
			Generate Texton Map
			Filter image using oriented gaussian filter bank
			"""
			texton_map = create_texton_map(img, DoG_filters, LM_filters, Gabor_filters)
		
			"""
			Generate texture ID's using K-means clustering
			Display texton map and save image as TextonMap_ImageName.png,
			use command "cv2.imwrite('...)"
			"""
			plt.imshow(texton_map, cmap='hsv')
			plt.axis('off')
			plt.savefig('../Results/map/'+str(name)+'/TextonMap_'+str(name)+'.png', bbox_inches='tight')
			
			"""
			Generate Texton Gradient (Tg)
			Perform Chi-square calculation on Texton Map
			Display Tg and save image as Tg_ImageName.png,
			use command "cv2.imwrite(...)"
			"""
			T_g = chi_gradient(texton_map, 64, HD_masks)
			plt.imshow(T_g, cmap='hsv')
			plt.axis('off')
			plt.savefig('../Results/map/'+str(name)+'/Tg_'+str(name)+'.png', bbox_inches='tight')
   
   
			"""
			Generate Brightness Map
			Perform brightness binning 
			"""
			brightness_map = create_brightness_map(img)
			plt.imshow(brightness_map, cmap='gray')
			plt.axis('off')
			plt.savefig('../Results/map/'+str(name)+'/BrightnessMap_'+str(name)+'.png', bbox_inches='tight')
			
			"""
			Generate Brightness Gradient (Bg)
			Perform Chi-square calculation on Brightness Map
			Display Bg and save image as Bg_ImageName.png,
			use command "cv2.imwrite(...)"
			"""
			B_g = chi_gradient(brightness_map, 16, HD_masks)
			plt.imshow(B_g, cmap='gray')
			plt.axis('off')
			plt.savefig('../Results/map/'+str(name)+'/Bg_'+str(name)+'.png', bbox_inches='tight')
   
			"""
			Generate Color Map
			Perform color binning or clustering
			"""
			color_map = create_color_map(img)
			plt.imshow(color_map, cmap='hsv')
			plt.axis('off')
			plt.savefig('../Results/map/'+str(name)+'/ColorMap_'+str(name)+'.png', bbox_inches='tight')

			
			"""
			Generate Color Gradient (Cg)
			Perform Chi-square calculation on Color Map
			Display Cg and save image as Cg_ImageName.png,
			use command "cv2.imwrite(...)"
			"""
			C_g = chi_gradient(color_map, 16, HD_masks)
			plt.imshow(C_g, cmap='hsv')
			plt.axis('off')
			plt.savefig('../Results/map/'+str(name)+'/Cg_'+str(name)+'.png', bbox_inches='tight')

			"""
			Read Sobel Baseline
			use command "cv2.imread(...)"
			"""
			sobel_base = cv2.imread(sobel_path+str(name)+'.png')
			sobel_base = cv2.cvtColor(sobel_base, cv2.COLOR_BGR2GRAY)

			"""
			Read Canny Baseline
			use command "cv2.imread(...)"
			"""
			canny_base = cv2.imread(canny_path+str(name)+'.png')
			canny_base = cv2.cvtColor(canny_base, cv2.COLOR_BGR2GRAY)
   
			"""
			Combine responses to get pb-lite output
			Display PbLite and save image as PbLite_ImageName.png
			use command "cv2.imwrite(...)"
			"""
			pb_edges = np.multiply(((T_g+C_g+B_g)/3), (0.1*sobel_base+0.9*canny_base))
			plt.imshow(pb_edges,cmap='gray')
			plt.axis('off')
			plt.savefig('../Results/map/'+str(name)+'/pb_'+str(name)+'.png', bbox_inches='tight')
			print('pb_edges:', str(name))

    
if __name__ == '__main__':
    main()
 


