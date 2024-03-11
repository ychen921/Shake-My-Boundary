import numpy as np
import math
import cv2
from skimage.transform import rotate
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


def gaussian_kernel(size, sigma):
    """ Create a Gaussian kernel """
    
    inter = np.linspace(-(size - 1) / 2., (size - 1) / 2., size)
    xx, yy = np.meshgrid(inter, inter)
    gaussian = np.exp(-0.5 * (np.square(xx) + np.square(yy)) / np.square(sigma)) * (0.5 * math.pi * sigma**2)
    
    # Normalization
    gaussian = gaussian / np.sum(gaussian)
    
    return gaussian


def DoG_show(DoG_filters):
    """ Plot DoG filter bank """
    filters_num = len(DoG_filters)
    
    plt.subplots(2, 16, figsize=(16,3))
    
    for i in range(filters_num):
        plt.subplot(2, 16, i+1)
        plt.axis('off')
        plt.imshow(DoG_filters[i],cmap='gray')
        
    plt.savefig('../Results/DoG.png', bbox_inches='tight')
    plt.show()
    plt.close()
    
    
def create_DoG_filters(scales=[20, 46], orientations=16, size=49):
    """Create DoG filter banks"""
    filter_bank = []
    
    # Sobel filter
    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    sobel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    
    # Slice 360 degree into 15 sections
    degrees = np.linspace(0, 360, orientations, endpoint=False)    
    
    for sigma in scales:
        # Create gaussian image
        gaussian = gaussian_kernel(size, sigma)
        
        # Convolution (Sobel operator)
        img_x = cv2.filter2D(src=gaussian, ddepth=-1, kernel=sobel_x)
        img_y = cv2.filter2D(src=gaussian, ddepth=-1, kernel=sobel_y)
        
        # Compute the magnitude
        magnitude = img_x + img_y
        
        for angle in degrees:
            DoG_f = rotate(magnitude, angle)
            filter_bank.append(DoG_f)
            
    return filter_bank


def gauss_1d(sigma, mean, x, ord):
    # 1-dim gaussian"""
    x = np.array(x) - mean
    
    gs = (1/np.sqrt(2*np.pi*sigma**2)) * np.exp(-x**2 / (2*sigma**2))
    
    # Compute derivative of 1-dim gaussian
    if ord == 1:
        gs_dot = (-x/sigma**2) * gs
        return gs_dot
    
    # Compute 2nd derivative of 1-dim gaussian
    elif ord == 2:
        gs_ddot = ((x**2/sigma**4) - (1/sigma)) * gs
        return gs_ddot
    
    else:
        return gs
        
def gauss_derivative_1d(scale, ord_x, ord_y, pts, size):
    
    gauss_x = gauss_1d(3*scale, 0, pts[0], ord_x)
    gauss_y = gauss_1d(scale, 0, pts[1], ord_y)
    
    gs_filter = gauss_x * gauss_y
    gs_filter = gs_filter.reshape(size, size)
    
    return gs_filter

def LoG(size, sigma):
    inter = np.linspace(-(size - 1) / 2., (size - 1) / 2., size)
    xx, yy = np.meshgrid(inter, inter)
    g = np.exp(-0.5 * (xx**2 + yy**2) / np.square(sigma)) * (0.5 * math.pi * sigma**2)
    log = (((xx**2+yy**2)/sigma**4) - (2/sigma**2))*g
    
    return log


# The LM code was referred to https://www.robots.ox.ac.uk/~vgg/research/texclass/filters.html
def create_LM_filters(img_size=49):
    
    # number of 2-dim Gaussian Filters, Laplacian of 
    # Gaussian (LOG) filters, and orientations
    n_gauss, n_LoG, n_orient = 4, 8, 6
    
    # Scales of LM small and large 
    scale_LMS  = np.array([1, np.sqrt(2), np.sqrt(2)**2])
    scale_LML  = np.array([np.sqrt(2), np.sqrt(2)**2, np.sqrt(2)**3, np.sqrt(2)**4])
    
    # number of gaussian first derivative filters
    n_gauss_fir_d = n_orient * len(scale_LMS)
    
    # number of gaussian second derivatives filters
    n_gauss_sec_d = n_orient * len(scale_LMS)
    
    # number of total filters 
    num_filters = n_gauss_fir_d + n_gauss_sec_d + n_gauss + n_LoG
    
    LM_filter_bank = np.zeros([img_size, img_size, num_filters])
    
    half_size  = (img_size-1) / 2
    
    [x, y] = np.meshgrid([np.arange(-half_size, half_size+1)], [np.arange(-half_size, half_size+1)])
    org_pts = np.asarray([x.flatten(), y.flatten()])
    
    angles = np.linspace(np.pi, 0, n_orient, endpoint=False)
    
    # First, generate first and second order derivatives of Gaussian
    # filters at 6 orientations and 3 scales. Total 36 filters
    
    cnt = 0 # counter for storing filter image in the correct position
    for sigma in scale_LMS:
        for angle in angles:
            
            rot_matrix = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
            rot_pts = rot_matrix @ org_pts
            
            # Generate and store first derivative gaussian filter
            LM_filter_bank[:, :, cnt] = gauss_derivative_1d(sigma, 0, 1, rot_pts, img_size)
            
            # Generate and store second derivatives gaussian filter
            LM_filter_bank[:,:, cnt+n_gauss_fir_d] = gauss_derivative_1d(sigma, 0, 2, rot_pts, img_size)
            cnt+=1
    
    cnt = n_gauss_fir_d + n_gauss_sec_d
    for sigma in scale_LML:
        LM_filter_bank[:, :, cnt] = LoG(img_size, sigma)
        cnt+=1
        
    for sigma in scale_LML:
        LM_filter_bank[:, :, cnt] = LoG(img_size, 3*sigma)
        cnt+=1
    
    for sigma in scale_LML:
        LM_filter_bank[:, :, cnt] = gaussian_kernel(img_size, sigma)
        cnt+=1
        
    return LM_filter_bank


def LM_show(LM_filters):
    """ Plot LM filter bank """
    filters_num = LM_filters.shape[2]
    plt.subplots(4, 12, figsize=(16,6))
    
    for i in range(filters_num):
        plt.subplot(4, 12, i+1)
        plt.axis('off')
        plt.imshow(LM_filters[:, :, i], cmap='gray')
    plt.savefig('../Results/LM.png', bbox_inches='tight')
    plt.show()
    plt.close()
    
    
# The make_gb code was referred to https://en.wikipedia.org/wiki/Gabor_filter
def make_gb(sigma, theta, Lambda=np.pi/3, psi=0.2, gamma=1):
    
    size = 43
    inter = np.linspace(-(size - 1) / 2., (size - 1) / 2., size)
    xx, yy = np.meshgrid(inter, inter)
    
    x_dot = xx*np.cos(theta) + yy*np.sin(theta)
    y_dot = -xx*np.sin(theta) + yy*np.cos(theta)
    
    gb = np.exp(-(x_dot**2+gamma**2*y_dot**2)/(2*(sigma**2))) * np.cos((2*np.pi*x_dot/Lambda)+psi)
    
    return gb

def create_gabor_filters(orientation=8):
    
    gb_filters = []
    scales = [3, 5, 7, 9, 11]
    theta = 0.2
    
    degrees = np.linspace(0, 180, orientation, endpoint=False)
    
    for sigma in scales:
        gb = make_gb(sigma=sigma, theta=theta, Lambda=np.pi/3, psi=0.5, gamma=1)
        for angle in degrees:
            gb_f = rotate(gb, angle)
            gb_filters.append(gb_f)
            
    return gb_filters

def Gb_show(gb_filters):
    plt.subplots(5, 12, figsize=(16,16))
    
    for i, gb in enumerate(gb_filters):
        plt.subplot(5, 8, i+1)
        plt.axis('off')
        plt.imshow(gb, cmap='gray')
        
    plt.savefig('../Results/Gabor.png', bbox_inches='tight')
    plt.show()
    plt.close()
    
def save_as_list(filters):
    new_list = []
    filters_num = filters.shape[2]
    
    for i in range(filters_num):
        new_list.append(filters[:, :, i])
        
    return new_list


def gen_HD(radius):
    
    X, Y = np.meshgrid(np.arange(-radius, radius + 1), np.arange(-radius, radius + 1))
    # Create a boolean mask for the half-disk
    mask = np.sqrt(X**2 + Y**2) <= radius
    half_disk_mask = mask & (X<=0)
    
    return half_disk_mask
    
    
def create_HD_masks(scales=[5, 15, 25], orientation=8):
    
    HD_filters = []
    angles = np.linspace(0, 180, orientation, endpoint=False)
    
    for sigma in scales:
        
        HD = gen_HD(sigma)
        HD_flip = np.flip(HD, axis=1)
        
        for angle in angles:
            
            HD_ = rotate(HD, angle)
            HD_flip_ = rotate(HD_flip, angle)

            HD_filters.append(HD_.astype(np.int8))
            HD_filters.append(HD_flip_.astype(np.int8))
            
    return HD_filters

def HD_show(HD_masks):
    plt.subplots(6, 8, figsize=(14,14))
    
    for i, gb in enumerate(HD_masks):
        plt.subplot(6, 8, i+1)
        plt.axis('off')
        plt.imshow(gb, cmap='gray')
        
    plt.savefig('../Results/HDMasks.png', bbox_inches='tight')
    plt.show()
    plt.close()
    
def texton(img, filters):
    
    tex_map = np.array(img)
    
    for filter in filters:
        img_f = cv2.filter2D(src=img, ddepth=-1, kernel=filter)
        tex_map = np.dstack((tex_map, img_f))
        
    return tex_map
    
def create_texton_map(img, DoG, LM, Gabor):
    
    img_w, img_h, _ = img.shape
    k_clusters = 64
    
    tex_DoG = texton(img, DoG)
    tex_LM = texton(img, LM)
    tex_Gb = texton(img, Gabor)
    
    texton_map = np.dstack((tex_DoG[:,:,1:],tex_LM[:,:,1:],tex_Gb[:,:,1:]))
    t_w, t_h, _= texton_map.shape
    
    texton_map_ = texton_map.reshape((img_w*img_h), texton_map.shape[2])
    
    texton_clusters = KMeans(n_clusters=k_clusters).fit(texton_map_)
    segmap = texton_clusters.labels_.reshape(t_w, t_h)
    
    return segmap

def create_brightness_map(img):
    
    k_clusters = 16
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_flatten = gray_image.reshape((-1, 1))
    
    brightness_clusters = KMeans(n_clusters=k_clusters).fit(img_flatten)
    segmap = brightness_clusters.labels_.reshape(gray_image.shape)
    
    return segmap

def create_color_map(img):

    k_clusters = 16
    w, h, d = img.shape
    RGB_image = img.reshape((-1, d))

    img_clusters = KMeans(n_clusters=k_clusters).fit(RGB_image)
    segmap = img_clusters.labels_.reshape(w,h)

    return segmap

def chi_gradient(img, bins, HD_masks):
    
    mask_num = int(len(HD_masks)/2)
    g_img = img
    
    for i in range(mask_num):
        chi_sqr_dist = img*0
        left_mask = HD_masks[2*i]
        right_mask = HD_masks[2*i+1]

        for bin in range(bins):
            tmp = np.where(img == bin, 1, 0).astype(np.float64)
            g_i = cv2.filter2D(src=tmp, ddepth=-1, kernel=left_mask.astype(np.float64))
            h_i = cv2.filter2D(src=tmp, ddepth=-1, kernel=right_mask.astype(np.float64))
            
            chi_sqr_dist = chi_sqr_dist + ((g_i-h_i)**2 /(g_i+h_i+np.exp(-7)))
    
        chi_sqr_dist = 0.5*chi_sqr_dist
        g_img = np.dstack((g_img, chi_sqr_dist))
        
    mean = np.mean(g_img, axis=2)
    
    return mean