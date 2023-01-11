import cv2
import numpy as np
import random
def trans_gray(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
def salt_pepper_noise(img):
    row , col = img.shape
    white_noise = black_noise = int(row * col * 0.2 * 0.5) #20%的雜訊
    while(white_noise > 0 or black_noise > 0):
        prob = random.random()
        noise_row = random.randint(0,row-1)
        noise_col = random.randint(0,col-1)
        if prob >0.5:
            if black_noise>0:
                if img[noise_row,noise_col] == 0 or img[noise_row,noise_col]==255 : continue
                img[noise_row,noise_col] = 0
                black_noise -=1
        else:
            if white_noise>0:
                if img[noise_row,noise_col] == 255 or img[noise_row,noise_col] == 0: continue
                img[noise_row,noise_col] = 255
                white_noise -=1
    return img

def Adaptive_Median(img , window_size , row , col):
    Smax = 7
    window = img[row:row+window_size , col:col+window_size]
    sorted_pixel = np.sort(window.flatten())
    zxy , zmin , zmed , zmax = img[row , col] , sorted_pixel[0] , sorted_pixel[len(sorted_pixel)//2] , sorted_pixel[-1]
    if zmin < zmed < zmax:
        if zmin < zxy < zmax:
            return zxy
        else:
            return zmed
    else:
        window_size += 2
        if window_size <= Smax :
            return Adaptive_Median(img, window_size, row, col)
        else:
            return zmed
def Adaptive_Median_Filter(img):
    window_size = 3
    output = np.zeros_like(img.copy())
    padded_image = np.pad(img.copy() , window_size//2 , mode = 'constant')
    for row in range(img.shape[0]):
        for col in range(img.shape[1]):
            output[row,col] = Adaptive_Median(padded_image , window_size , row , col)
    return output

def calculate_noise(img):
    white , black = 0 , 0
    for i in range(img.shape[0]):
        for j in  range(img.shape[1]):
            if img[i,j]==0:
                black+=1
            if img[i,j]==255:
                white+=1
    return white , black

img = cv2.imread('Lenna.jpg')
img_gray = trans_gray(img.copy())
org_white , org_black = calculate_noise(img_gray.copy())
print(f'原本的black = {org_black}\n原本的white = {org_white}')
img_noise = salt_pepper_noise(img_gray.copy())
noise_white , noise_black = calculate_noise(img_noise.copy())
print(f'增加雜訊後的black = {noise_black} \n增加雜訊後的white = {noise_white}')
img_adaptive = Adaptive_Median_Filter(img_noise.copy())
denoise_white , denoise_black = calculate_noise(img_adaptive.copy())
print(f'影像還原後的black = {denoise_black}\n影像還原後的white = {denoise_white}')
cv2.imshow('gray',img_gray)
cv2.imshow('salt_img',img_noise)
cv2.imshow('img_adaptive',img_adaptive)
cv2.waitKey(0)