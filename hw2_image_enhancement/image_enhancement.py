import cv2
import numpy as np

def trans_gray(img):
    return cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

def output_img(img,conv):
    result = np.zeros((img.shape[0],img.shape[1]),dtype=conv.dtype)
    for i in range(conv.shape[0]):
        for j in range(conv.shape[1]):
            result[i][j] = int(conv[i][j]) if int(conv[i][j])<256 else 255
    return result

def Convolution(img , kernel):
    print(img.shape)
    img_height = img.shape[0]
    img_width = img.shape[1]
    kernel_rows = kernel.shape[0]
    kernel_cols = kernel.shape[1]
    
    kernel_center_x = kernel_rows // 2
    kernel_center_y = kernel_cols // 2
    result = np.zeros((img_height, img_width), dtype=img.dtype)

    for i_h in range(kernel_center_x,img_height-kernel_center_x):
        for i_w in range(kernel_center_y,img_width-kernel_center_y):
            conv_sum = 0
            for k_h in range(kernel_rows):
                for k_w in range(kernel_cols):
                    #計算kernel在img上的對應位置
                    i_k_h = i_h + k_h - kernel_center_x
                    i_k_w = i_w + k_w - kernel_center_y
                    conv_sum += (img[i_k_h][i_k_w]) * (kernel[k_h][k_w])
            result[i_h][i_w] = min(255,max(0,conv_sum))
    print(result.shape)
    return result

def sobel_operator(img):
    #定義計算水平和垂直邊緣的kernel
    horizontal_kernel = np.array([
                        [-1, 0, 1],
                        [-2, 0, 2], 
                        [-1, 0, 1]])
    
    vertical_kernel = np.array([
                        [-1, -2, -1], 
                        [0, 0, 0], 
                        [1, 2, 1]])
    #計算水平和垂直邊緣
    horizontal_edge , vertical_edge = Convolution(img,horizontal_kernel),Convolution(img,vertical_kernel)
    
    #輸出圖片
    sobel_img = np.zeros((img.shape[0],img.shape[1]),dtype=img.dtype)

    #計算水平和垂直相加後輸出的結果
    for i in range(horizontal_edge.shape[0]):
        for j in range(vertical_edge.shape[1]):
            result = int(horizontal_edge[i][j]) + int(vertical_edge[i][j])
            sobel_img[i][j] = result if result <256 else 255
    return sobel_img

def blur(img):
    result = np.zeros((img.shape[0],img.shape[1]),dtype=np.float32)
    blur_kernel = np.array([
               [1/9, 1/9, 1/9],
               [1/9, 1/9, 1/9],
               [1/9, 1/9, 1/9]])
    blur = Convolution(img, blur_kernel)
    result = output_img(img,blur)
    return result

def laplacian(img):
    result = np.zeros((img.shape[0],img.shape[1]),dtype=np.float32)
    laplacian_kernel =np.array([
        [-1,-1,-1],
        [-1,8,-1],
        [-1,-1,-1]])
    laplacian = Convolution(img,laplacian_kernel)
    result = output_img(img,laplacian)
    return result

def blur_normalize(img):
    img_min = np.amin(img)
    img_max = np.amax(img)
    normalize = np.zeros((img.shape[0],img.shape[1]),dtype=np.float32)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            normalize[i][j] = float((img[i][j]-img_min) / (img_max-img_min))
    return normalize

def multiple(blur_nor,img2):
    result = np.zeros((img2.shape[0],img2.shape[1]),dtype=img2.dtype) #dtype = img2.dtype 則mul結果正常 , dtype= np.float32 則 最後結果正常
    for i in range(img2.shape[0]):
        for j in range(img2.shape[1]):
            result[i][j] = float(blur_nor[i][j]) * img2[i][j]
    return result

def add(org_img , mul_img):
    result = np.zeros((org_img.shape[0],org_img.shape[1]),dtype = org_img.dtype)
    for i in range(org_img.shape[0]):
        for j in range(org_img.shape[1]):
            result[i][j] = org_img[i][j]+mul_img[i][j] if org_img[i][j]+mul_img[i][j] < 256 else 255
    return result

    
img = cv2.imread('Lenna.jpg')
img_gray = trans_gray(img.copy())
sobel_img = sobel_operator(img_gray.copy()) #一階微分後結果
blur_img = blur(sobel_img.copy())
laplacian_img = laplacian(img_gray.copy())
blur_normalize = blur_normalize(blur_img.copy())
multiple_img= multiple(blur_normalize,laplacian_img)
result = add(img_gray.copy(),multiple_img.copy())

cv2.imshow('img',img_gray)
cv2.imshow('sobel',sobel_img)
cv2.imshow('blur',blur_img)
cv2.imshow('laplacian',laplacian_img)
cv2.imshow('mul',multiple_img)
cv2.imshow('final',result)
cv2.waitKey(0)
