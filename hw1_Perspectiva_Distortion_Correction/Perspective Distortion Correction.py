import os
import cv2
import numpy as np
def search_file(path): #return all file name
    file=[]
    for i in os.listdir(path):
        if (i[-4:]=='.jpg' or i[-4:]=='.JPG' or i[-4:]=='.png') and i[:-4] not in file:
            i = i[:-4]
            file.append(i)
    return file

def show_xy(event,x,y,flags,param):
    global point, img2
    if event ==1:
        print(x,y)
        point.append([x,y])
        img_circle = cv2.circle(img2, (x,y), 10, (0,0,225), -1)
        cv2.imshow("img",img2)
    if len(point)==4:
        cv2.destroyAllWindows()

def calculate(img , expect_size ,point):
    right_up = []
    right_down = []
    left_up = []
    left_down = []
    x_max = 0 
    x_max_index=0 
    x_second=0 
    x_second_index = 0
    for i in range(len(point)):
        if point[i][0]>x_max:
            x_max=point[i][0]
            x_max_index = i
    for i in range(len(point)):
        if point[i][0]>x_second and i!=x_max_index:
            x_second = point[i][0]
            x_second_index = i
    if point[x_max_index][1]>point[x_second_index][1]:
        right_down=(point[x_max_index])
        right_up=(point[x_second_index])
    else:
        right_up=(point[x_max_index])
        right_down=(point[x_second_index])
    point.remove(right_up)
    point.remove(right_down)
    if point[0][1] < point[1][1]:
        left_down=point[1]
        left_up = point[0]
    else:
        left_down=point[0]
        left_up = point[1]
    point = [left_up,right_up,right_down,left_down]
    x1_ = y1_ = x2_ = y4_ = 0
    x4_ , x3_ , y2_ ,y3_ = expect_size[0] , expect_size[0] , expect_size[1] ,expect_size[1]
    A = np.array([(x1_,y1_,x1_*y1_,1,0,0,0,0),(x2_,y2_,x2_*y2_,1,0,0,0,0),(x3_,y3_,x3_*y3_,1,0,0,0,0),
                        (x4_,y4_,x4_*y4_,1,0,0,0,0),(0,0,0,0,x1_,y1_,x1_*y1_,1),(0,0,0,0,x2_,y2_,x2_*y2_,1),
                        (0,0,0,0,x3_,y3_,x3_*y3_,1),(0,0,0,0,x4_,y4_,x4_*y4_,1)])
    b = np.array([(point[0][0]),(point[1][0]),(point[2][0]),(point[3][0]),(point[0][1]),(point[1][1]),\
                  (point[2][1]),(point[3][1])])
    x = np.linalg.solve(A, b)
    return x

def inverse_mapping(b,input_img,output_img):
    print(input_img.shape , output_img.shape)
    row ,col = output_img.shape[:2]
    for i in range(row):
        for j in range(col):
            d_y = ((b[0]*i) + (b[1]*j) + (b[2]*i*j) + b[3])
            d_x = ((b[4]*i) + (b[5]*j) + (b[6]*i*j) + b[7])
            y , x = int(d_y) , int(d_x)
            u , v = d_y-y , d_x-x
            for c in range(3):
                output_img[i,j,c] = (1-u) * (1-v) * input_img[x,y][c] + u *(1-v) * input_img[x,y+1][c] + \
                v * (1-u) * input_img[x+1,y][c] + u*v*input_img[x+1,y+1][c]
    cv2.imshow("img",output_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
img_path = './'
image = search_file(img_path)
if 'desktop' in image:image.remove('desktop')


for i in image:
    expect_size = [1200,1000]
    point=[]
    img = cv2.imread(f'{img_path}{i}.JPG')
    img = cv2.resize(img,(int(img.shape[0]*0.3),int(img.shape[1]*0.3)))
    img2 = img.copy()
    cv2.imshow("img",img2)
    cv2.setMouseCallback('img',show_xy)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    x = calculate(img ,expect_size, point)
    output_img = np.zeros((expect_size[0],expect_size[1],3),dtype=img.dtype)
    inverse_mapping(x,img,output_img)