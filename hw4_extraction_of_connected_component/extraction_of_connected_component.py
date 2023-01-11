import cv2
import numpy as np
def ConnectedComponents(img):
    row , col = img.shape[0] , img.shape[1]
    visited = [[False for i in range(col)] for j in range(row)] #建立一個跟圖片一樣大小的booling，用於紀錄是否拜訪過
    components = [] #儲存所有components的list
    for i in range(row):
        for j in range(col):
            if not visited[i][j] and img[i][j]==0:
                component = []
                DFS(img , i , j , visited , component) #對pixel做DFS搜尋
                components.append(component)
    return components

def DFS(img , row , col , visited , component):
    if img[row][col] == 0 and not visited[row][col]:
        visited[row][col] = True #標記為拜訪過的pixel
        component.append((row,col))
        #對上下左右的像素進行DFS
        DFS(img, row - 1, col, visited, component)
        DFS(img, row + 1, col, visited, component)
        DFS(img, row, col - 1, visited, component)
        DFS(img, row, col + 1, visited, component)

def draw_img():
    img = 255 * np.ones((100,300) , dtype=np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img , 'NCHU SIP Lab' , (0,25) , font , 1 , 0 ,2)
    cv2.putText(img , 'Happy New Year' , (0,50) , font , 1 , 0 ,2)
    cv2.putText(img , '2023' , (0,90) , font , 1 , 0 ,2)
    return img

def print_table(components):
    print('-'*40)
    print(f'{"Connected":<8}          {"No. of pixels in":<15}')
    print(f'{"conponent":<8}          {"connected component":<15}')
    print('-'*40)
    for i , component in enumerate(components):
        if i<9:
            print(f'0{i+1:<7}             {len(component):<15}')
        else:
            print(f'{i+1:<8}             {len(component):<15}')

img = draw_img()
components = ConnectedComponents(img.copy())
print_table(components)
cv2.imwrite('nchu.png',img)
cv2.imshow('img',img)
cv2.waitKey(0)
