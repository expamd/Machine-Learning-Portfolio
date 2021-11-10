'''
Krótki skrypt pozwalający tworzyć zbiór danych. Zapisuje klatki RGB i HSV z odpowiednimi etykietami. To wszystko jest zapisywane w pliku CSV.
'''

import cv2
import numpy as np
import pandas as pd

obj_values=[]
img_org_names=[]
img_mask_names=[]

def cam():
    lower_RGB=np.array([0,0,0])
    upper_RGB=np.array([30,255,15])

    cap=cv2.VideoCapture(0)
    count=2985
    count_img=2985

    while True:
        ret,frame=cap.read()
        frame=cv2.rotate(frame,cv2.ROTATE_180)
        frame_copy=frame.copy()
        frame_copy=cv2.cvtColor(frame_copy,cv2.COLOR_BGR2RGB)
        frame_copy=cv2.cvtColor(frame_copy,cv2.COLOR_RGB2HSV)
        
        mask=cv2.inRange(frame_copy,lower_RGB,upper_RGB)
        
        count +=1
        count_img +=1
        if count%15==0:
            decision=int(input('0/1: '))
            if decision>1:
                decision=1
            csv_write(img_org=frame,img_mask=mask,count=count_img,obj=decision,
                      obj_values=obj_values,img_org_names=img_org_names,img_mask_names=img_mask_names)

        save_csv()
        frame_copy=cv2.cvtColor(frame_copy,cv2.COLOR_HSV2RGB)
           
        cv2.imshow('mask',mask)
        cv2.imshow('img',frame_copy)
        print('licznik',count)
        key=cv2.waitKey(1)&0xFF
        if key==ord('q'):
            break

    cv2.destroyAllWindows()
    cap.release()




def csv_write(img_org,img_mask,count,obj,
              obj_values,img_org_names,img_mask_names):
    img_path='/home/pi/images/org/'
    img_path2='/home/pi/images/mask/'
    img_org=cv2.resize(img_org, (320,240))
    img_mask=cv2.resize(img_mask,(320,240))
    #img_org=cv2.cvtColor(img_org,cv2.COLOR_BGR2RGB)
    count=int(count/15)
    cv2.imwrite(img_path+str(count)+'.jpg',img_org)
    cv2.imwrite(img_path2+'mask_'+str(count)+'.jpg',img_mask)
    img_org_name=str(count)+'.jpg'
    img_mask_name='mask_'+str(count)+'.jpg'
    img_org_names.append(img_org_name)
    img_mask_names.append(img_mask_name)
    obj_values.append(obj)

def save_csv():
    boxes={'orginal_images': img_org_names,
           'mask_images': img_mask_names,
           'object': obj_values}
    df=pd.DataFrame(boxes,columns=['orginal_images','mask_images','object'])
    df.to_csv('/home/pi/labels2.csv',index=False,header=True)


cam()
