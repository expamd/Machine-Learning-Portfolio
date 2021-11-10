'''
Program do automatycznej i szybkiej obróbki zdjęć z OneDrive wykorzystujący kilka wytrenowanych sieci konwolucyjnych. Program głównie działa
na raspberrypi podłączonym do sieci i chmury OneDrive. OneDrive jest zmontowany na raspberrypi przez biblioteke rclone. Na wejściu
program szuka w folderze zdjęcia do obórbki. Następnie zdjęcie jest przetwarzane przez sieć do klasyfikacji która rozróżnia 7 różnych
scen(Krajobraz,górski,miejski,leśny,wodny,wschód/zachód,nocny). Po udanej prognozie sieci zdjęcie jest poddane podstawowej obróbce
(ekspozycja,kontrast,nasycenie,wyostrzanie)unikalnej dla każdej klasy. Następnie zdjęcie jest odszumiane i wyostrzanie wytrenowanymni
do tego sieciami U-Net. Potem w zależności od wymiarów zdjęcie jest powiększane siecią SRGAN 2-krotnie lub 4-krotnie. Na końcu zdjęcie jest
zapisywane i usuwane z folderu wejściowego, aby zwalniać miejsce na kolejne.


'''

#Importowanie potrzebnych bibliotek
import cv2
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image as image1
from tensorflow.keras.preprocessing.image import img_to_array
from wand.image import Image
from datetime import date
import time
from PIL import Image as pil
from PIL import ImageEnhance as enh
from tensorflow import keras
from tensorflow.keras import backend as K
import matplotlib.image as mpimg
import tensorflow as tf
import tensorflow_addons as tfa
from skimage.io import imsave
import gc
import random
from time import sleep
from skimage import exposure

os.system('export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libgomp.so.1')

K.clear_session()
#Przypisywanie ilości wątków do konkretnych zadań aby zredukować ilość alokowanej pamięci RAM
tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(4)
with tf.device('/CPU:0'):

    input_path='/home/pi/OneDrive/Auto_Process/Input/'
    output_path='/home/pi/OneDrive/Auto_Process/Output/'

#Funkcja wykorzystująca wcześniej wytrenowaną sieć do klasyfikacji obrazów
    def scene_predict(img_path):
        global a,b,c,d,e,f,g
        #Wczytanie modelu sieci i odpowiednie przetworzenie zdjęcia wejściowego
        model=load_model('/home/pi/FTPR4.h5')
        img_to_pred=cv2.imread(img_path,cv2.IMREAD_UNCHANGED)
        img_to_pred=cv2.resize(img_to_pred,(480,416))
        img_to_pred=img_to_pred.astype('float32')/255
        img_to_pred=np.expand_dims(img_to_pred,axis=0)
        prediction=model.predict(img_to_pred)
        
        pred=prediction.tolist()
        for i in pred:
            print(i)

        a=i[0]
        b=i[1]
        c=i[2]
        d=i[3]
        e=i[4]
        f=i[5]
        g=i[6]

        #Zwalnianie niepotrzebnych zasobów
        K.clear_session()
        del model
        gc.collect()

    #Funkcja odpowiadająca za odszumianie zdjęcia wejściowego w celu lepszej jakości wizualnej i dalszej obróbki. Odbywa się to za pomocą wytrenowanej małej sieci U-net metodą GAN,
        #sieć została wytrenowana na kilku tysiącach zdjęć zaszumionych i normalnych. 
    def denoise(img_array):
        denoiser=load_model('/home/pi/models/DNGEN33v2.h5',custom_objects={'LeakyReLU': tf.keras.layers.LeakyReLU},compile=False)
        x=div_check(img_array.shape[1])
        y=div_check(img_array.shape[0])
        img_array=resizer(img_array,x,y)
        t=np.array(img_array)/255.
        t=t.astype('float32')
        print('Odszumiam...')
        t=np.expand_dims(t,axis=0)
        img=denoiser.predict(t)
        img=img[0,:,:,:]
        img=(img*255).astype(np.uint8)
        img_add=layer_thresh(img,img_array,0.4)
        del denoiser
        K.clear_session()
        gc.collect()
        return img_add

#Podobnie jak funkcja wyżej odszumia zdjęcie wejściowe za pomocą sieci U-net ale tylko dla scenerii nocnej.
    def night_denoise(img_array):
        denoiser_night=load_model('/home/pi/models/NVHRrasp8U.h5')
        x=div_check(img_array.shape[1])
        y=div_check(img_array.shape[0])
        img_array=resizer(img_array,x,y)
        tn=np.array(img_array)/255.
        tn=tn.astype('float32')
        print('Odszumiam...')
        tn=np.expand_dims(tn,axis=0)
        img=denoiser_night.predict(tn)
        img=img[0,:,:,:]
        img=(img*255).astype(np.uint8)
        img_add=layer_thresh(img,img_array,0.75)
        del denoiser_night
        K.clear_session()
        gc.collect()
        return img_add

#Funkcja odpowiadająca za wyostrzanie zdjęcia w celu poprawy jakości i szczegółowości. Również sieć U-Net trenowana metodą GAN.
    #Zbiór danych został wcześniej przygotowany poprzez degradajce zdjęcia wejśćiowego(Różne poziomy rozmycia gaussowskiego i zaszumienie).  
    def sharpen(img_array):
        sharpener=load_model('/home/pi/DB34v4.h5')
        t=np.array(img_array)/127.5-1
        t=t.astype('float32')
        print('Wyostrzam...')
        t=np.expand_dims(t,axis=0)
        img=sharpener.predict(t)
        img=img[0,:,:,:]
        img=((img+1)*127.5).astype(np.uint8)
        img_add=layer_thresh(img,img_array,0.8)
        del sharpener
        K.clear_session()
        gc.collect()
        return img_add

#Funkcja, której zadaniem jest wczytanie odpowiedniego modelu sieci i zwiększenie natywnej rozdzielczości zdjęcia(2x,4x).
    #Więcej szczegółów w pliku SRGAN.ipynb
    def SRGAN(img_array,mode):
        if mode=='4x':
            gan=load_model('/home/pi/SRmodels/GEN200v3-4x.h5',compile=False,custom_objects={'PReLU':tf.keras.layers.PReLU))
        else:
            gan=load_model('/home/pi/SRmodels/GEN190v3-2x.h5',compile=False,custom_objects={'LeakyReLU':tf.keras.layers.LeakyReLU})
        img=cv2.cvtColor(img_array,cv2.COLOR_RGB2BGR)
        img=img.astype('float32')/127.5-1
        img=np.expand_dims(img,axis=0)
        print('Zwiększanie Rozdzielczości...')
        score=gan.predict(img)
        score=score[0,:,:,:]
        score=cv2.cvtColor(score,cv2.COLOR_BGR2RGB)
        score=((score+1)*127.5).astype(np.uint8)
        final=histogram_match(img_array,score)
        return final
        del gan
        K.clear_session()
        gc.collect()

#Funkcja która pozwala na dostosowanie pokrycia zdjęcia przetworzonego z referencyjnym       
    def layer_thresh(img1,img2,alpha):
        beta=(1.0-alpha)
        result=cv2.addWeighted(img1,alpha,img2,beta,0.0)
        return result

#Funkcja pozwalająca na dopasowanie histogramów zdjęcia przetworzonego z referencyjnym
    def histogram_match(img1,img2):
        y,x,c=img2.shape
        img1=cv2.resize(img1,(x,y),interpolation=cv2.INTER_CUBIC)
        matched=exposure.match_histograms(img2,img1,multichannel=True)
        return matched
        

#Zbiór funkcji obróbki zdjęć dla siedmiu scenerii  
def gory(img_path,radius1,radius2,sigma1,sigma2):
    img=Image(filename=img_path)
    img.gamma(1.1)
    img.level(0.11,0.9,gamma=1.05)
    img.adaptive_blur(radius=radius1, sigma=sigma1)
    img.colorize(color='green',alpha='rgb(3%,4%,5%)')
    img.adaptive_sharpen(radius=radius2, sigma=sigma2)
    img=np.array(img)
    img=pil.fromarray(img)
    enhancer=enh.Color(img)
    img=enhancer.enhance(1.4)
    img=np.array(img)
    img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    return img


def miejski(img_path,radius1,radius2,sigma1,sigma2,amount):
    img=Image(filename=img_path)
    img.level(0.1,0.95,gamma=1.05)
    img.adaptive_blur(radius=radius1, sigma=sigma1)
    img.unsharp_mask(radius=radius2, sigma=sigma1,amount=amount)
    img=np.array(img)
    img=pil.fromarray(img)
    enhancer=enh.Color(img)
    img=enhancer.enhance(1.4)
    img=np.array(img)
    img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    return img
    

def lesny(img_path,radius1,radius2,sigma1,sigma2):
    img=Image(filename=img_path)
    img.gamma(1.11)
    img.adaptive_blur(radius=radius1, sigma=sigma1)
    img.gaussian_blur(sigma=sigma2)
    img.adaptive_sharpen(radius=radius2, sigma=sigma1)
    img.colorize(color='green',alpha='rgb(3%,5%,10%)')
    img=np.array(img)
    img=pil.fromarray(img)
    enhancer=enh.Color(img)
    img=enhancer.enhance(1.45)
    img=np.array(img)
    img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    return img

def krajobraz(img_path,radius1,radius2,radius4,sigma1,sigma2):
    img=Image(filename=img_path)
    img.gamma(1.11)
    img.adaptive_blur(radius=radius1, sigma=sigma1)
    img.gaussian_blur(sigma=sigma2)
    img.adaptive_sharpen(radius=radius2, sigma=sigma1)
    img.adaptive_sharpen(radius=radius4, sigma=sigma1)
    img=np.array(img)
    img=pil.fromarray(img)
    enhancer=enh.Color(img)
    img=enhancer.enhance(1.4)
    img=np.array(img)
    img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    return img
    
def wodny(img_path,radius1,radius2,sigma1,sigma2,amount):
    img=Image(filename=img_path)
    img.gamma(1.1)
    img.unsharp_mask(radius=radius1,sigma=sigma1,amount=amount,threshold=0)
    img.colorize(color='green',alpha='rgb(3%,7%,16%)')
    img.adaptive_blur(radius=radius2, sigma=sigma2)
    img.colorize(color='green',alpha='rgb(3%,7%,16%)')
    img=np.array(img)
    img=pil.fromarray(img)
    enhancer=enh.Color(img)
    img=enhancer.enhance(1.2)
    img=np.array(img)
    img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    return img
    
def wsch(img_path,radius1,radius2,sigma1,sigma2,amount):
    img=Image(filename=img_path)
    img.gamma(1.1)
    img.unsharp_mask(radius=radius1,sigma=sigma1,amount=amount,threshold=0)
    img.adaptive_blur(radius=radius2, sigma=sigma2)
    img.colorize(color='red',alpha='rgb(5%,7%,3%)')
    img=np.array(img)
    img=pil.fromarray(img)
    enhancer=enh.Color(img)
    img=enhancer.enhance(1.4)
    img=np.array(img)
    img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    return img


def nocny(img_path,radius1,radius2,sigma1,sigma2):
    img=Image(filename=img_path)
    img.gamma(1.14)
    img.adaptive_blur(radius=radius1, sigma=sigma1)
    img.gaussian_blur(sigma=sigma2)
    img.adaptive_sharpen(radius=radius2, sigma=sigma1)
    img=np.array(img)
    img=pil.fromarray(img)
    enhancer=enh.Color(img)
    img=enhancer.enhance(1.1)
    img=np.array(img)
    img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    return img

#Funkcja sprawdzająca wymiary zdjęcia wejściowego, aby móc przeprowadzić prognoze sieci U-net. 
def div_check(x):
    while True:
        if int(x)%8==0:
            return x
            break
        else:
            x=x-1
    
#Funkcja zmieniająca wymiary zdjęcia
def resizer(img_array,x,y):
        img_array=cv2.resize(img_array,(x,y),interpolation=cv2.INTER_AREA)
        return img_array
#Funkcja zapisująca zdjęcie
def img_saver(img_array,date):
    cv2.imwrite(output_path+date+'.png',img_array)
    

input_img='calibration_field'

#Pętla przeszukująca folder w poszukiwaniu zdjęć
while True:
    try:
        input_img=random.choice(os.listdir(input_path))
    except IndexError:
        print('Pusto')
    if os.path.exists(input_path+input_img) is True:
        img_path=input_path+input_img
        print(img_path)
        break
    else:
        print('śpie')
        sleep(10)
        continue

scene_predict(img_path)

#Pobieranie informacji o dacie i czasie w momencie zapisywania pliku
date=date.today()
d1=date.strftime('%d%m%Y')
t=time.localtime()
ct=time.strftime('%H%M%S',t)
out='IMG'+str(d1)+'_'+str(ct)+'_MN10'
#Zbiór instrukcji warunkowych, które decydują która klasa z sieci do prognozy scenerii ma być wykonana oraz czy wykonać interpolacje zdjęcia.
if a>b and a>c and a>d and a>e and a>f and a>g:
    print('Krajobraz Górski')
    image=gory(img_path,1,6,1,5)
    image=denoise(image)
    image=sharpen(image)
    if image.shape[1]<1120 and image.shape[0]<960:
        image=SRGAN(image,'4x')
        img_saver(image,out)
    elif image.shape[1]>1120 and image.shape[1]<2100 and image.shape[0]>960 and image.shape[0]<1700:
        image=SRGAN(image,'2x')
        img_saver(image,out)
    else:
        img_saver(image,out)
    print('Zapisano')
elif b>a and b>c and b>d and b>e and b>f and b>g:
    print('Krajobraz Wodny')
    image=wodny(img_path,8,1,4,4,1)
    image=denoise(image)
    image=sharpen(image)
    if image.shape[1]<1120 and image.shape[0]<960:
        image=SRGAN(image,'4x')
        img_saver(image,out)
    elif image.shape[1]>1120 and image.shape[1]<2100 and image.shape[0]>960 and image.shape[0]<1700:
        image=SRGAN(image,'2x')
        img_saver(image,out)
    else:
        img_saver(image,out)
    print('Zapisano')
elif c>a and c>b and c>d and c>e and c>f and c>g:
    print('Krajobraz Miejski')
    image=miejski(img_path,2,8,2,1,2)
    image=denoise(image)
    image=sharpen(image)
    if image.shape[1]<1120 and image.shape[0]<960:
        image=SRGAN(image,'4x')
        img_saver(image,out)
    elif image.shape[1]>1120 and image.shape[1]<2100 and image.shape[0]>960 and image.shape[0]<1700:
        image=SRGAN(image,'2x')
        img_saver(image,out)
    else:
        img_saver(image,out)
    print('Zapisano')
elif d>a and d>b and d>c and d>e and d>f and d>g:
    print('Krajobraz Wsch/Zach')
    image=wsch(img_path,5,2,4,2,4)
    image=denoise(image)
    image=sharpen(image)
    if image.shape[1]<1120 and image.shape[0]<960:
        image=SRGAN(image,'4x')
        img_saver(image,out)
    elif image.shape[1]>1120 and image.shape[1]<2100 and image.shape[0]>960 and image.shape[0]<1700:
        image=SRGAN(image,'2x')
        img_saver(image,out)
    else:
        img_saver(image,out)
    print('Zapisano')
elif e>a and e>b and e>c and e>d and e>f and e>g:
    print('Krajobraz Leśny')
    image=lesny(img_path,2,7,2,0.1)
    image=denoise(image)
    image=sharpen(image)
    if image.shape[1]<1120 and image.shape[0]<960:
        image=SRGAN(image,'4x')
        img_saver(image,out)
    elif image.shape[1]>1120 and image.shape[1]<2100 and image.shape[0]>960 and image.shape[0]<1700:
        image=SRGAN(image,'2x')
        img_saver(image,out)
    else:
        img_saver(image,out)
    print('Zapisano')
elif f>a and f>b and f>c and f>d and f>e and f>g:
    print('Krajobraz')
    image=krajobraz(img_path,1,4,5,1,0.1)
    image=denoise(image)
    image=sharpen(image)
    if image.shape[1]<1120 and image.shape[0]<960:
        image=SRGAN(image,'4x')
        img_saver(image,out)
    elif (image.shape[1]>1120 and image.shape[1]<2100) and (image.shape[0]>960 and image.shape[0]<1700):
        image=SRGAN(image,'2x')
        img_saver(image,out)
    else:
        img_saver(image,out)
    print('Zapisano')
elif g>a and g>b and g>c and g>d and g>e and g>f:
    print('Krajobraz Nocny')
    nocny(img_path,2,4,3,0.15)
    image=denoise(image)
    image=sharpen(image)
    if image.shape[1]<1120 and image.shape[0]<960:
        image=SRGAN(image,'4x')
        img_saver(image,out)
    elif image.shape[1]>1120 and image.shape[1]<2100 and image.shape[0]>960 and image.shape[0]<1700:
        image=SRGAN(image,'2x')
        img_saver(image,out)
    else:
        img_saver(image,out)
    print('Zapisano')

#Usunięcie zdjęcia z folderu    
os.remove(img_path)
#Ponowne odtworzenie programu
os.system('python3 /home/pi/PRauto.py')



