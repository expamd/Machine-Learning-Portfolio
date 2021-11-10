'''
Projekt ten to od podstaw zaprojektowany mały, jeżdżący robot gdzie główną jednostką obliczeniową i wykonawczą jest Raspberry Pi 4.
Głównym systemem nawigacji są czujniki ultradźwiękowe i wizja komputerowa. Czyli kamera na przodzie w czasie rzeczywistym rejestruje
obraz i jest on przetwarzany w odpowiedni sposób aby następnie móc przeprowadzić prognozę za pomocą prostej i lekkiej sieci.
Sieć bierze na wejśćiu zdjęcie zwykłe z kamery a drugie wejście to utworzona maska od niego w przestrzeni barw HSV. Wynik sieci jest
przedstawiony w zakresie[0,1], gdzie 0 to wykryta przeszkoda, a 1 to brak przeszkód na drodze. Kierunek skrętu, aby wyminąć potencjalną
przeskodę jest obliczany na podstawie pola krawędzi na danej połowie obrazu.
Do zrobienia: Zamontowanie akcelerometru i kompasu aby móc nadać pełną autonomie podczas jazdy. Dodać mikrofon i głośnik, aby móc wchodzić
w interakcje.
 
'''

#Importowanie koniecznych bibliotek
import cv2
import numpy as np
import os
import gc
import random
import RPi.GPIO as GPIO
from time import sleep, time
import tensorflow as tf
from tensorflow.keras.models import Model,load_model

#Definiowanie zmiennych dla silniczków 
motor_right1=16
motor_right2=26
motor_right3=13#pwm

motor_left1=5
motor_left2=6
motor_left3=12#pwm

sensor1_echo=25
sensor1_trig=24

#Funkcja uruchamiająca silniki, aby były gotowe do pracy 
def setup(hz,mode):
    global pwm_left,pwm_right
    GPIO.setmode(mode)
    GPIO.setup(motor_left1,GPIO.OUT)
    GPIO.setup(motor_left2,GPIO.OUT)
    GPIO.setup(motor_left3,GPIO.OUT)
    pwm_left=GPIO.PWM(motor_left3, hz)

    GPIO.setup(motor_right1,GPIO.OUT)
    GPIO.setup(motor_right2,GPIO.OUT)
    GPIO.setup(motor_right3,GPIO.OUT)
    pwm_right=GPIO.PWM(motor_right3, hz)
    
    GPIO.setup(sensor1_echo,GPIO.IN)
    GPIO.setup(sensor1_trig,GPIO.OUT)
    
    
    pwm_left.start(0)
    pwm_right.start(0)
    
#Funkcja odpowiadająca za możliwość poruszania się do przodu,tyłu i na boki. 
def move(speed=0.5,turn=0):
    speed *=100
    turn *=100
    leftspeed=speed-turn
    rightspeed=speed+turn
    if leftspeed>100:
        leftspeed=100
    elif leftspeed<-100:
        leftspeed=-100
    if rightspeed>100:
        rightspeed=100
    elif rightspeed<-100:
        rightspeed=-100
    pwm_left.ChangeDutyCycle(abs(leftspeed))
    pwm_right.ChangeDutyCycle(abs(rightspeed))
    
    if leftspeed>0:
        GPIO.output(motor_left1,GPIO.HIGH)
        GPIO.output(motor_left2,GPIO.LOW)
    else:
        GPIO.output(motor_left1,GPIO.LOW)
        GPIO.output(motor_left2,GPIO.HIGH)
        
    if rightspeed>0: 
        GPIO.output(motor_right1,GPIO.LOW)
        GPIO.output(motor_right2,GPIO.HIGH)
    else:
        GPIO.output(motor_right1,GPIO.HIGH)
        GPIO.output(motor_right2,GPIO.LOW)
    #sleep(t)
    
#Funkcja zatrzymująca prace silników
def stop(t=0):
    pwm_right.ChangeDutyCycle(0)
    pwm_left.ChangeDutyCycle(0)
    sleep(t)

#Czyszczenie GPIO,konieczne aby potem program poprawnie działał
def clean():
    GPIO.cleanup() 
    
#Wczytywanie wcześniej wytrenowanego modelu do poruszania się w terenie
def model_load(model_name):
    model=load_model('/home/pi/'+model_name+'.h5',custom_objects={'LeakyReLU':tf.keras.layers.LeakyReLU})
    return model

#Funkcja do przeprowadzenia prognozy z sieci
def model_predict(model,img_array,mask_array):
    img_array=img_array.astype('float32')/255
    mask_array=mask_array.astype('float32')/255
    img_array=cv2.resize(img_array,(220,160))
    mask_array=cv2.resize(mask_array,(220,160))
    img_array=np.expand_dims(img_array,axis=0)
    mask_array=np.expand_dims(mask_array,axis=0)
    prediction=model.predict([img_array,mask_array])
    return prediction

#Generowanie maski do zdjęcia wejściowego w celu poprawy widoczności dla bliższych obiektów
def generate_mask(image):
    lower_RGB=np.array([0,0,0])
    upper_RGB=np.array([30,255,15])
    image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    image=cv2.cvtColor(image,cv2.COLOR_RGB2HSV)    
    mask=cv2.inRange(image,lower_RGB,upper_RGB)
    return mask

#Dzielnie zdjęcia wejściowego na pół aby móc obliczyć pole konturów i wybrać kierunek jazdy
def intrest_region(image):
    h=image.shape[0]
    left_half=image[0:480,0:320]
    right_half=image[0:480,320:640]
    return left_half,right_half

#Rysowanie konturów na zdjęciu wejściowym
def get_contours(img,imgcon,l):
    contours,_=cv2.findContours(img,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    for cnt in contours:
        area=cv2.contourArea(cnt)
        l.append(area)
        if area>100:
            cv2.drawContours(imgcon,cnt,-1,(255,0,0),3)

#Funkcja główna która zczytuje obraz z kamery i znajduje krawędzie. Na podstawie tego obliczana jest suma pól wszystkich krawędzi i konturów
            #na lewej lub prawej części zdjęcia, w ten sposób wybierany jest kierunek jazdy. Również w tej funkcji sieć decyduje czy na terenie przed
            #jest jakaś przeszkoda czy też nie
def camera_drive(model):
    cap=cv2.VideoCapture(0)
    count=0
    area_left=[]
    area_right=[]
    while True:
        ret,frame=cap.read()
        frame=cv2.rotate(frame,cv2.ROTATE_180)
        mask=generate_mask(frame)
        left_img,right_img=intrest_region(frame)
        left_edges=cv2.Canny(left_img,50,100)
        right_edges=cv2.Canny(right_img,50,100)
        get_contours(left_edges,left_img,area_left)
        get_contours(right_edges,right_img,area_right)
        area_left_sum=sum(area_left)
        area_right_sum=sum(area_right)
        
        count +=1
        
        result=model_predict(model,frame,mask)
        print(result)

        #Instrukcje warunkowe, decydujące o kierunku jazdy. 
        if result>0.50:
            move(0.8,0)
        else:
            stop(1.5)
            move(-0.9,0)
            if area_left_sum>area_right_sum:
                move(0,0.9)
                sleep(0.5)
            else:
                move(0,-0.9)
                sleep(0.5)
                

        cv2.imshow('frame',frame)
        gc.collect()
        
        if cv2.waitKey(1)==ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

#Główne wykonanie skryptu    
try:
    setup(50,GPIO.BCM)
    model=model_load('drive_model13')
    camera_drive(model)
    clean()
except KeyboardInterrupt:
    clean()
    
