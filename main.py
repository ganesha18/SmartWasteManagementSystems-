#Include Libraries
import RPi.GPIO as GPIO
import time
import _thread
import cv2
import numpy as np
from PIL import Image
import tensorflow as tf
import os

#
#Setup Pins
#


GPIO.setwarnings(False)
GPIO.setmode(GPIO.BCM)

# Setup IR Sensor Pins
IR_SENSOR =14 
IR_LED = 15
s = False

GPIO.setup(IR_SENSOR, GPIO.IN)
GPIO.setup(IR_LED, GPIO.OUT)

# Declare the Camera Variables
CAM = cv2.VideoCapture(0)
sampleImg = '/home/pi/Downloads/demo/images/sampleImg.jpg'

#Servo Motor Pins
SERVO_MOTOR = 12
GPIO.setup(SERVO_MOTOR, GPIO.OUT)

SERVO = GPIO.PWM(SERVO_MOTOR, 50)
SERVO.start(5)

#
#Initializing while Loop
#

while s == False:      
    #Initializing the IR Sensor
    if GPIO.input(IR_SENSOR):
        GPIO.output(IR_LED, False)
        #while GPIO.input(IR_SENSOR):
            #time.sleep(0.2)

    else:
        GPIO.output(IR_LED, True)
        s = True
        
    #Initializing the Camera Module
    ret,img = CAM.read()
    cv2.imshow('Test',img)
    if not ret:
        break
    cv2.waitKey(1)
        
    if s == True:
        #Saving Photo of the Object
        cv2.imwrite(sampleImg,img)
        GPIO.output(IR_LED, False)
    
    CAM.release
    
    if os.path.exists(sampleImg):
        #time.sleep(0.5)
        
        #Load Model and predict the Result
        interpreter = tf.lite.Interpreter(model_path='/home/pi/Downloads/model_unquant.tflite')
        interpreter.allocate_tensors()

        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        # check the type of the input tensor
        floating_model = input_details[0]['dtype'] == np.float32

        # NxHxWxC, H:1, W:2
        height = input_details[0]['shape'][1]
        width = input_details[0]['shape'][2]
        img = Image.open(sampleImg).resize((width, height))

        # add N dim
        input_data = np.expand_dims(img, axis=0)

        if floating_model:
            input_data = (np.float32(input_data) - 127.5) / 127.5

        interpreter.set_tensor(input_details[0]['index'], input_data)

        start_time = time.time()
        interpreter.invoke()
        stop_time = time.time()

        output_data = interpreter.get_tensor(output_details[0]['index'])
        results = np.squeeze(output_data)

        top_k = results.argsort()[-1:][::1]
        time_taken = (stop_time - start_time) * 1000
        
        print(results)
        print(top_k)

        if(top_k == 0):
            print('Plastic, Time Taken: {:.3f}ms, Motor Turning Left'.format(time_taken))
            time.sleep(1.5)
            SERVO.ChangeDutyCycle(12.5)
            time.sleep(1)
            SERVO.ChangeDutyCycle(7.5)
            time.sleep(0.5)
            SERVO.ChangeDutyCycle(0)
            os.remove(sampleImg)
            s = False
            
        else:
            print('Non Plastic, Time Taken: {:.3f}ms, Motor Turning Right'.format(time_taken))
            time.sleep(1.5)
            SERVO.ChangeDutyCycle(2.5)
            time.sleep(1)
            SERVO.ChangeDutyCycle(7.5)
            time.sleep(0.5)
            SERVO.ChangeDutyCycle(0)
            os.remove(sampleImg)
            s = False
