from controller import Robot, Keyboard, Display, Motion, Camera
import cv2
import io
from PIL import Image
import warnings

import numpy as np
import math

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.distributions import constraints, multivariate_normal
from torch.distributions.distribution import Distribution

# Turn off warnings
# warnings.filterwarnings('ignore')

class MyRobot(Robot):
    def __init__(self, ext_camera_flag):
        super(MyRobot, self).__init__()
        print('> Starting robot controller')
        
        self.timeStep = int(self.getBasicTimeStep()) #32 # Milisecs to process the data (loop frequency) - Use int(self.getBasicTimeStep()) for default
        self.state = 0 # Idle starts for selecting different states
        
        # Sensors init
        
        self.step(self.timeStep) # Execute one step to get the initial position
        #print("Self.step: ", self.step)
        self.ext_camera = ext_camera_flag        
        #self.displayCamExt = self.getDisplay('CameraExt')
        
        #external camera
        #if self.ext_camera:
        #    self.cameraExt = cv2.VideoCapture(0)
                
        # Keyboard
        self.keyboard.enable(self.timeStep)
        self.keyboard = self.getKeyboard()
        
        # Camera of Tiago (currently not working)
        self.camera = self.getDevice('camera')
        self.camera.enable(self.timeStep)
        print("Camera: ", self.camera)
        
        # Left shoulder and elbow of Nao        
        #self.leftShoulder = self.getMotor('LShoulderPitch')
        #self.leftShoulderRoll = self.getMotor('LShoulderRoll')
        #self.leftElbow = self.getMotor('LElbowRoll')
        #self.head_yaw = self.getDevice(12)
        
        for i in range(self.getNumberOfDevices()):
            #break
            print(i, self.getDeviceByIndex(i))
        
        #Cameras from external source (MultiSenseS21)
        self.inertialunit = self.getDeviceByIndex(42)
        #print(self.inertialunit.SFRotation)
        
        #self.leftcamera = self.getDeviceByIndex(43) 
        #self.rightcamera = self.getDeviceByIndex(44)
        #print("left: ", self.leftcamera)
        #print("right:, ", self.rightcamera)
        
        #self.leftcamera.disable()
        #self.rightcamera.disable()

        #self.leftcamera.enable(self.timeStep)
       # self.rightcamera.enable(self.timeStep)
        

        
    def camera_read_external(self):
        img = []
        if self.ext_camera:
            # Capture frame-by-frame
            ret, frame = self.cameraExt.read()
            # Our operations on the frame come here
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # From openCV BGR to RGB
            img = cv2.resize(img, None, fx=0.5, fy=0.5) # image downsampled by 2
                        
        return img
            
    # Displays the image on the webots camera display interface
    def image_to_display(self, img):
        if self.ext_camera:
            height, width, channels = img.shape
            imageRef = self.displayCamExt.imageNew(cv2.transpose(img).tolist(), Display.RGB, width, height)
            self.displayCamExt.imagePaste(imageRef, 0, 0)
        
    def printHelp(self):
        print(
            'Commands:\n'
            ' H for displaying the commands\n'
            ' UP for green\n'
            ' Down for black\n'
            ' Left for blue\n'
            ' Right for red'
        )
        
    def get_colour_mask(self, colour):
        # Mask possibilities: red, blue, green, black
        # colour is a list with bgr values [b, g, r]
                
        #use the camera
        self.camera = self.leftcamera
        
        while self.step(self.timeStep) != -1:       
            #Credits to Niels Cornelissen on the discord
            w = self.camera.getWidth()
            h = self.camera.getHeight()
            img = self.camera.getImage()
            img = Image.frombytes('RGBA', (w, h), img, 'raw', 'BGRA')
            img = np.array(img.convert('RGB'))
            img = cv2.resize(img, None, fx=2, fy=2)
            self.image_to_display(img)
            
            img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            bgr = colour
            green_HSV = cv2.cvtColor(np.uint8([[bgr]] ), cv2.COLOR_BGR2HSV)[0][0]

            #40 is chosen as +- cutoff, can change
            lower = np.array([green_HSV[0] - 40, green_HSV[1] - 40, green_HSV[2] - 40])
            upper = np.array([green_HSV[0] + 40, green_HSV[1] + 40, green_HSV[2] + 40])

            mask = cv2.inRange(img, lower, upper)

            m = cv2.moments(mask)
            #If colour is in fov, get location of colour on the camera
            if m["m00"] != 0:
                cx, cy = int(m["m10"] / m["m00"]), int(m["m01"] / m["m00"])
                print(cx,cy)
                # Need to find a better solution to ending this loop
                # Want it to end when another button is pressed
                break
        # Disable the camera after use     
        self.camera.disable()
    
    def choose_colour(self, input):
    # choose which colour, which then is set in BGR
        if input == 'red':
            colour = [0, 0, 255]
        elif input == 'blue':
            colour = [255, 0, 0]
        elif input == 'green':
            colour = [65, 229, 158] #[0, 255, 0]?
        elif input == 'black':
            colour = [0, 0, 0]   
        
        self.get_colour_mask(colour)
        return colour
        
    def run_keyboard(self):
        dict = {'red': [0, 0, 255]}
        # Main loop.
        while True:
            # Deal with the pressed keyboard key.
            k = self.keyboard.getKey()
            message = ''
            if k == ord('H'):
                self.printHelp()
            elif k == ord('C'):
                self.use_camera()
            elif k == Keyboard.UP:
                print("green: ", self.choose_colour('green'))
            elif k == Keyboard.DOWN:
                print("black: ", self.choose_colour('black'))
            elif k == Keyboard.LEFT:
                print('blue: ', self.choose_colour('blue'))
            elif k == Keyboard.RIGHT:
                print('red: ', self.choose_colour('red'))
            
            if self.step(self.timeStep) == -1:
                break
                

robot = MyRobot(ext_camera_flag = False)
robot.run_keyboard()