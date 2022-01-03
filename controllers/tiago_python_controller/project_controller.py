from controller import Robot, Keyboard, Display, Motion, Camera
import cv2
import io
from PIL import Image
import warnings

import numpy as np
import math

#import torch
#import torch.nn as nn
#from torch.autograd import Variable
#from torch.distributions import constraints, multivariate_normal
#from torch.distributions.distribution import Distribution

# Turn off warnings
# warnings.filterwarnings('ignore')

class MyRobot(Robot):
    def __init__(self):
        super(MyRobot, self).__init__()
        print('> Starting robot controller')
        
        self.timeStep = int(self.getBasicTimeStep()) #32 # Milisecs to process the data (loop frequency) - Use int(self.getBasicTimeStep()) for default
        self.state = 0 # Idle starts for selecting different states
        
        self.step(self.timeStep) # Execute one step to get the initial position
                
        # Keyboard
        self.keyboard.enable(self.timeStep)
        self.keyboard = self.getKeyboard()
        
        for i in range(self.getNumberOfDevices()):
            break
            print(i, self.getDeviceByIndex(i))
        
        #Singular camera 
        self.camera = self.getCamera('camera')
        self.camera.enable(self.timeStep)

        # Head motors:
        self.head_horizontal = self.getMotor("head_1_joint")
        self.head_vertical = self.getMotor("head_2_joint")
        
        self.horizontal_sensor = self.head_horizontal.getPositionSensor()
        self.horizontal_sensor.enable(self.timeStep)
        self.vertical_sensor = self.head_vertical.getPositionSensor()
        self.vertical_sensor.enable(self.timeStep)
        
        # Wheel motors
        self.wheel_left = self.getMotor("wheel_left_joint")
        self.wheel_right = self.getMotor("wheel_right_joint")
        
        self.wheel_left_sensor = self.wheel_left.getPositionSensor()
        self.wheel_left_sensor.enable(self.timeStep)
        self.wheel_right_sensor = self.wheel_right.getPositionSensor()
        self.wheel_right_sensor.enable(self.timeStep)
        
        self.qrDecoder = cv2.QRCodeDetector()
        
    def printHelp(self):
        print(
            'Commands:\n'
            ' H for displaying the commands\n'
            ' UP for green\n'
            ' Down for pink (?)\n'
            ' Left for blue\n'
            ' Right for red\n'
            ' O for turning left\n'
            ' P for turning right\n'
            ' I for information on wheels\n'
            ' L for stopping turning\n'
            ' U for positional values of the head\n'
            ' W for turning the head upwards\n'
            ' A for turning the head to the left\n'
            ' S for turning the head downwards\n'
            ' D for turning the head to the right\n'
        )
        
    def get_colour_mask(self, colour, colour_name):
        # Mask possibilities: red, blue, green, black
        # colour is a list with bgr values [b, g, r]
        
        self.camera.enable(self.timeStep)
        checked = 0
        found = False
        cutoff = 250
        middle = 250
        self.return_to_position(velocity = 0.15, v_position = -0.3)
        
        while self.step(self.timeStep) != -1: 
            #40 is chosen as +- cutoff, can change
            colour_HSV = colour
            boundary = 40
            lower = np.array([colour_HSV[0] - boundary, colour_HSV[1] - boundary, colour_HSV[2] - boundary])
            upper = np.array([colour_HSV[0] + boundary, colour_HSV[1] + boundary, colour_HSV[2] + boundary])
            
            while self.step(self.timeStep) != -1:
                #Credits to Niels Cornelissen on the discord
                img = self.get_camera_image()
                
                mask = cv2.inRange(img, lower, upper)
                m = cv2.moments(mask)
                
                self.turn_body(1)
                #If colour is in fov, get location of colour on the camera
                if m["m00"] != 0:
                    cx, cy = int(m["m10"] / m["m00"]), int(m["m01"] / m["m00"])
                    print(f"Found the {colour_name} box at: ", cx,cy)\
                    #Turn cx, cy into head positioning (cx, cy in the middle of the screen)
                    #left-down = 0, 500, right-down = 500, 500
                    #left-up should be 0, 0, right up should be 500, 0
                    
                    hori = self.head_horizontal
                    vert = self.head_vertical
                    found = True                    
                    x_speed = abs(middle- cx)/middle
#                    if not found:
#                        position_h = self.horizontal_sensor.getValue()
#                        position_v = self.vertical_sensor.getValue()
#                        if cx > cutoff:
#                            input = -0.1
#                        elif cx < cutoff:
#                            input = 0.1
#                        else:
#                            input = 0
#                        self.move_joint(hori, position_h, input*x_speed)    
#                        
#                        y_speed = abs(middle - cy)/middle
#                        if cy > cutoff: 
#                            input = -0.1
#                        elif cy < cutoff:
#                            input = 0.1
#                        else:
#                            input = 0
#                        self.move_joint(vert, position_v, input*y_speed)
#                        if cx == cutoff and cy == cutoff:
#                            found = True
#                            self.return_to_position(velocity = 0.15)
#                            self.turn_to_can(cx, turn_speed = x_speed)
#                            #break

                    self.turn_to_can(cx, turn_speed = x_speed)  
                                           
                    #if 245 < cx < 255 and -0.1 < self.horizontal_sensor.getValue() < 0.1:
                    if cx == 250 and -0.1 < self.horizontal_sensor.getValue() < 0.1:
                        self.wheel_left.setVelocity(0)
                        self.wheel_right.setVelocity(0)
                        break
                    
                    # Need to find a better solution to ending this loop
                    # Want it to end when another button is pressed
                    #break
                
                #Searching for the colour
#                else:
#                    if checked < 7:
#                        joint, position, input, mid_pos = self.search_routine(checked)
#                        checked += self.turn_head(joint, position, input, mid_pos)
#                    else:
#                        print(f"{colour_name} not found")
#                        self.return_to_position()
#                        break
                        
            break
    
    def get_camera_image(self):
        #Credits to Niels Cornelissen on the discord
        w = self.camera.getWidth()
        h = self.camera.getHeight()
        img = self.camera.getImage()
        img = Image.frombytes('RGBA', (w, h), img, 'raw', 'BGRA') 
        img = np.array(img.convert('RGB'))
        img = cv2.resize(img, None, fx=2, fy=2)
        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        return img
    
    
    def choose_colour(self, input):
    # choose which colour, which then is set in BGR
        if input == 'red':
            colour = [255,0,0]
        elif input == 'blue':
            colour = [0,0,255] # [207, 98, 100]
        elif input == 'green':
            colour =  [0,255,0] # [65, 229, 158] 
        elif input == 'yellow':
            colour = [int(255*0.8),int(255*0.7),int(255*0.1)] 
        # BGR and RGB are switched around for some reason
        colour = cv2.cvtColor(np.uint8([[colour]] ), cv2.COLOR_BGR2HSV)[0][0]
        print("Colour: ", colour)
        print(f"Searching for {input}")
        self.get_colour_mask(colour, input)
        return colour
    
    def return_to_position(self, velocity = 1, v_position = -0.12):
        self.head_horizontal.setPosition(0)
        self.head_horizontal.setVelocity(velocity)
        
        self.head_vertical.setPosition(v_position)
        self.head_vertical.setVelocity(velocity)
    
    def search_routine(self, checked):
        middle = False
        if checked == 0:
            joint = self.head_horizontal
            position = self.horizontal_sensor.getValue()
            input = 0.1     
        elif checked == 1:
            joint = self.head_vertical
            position = self.vertical_sensor.getValue()
            input = -0.1
        elif checked == 2:
            joint = self.head_vertical
            position = self.vertical_sensor.getValue()
            input = 0.1
        elif checked == 3: #Back to middle
            joint = self.head_vertical
            position = self.vertical_sensor.getValue()
            input = -0.1
            middle = True   
        elif checked == 4:
            joint = self.head_horizontal
            position = self.horizontal_sensor.getValue()
            input = -0.1
        elif checked == 5:
            joint = self.head_vertical
            position = self.vertical_sensor.getValue()
            input = 0.1
        elif checked == 6: #Back to middle
            joint = self.head_vertical
            position = self.vertical_sensor.getValue()
            input = -0.1
            middle = True 
        return joint, position, input, middle
    
    def turn_to_can(self, cx, turn_speed = 0.1):
        print(cx)
        if cx > 270:
            self.turn_body(-1, turn_speed)
        elif cx < 230:
            self.turn_body(1, turn_speed)
        else:
            self.wheel_left.setPosition(0)
            self.wheel_right.setPosition(0)
        
    def move_joint(self, joint, position, input):
        if( joint.getMaxPosition() > (position + input) > joint.getMinPosition()):
                    joint.setPosition(float(position + input))
                    joint.setVelocity(1)
                    return 0
        else:
            return 1    
            
    def turn_head(self, joint, position, input, middle = False):
        mid_position = (abs(joint.getMaxPosition()) + abs(joint.getMinPosition()))/2
        mid_position = joint.getMaxPosition() - mid_position
        if middle:
            mid_position -= 0.15
            
            if mid_position < (position + input):
                joint.setPosition(float(position+input))
                joint.setVelocity(1)
                return 0
        elif( joint.getMaxPosition() > (position + input) > joint.getMinPosition()):
                    joint.setPosition(float(position + input))
                    joint.setVelocity(1)
                    return 0
        return 1  
           
     
    def turn_body(self, input, turn_speed = 0.1):
        self.wheel_left.setVelocity(1)
        self.wheel_right.setVelocity(1)
        # Turn left
        if input > 0:
            self.wheel_left.setPosition(self.wheel_left_sensor.getValue()-turn_speed)
            self.wheel_right.setPosition(self.wheel_right_sensor.getValue()+turn_speed)
        # Turn right
        else:
            self.wheel_left.setPosition(self.wheel_left_sensor.getValue()+turn_speed)
            self.wheel_right.setPosition(self.wheel_right_sensor.getValue()-turn_speed)
        return
    
    def look_for_QR(self):
        while self.step(self.timeStep) != -1:
            img = self.get_camera_image()
            data,bbox,rectifiedImage = self.qrDecoder.detectAndDecode(img)
            print(len(data))
            if len(data) == 0:
                self.return_to_position(v_position = 0)
                self.turn_body(1)
            else:
                print("Found QR-code. Tiago should try to position itself towards the")
                print("middle of the code, and then move towards the person.")
                break
    
        
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
                print("yellow: ", self.choose_colour('yellow'))
            elif k == Keyboard.LEFT:
                print('blue: ', self.choose_colour('blue'))
            elif k == Keyboard.RIGHT:
                print('red: ', self.choose_colour('red'))
            
            #Rotate head up (W), left (A), down (S), right (D)
            elif k == ord('A'):
                # Rotate to the left
                self.move_joint(self.head_horizontal, self.horizontal_sensor.getValue(), 0.1)
                
            elif k == ord('D'):
                # Rotate to the right
                self.move_joint(self.head_horizontal, self.horizontal_sensor.getValue(), -0.1) 
            elif k == ord('W'):
                #Rotate upwards 
                self.move_joint(self.head_vertical, self.vertical_sensor.getValue(), 0.1)
            elif k == ord('S'):
                #Rotate  downwards
                self.move_joint(self.head_vertical, self.vertical_sensor.getValue(), -0.1)
            
            # Something weird is happening here. They either both go to the right
            # or both go to the left.
            elif k == ord('O'):
                #Rotate body to the left
                print("Turning left")
                self.turn_body(1)
                
            elif k == ord('P'):
                #Rotate  body to the right
                print("Turning right") 
                self.turn_body(-1)
                           
            elif k == ord("I"):
                print("Info:")
                print("Min-pos: ", self.wheel_left.getMinPosition())
                print("Max-pos: ", self.wheel_left.getMaxPosition())
                print("Acceleration: ", self.wheel_left.getAcceleration())
                print("Velocity: ", self.wheel_left.getVelocity())
                print("Max-Velocity: ", self.wheel_left.getMaxVelocity())
                print("Left Wheel Position: ", self.wheel_left_sensor.getValue())
                print("Right Wheel Position: ", self.wheel_right_sensor.getValue())
                print("Einde info")
                
            elif k == ord("U"):
                print("Head joint horizontal: ", self.horizontal_sensor.getValue())
                print("Head joint vertical: ", self.vertical_sensor.getValue())
                
            elif k == ord("L"):
                print("Stopping")
                self.wheel_left.setPosition(0)
                self.wheel_right.setPosition(0)
                self.wheel_left.setVelocity(0)
                self.wheel_right.setVelocity(0)

            elif k == ord("B"):
                print("looking for QR code")
                self.look_for_QR()
            
            if self.step(self.timeStep) == -1:
                break
                

robot = MyRobot()
robot.run_keyboard()
