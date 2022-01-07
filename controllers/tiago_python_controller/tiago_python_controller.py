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
import time

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
        
        # Camera
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
        #self.inertialunit = self.getDeviceByIndex(42)
        
        #for i in range(self.getNumberOfDevices()):
        #    break
        #    print(i, self.getDeviceByIndex(i))
        

        # Head motors:
        #self.head_horizontal = self.getMotor("head_1_joint")
        #self.head_vertical = self.getMotor("head_2_joint")
        
        #self.horizontal_sensor = self.head_horizontal.getPositionSensor()
        #self.horizontal_sensor.enable(self.timeStep)
        #self.vertical_sensor = self.head_vertical.getPositionSensor()
        #self.vertical_sensor.enable(self.timeStep)
        
        # Wheel motors
        #self.wheel_left = self.getMotor("wheel_left_joint")
        #self.wheel_right = self.getMotor("wheel_right_joint")
        
        #self.wheel_left_sensor = self.wheel_left.getPositionSensor()
        #self.wheel_left_sensor.enable(self.timeStep)
        #self.wheel_right_sensor = self.wheel_right.getPositionSensor()
        #self.wheel_right_sensor.enable(self.timeStep)
        #print(self.inertialunit.SFRotation)
        
        #self.leftcamera = self.getDeviceByIndex(43) 
        #self.rightcamera = self.getDeviceByIndex(44)
        #print("left: ", self.leftcamera)
        #print("right:, ", self.rightcamera)
        
        #self.leftcamera.disable()
        #self.rightcamera.disable()

        #self.leftcamera.enable(self.timeStep)
        #self.rightcamera.enable(self.timeStep)
        
        # actuators:
        self.actuators = {
            'head_horizontal':{  # Head
                'motor': self.getDevice('head_1_joint'),
                'sensor': self.getDevice('head_1_joint_sensor')
            },
            'head_vertical':{
                'motor': self.getDevice('head_2_joint'),
                'sensor': self.getDevice('head_2_joint_sensor')
            },
            'wheel_left':{
                'motor': self.getDevice('wheel_left_joint'),
                'sensor': self.getDevice('wheel_left_joint_sensor')
            },
            'wheel_right':{
                'motor': self.getDevice('wheel_right_joint'),
                'sensor': self.getDevice('wheel_right_joint_sensor')
            },
            'left_hand_left_finger': {  # Hands
                'motor': self.getDevice('left_hand_gripper_left_finger_joint'),
                'sensor': self.getDevice('left_hand_gripper_left_finger_joint_sensor')
            },   
            'left_hand_right_finger': {
                'motor': self.getDevice('left_hand_gripper_right_finger_joint'),
                'sensor': self.getDevice('left_hand_gripper_right_finger_joint_sensor')
            },
            'right_hand_left_finger': {
                'motor': self.getDevice('right_hand_gripper_left_finger_joint'),
                'sensor': self.getDevice('right_hand_gripper_left_finger_joint_sensor')
            },
            'right_hand_right_finger': {
                'motor': self.getDevice('right_hand_gripper_right_finger_joint'),
                'sensor': self.getDevice('right_hand_gripper_right_finger_joint_sensor')
            },
            'left_elbow_bend': {  # elbows
                'motor': self.getDevice('arm_left_4_joint'),
                'sensor': self.getDevice('arm_left_4_joint_sensor')
            },         
            'left_elbow_roll': {
                'motor': None,
                'sensor': None
            },
            'right_elbow_bend': {
                'motor': None,
                'sensor': None
            },
            'right_elbow_roll': {
                'motor': None,
                'sensor': None
            },
            'left_wrist_bend': {  # wrists
                'motor': None,
                'sensor': None
            },         
            'left_wrist_roll': {
                'motor': self.getDevice('arm_left_7_joint'),
                'sensor': self.getDevice('arm_left_7_joint_sensor'),
            },
            'right_wrist_bend': {
                'motor': None,
                'sensor': None
            },
            'right_wrist_roll': {
                'motor': None,
                'sensor': None
            },
            'left_shoulder_yaw': {  # shoulders
                'motor': None,
                'sensor': None
            },       
            'left_shoulder_pitch': {
                'motor': self.getDevice('arm_left_2_joint'),
                'sensor': self.getDevice('arm_left_2_joint_sensor')
            },
            'left_shoulder_roll': {
                'motor': None,
                'sensor': None
            },
            'right_shoulder_yaw': {
                'motor': None,
                'sensor': None
            },
            'right_shoulder_pitch': {
                'motor': None,
                'sensor': None
            },
            'right_shoulder_roll': {
                'motor': None,
                'sensor': None
            }
        }
        # add extra values to actuator dictionary and enable sensors:
        for actuator in self.actuators.values():
            actuator['last_pos'] = 0
            actuator['moving'] = False
            if actuator['sensor']:
                actuator['sensor'].enable(self.timeStep)
            
            
        # checking routine:
        self.checks = {
            0: {
                'actuator': 'head_horizontal',
                'change': 0.1 
            },
            1: {
                'actuator': 'head_vertical',
                'change': -0.1 
            },
            2:{
                'actuator': 'head_vertical',
                'change': 0.1 
            },
            3:{
                'actuator': 'head_horizontal',
                'change': -0.1 
            },
            4:{
                'actuator': 'head_vertical',
                'change': 0.1 
            },
            5:{
                'actuator': 'head_vertical',
                'change': -0.1 
            }
        }
        
    def camera_read_external(self):
        img = []
        if self.ext_camera:
            # Capture frame-by-frame
            ret, frame = self.cameraExt.read()
            # Our operations on the frame come here
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # From openCV BGR to RGB
            img = cv2.resize(img, None, fx=0.5, fy=0.5) # image downsampled by 2
                        
        return img
 
    # Object manipulation:
                
    def grab(self, side):
        print('start grab')
        self.actuators[side+'_hand_left_finger']['motor'].setPosition(0)
        # self.actuators[side+'_hand_right_finger']['motor'].setPosition(0)
        l_pos = self.actuators[side+'_hand_left_finger']['sensor'].getValue()
        # r_pos = self.actuators[side+'_hand_right_finger']['sensor'].getValue()
        self.actuators[side+'_hand_left_finger']['last_pos'] = l_pos
        # self.actuators[side+'_hand_right_finger']['last_pos'] = r_pos
        time_steps = 0
        count_time_steps = False
        while self.step(self.timeStep) != -1: 
            if time_steps == 3:
                break
            if count_time_steps:
                time_steps += 1
            l_pos = self.actuators[side+'_hand_left_finger']['sensor'].getValue()
            # r_pos = self.actuators[side+'_hand_right_finger']['sensor'].getValue()
            l_pos_old = self.actuators[side+'_hand_left_finger']['last_pos']
            # r_pos_old = self.actuators[side+'_hand_right_finger']['last_pos']
            if  l_pos > l_pos_old: # or r_pos > r_pos_old:
                self.actuators[side+'_hand_left_finger']['motor'].setPosition(l_pos)
                # self.actuators[side+'_hand_right_finger']['motor'].setPosition(r_pos)
                print('done grab')
                count_time_steps = True
            self.actuators[side+'_hand_left_finger']['last_pos'] = l_pos
            # self.actuators[side+'_hand_right_finger']['last_pos'] = r_pos
    
    def release(self, side):
        self.actuators[side+'_hand_left_finger']['motor'].setPosition(0.045)
        self.actuators[side+'_hand_right_finger']['motor'].setPosition(0.045)
       
    def move_hand_to_coordinate(self, side, coordinate):
        pass
        
    def rotate_wrist(self, side):
        self.actuators[side+'_wrist_roll']['motor'].setPosition(-1.6)
        while self.step(self.timeStep) != -1: 
            if self.actuators[side+'_wrist_roll']['sensor'].getValue() <= -1.57:
                break
        
        
        
    def pickup_here(self, side):
        self.lower_arm(side)
        self.grab(side)
        print('grabbed')
        self.lift_arm(side)
        
    def putdown_here(self, side):
        self.lower_arm(side)
        self.release(side)
        self.lift_arm(side)
        
        
    def lower_arm(self, side):
        print('moving arm')
        self.actuators['left_elbow_bend']['motor'].setPosition(0.45)      # 55
        self.actuators['left_shoulder_pitch']['motor'].setPosition(0.55)  # 55
        while self.step(self.timeStep) != -1:
            if self.actuators['left_shoulder_pitch']['sensor'].getValue() >= 0.50:  # 50
                print('stop moving')
                break
                
    def lift_arm(self, side):
        print('moving arm')
        self.actuators['left_elbow_bend']['motor'].setPosition(0)
        self.actuators['left_shoulder_pitch']['motor'].setPosition(0)
        while self.step(self.timeStep) != -1:
            if self.actuators['left_elbow_bend']['sensor'].getValue() <= 0:
                print('stop moving')
                break
        
    def test_hands(self):    # find way to stop moving fingers when holding object!! preferably without hardcoding..
        for actuator in self.actuators.values():
            if actuator['sensor']:
                actuator['sensor'].enable(2)  # poll sensor every 2 ms
        self.release('left')
        self.rotate_wrist('left')
        print('rotated wrist')
        #self.lower_arm('left')
        
        #self.move_hand_to_coordinate('left', None)
        #self.grab('left')
        #while self.step(self.timeStep) != -1: 
           # for actuator in self.actuators.values():
               # if actuator['moving']:
                   # pos = actuator['sensor'].getValue()
                   # if abs(pos - actuator['last_pos']) <= 0.01:
                       # actuator['motor'].setPosition[pos]
                       # self.actuators['left_hand_left_finger']['motor'].setPosition[self.actuators['left_hand_left_finger']['sensor'].getValue()]
                       # self.actuators['left_hand_right_finger']['motor'].setPosition[self.actuators['left_hand_right_finger']['sensor'].getValue()]
                       # self.actuators['left_hand_left_finger']['moving'] = False
                       # self.actuators['left_hand_right_finger']['moving'] = False
                  # self.check_stop_condition[actuator]
                
            # if self.actuators['left_hand_left_finger']['sensor'].getValue() >= 0.045:
                # break
                
        # wait 2 seconds:
        #time.sleep(4)
                
        #self.grab('left')
        #time.sleep(2)
        #print('grabbed')
        #self.lift_arm('left')
        self.pickup_here('left')
        time.sleep(2)
        self.putdown_here('left')
        while self.step(self.timeStep) != -1: 
            pass
            
        
        
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
        cutoff = 250 #360
        middle = 250
        
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
                
                #If colour is in fov, get location of colour on the camera
                if m["m00"] != 0:
                    cx, cy = int(m["m10"] / m["m00"]), int(m["m01"] / m["m00"])
                    print(f"Found the {colour_name} box at: ", cx,cy)\
                    #Turn cx, cy into head positioning (cx, cy in the middle of the screen)
                    #left-down = 0, 500, right-down = 500, 500
                    #left-up should be 0, 0, right up should be 500, 0
                    
                    x_speed = abs(middle- cx)/middle
                    if not found:
                        if cx > cutoff:
                            input = -0.1
                        elif cx < cutoff:
                            input = 0.1
                        else:
                            input = 0
                        self.move_joint('head_horizontal', input*x_speed)    
                        
                        y_speed = abs(middle - cy)/middle
                        if cy > cutoff: 
                            input = -0.1
                        elif cy < cutoff:
                            input = 0.1
                        else:
                            input = 0
                        self.move_joint('head_vertical', input*y_speed)
                        if cx == cutoff and cy == cutoff:
                            found = True
                            self.actuators['head_horizontal']['motor'].setPosition(0)
                            self.actuators['head_horizontal']['motor'].setVelocity(0.15)
                            self.turn_to_can(cx, turn_speed = x_speed)
                            #break
                
                    if found:
                        test = self.turn_to_can(cx)
                        hor_pos = self.actuators['head_horizontal']['sensor'].getValue()  
                        print(hor_pos)                     
                        if 240 < cx < 260 and -0.1 < hor_pos < 0.1:
                            self.actuators['wheel_left']['motor'].setVelocity(0)
                            self.actuators['wheel_right']['motor'].setVelocity(0)
                            break
                    
                    # Need to find a better solution to ending this loop
                    # Want it to end when another button is pressed
                    #break
                
                #Searching for the colour
                else:
                    if checked < 6:
                        # joint, position, input = self.search_routine(checked)
                        check_stage = self.checks[checked]
                        checked += self.move_joint(check_stage['actuator'], check_stage['change'])
                    else:
                        print(f"{colour_name} not found")
                        break
                    
                        
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
            colour = [int(255*0.9),int(255*0.9),int(255*0.2)] 
        # BGR and RGB are switched around for some reason
        colour = cv2.cvtColor(np.uint8([[colour]] ), cv2.COLOR_BGR2HSV)[0][0]
        print("Colour: ", colour)
        print(f"Searching for {input}")
        self.get_colour_mask(colour, input)
        return colour
    
    def search_routine(self, checked):
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
        elif checked == 3:
            joint = self.head_horizontal
            position = self.horizontal_sensor.getValue()
            input = -0.1
        elif checked == 4:
            joint = self.head_vertical
            position = self.vertical_sensor.getValue()
            input = 0.1
        elif checked == 5:
            joint = self.head_vertical
            position = self.vertical_sensor.getValue()
            input = -0.1
        
        return joint, position, input
    
    def turn_to_can(self, cx, turn_speed = 0.1):
        print(cx)
        if cx > 270:
            self.turn_body(-1, turn_speed)
        elif cx < 230:
            self.turn_body(1, turn_speed)
        else:
            self.actuators['wheel_left']['motor'].setPosition(0)
            self.actuators['wheel_right']['motor'].setPosition(0)
        
    def move_joint(self, joint, input):
        motor = self.actuators[joint]['motor']
        sensor = self.actuators[joint]['sensor']
        if(motor.getMaxPosition() > (sensor.getValue() + input) > motor.getMinPosition()):
                    motor.setPosition(float(sensor.getValue() + input))
                    motor.setVelocity(1)
                    return 0
        else:
            return 1    
            
    def turn_body(self, input, turn_speed = 0.1):
        wheel_l = self.actuators['wheel_left']
        wheel_r = self.actuators['wheel_right']
        wheel_l['motor'].setVelocity(1)
        wheel_r['motor'].setVelocity(1)
        # Turn left
        if input > 0:
            wheel_l['motor'].setPosition(wheel_l['sensor'].getValue()-turn_speed)
            wheel_r['motor'].setPosition(wheel_r['sensor'].getValue()+turn_speed)
        # Turn right
        else:
            wheel_l['motor'].setPosition(wheel_l['sensor'].getValue()+turn_speed)
            wheel_r['motor'].setPosition(wheel_r['sensor'].getValue()-turn_speed)
        return
        
        
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
                self.move_joint('head_horizontal', 0.1)
                
            elif k == ord('D'):
                # Rotate to the right
                self.move_joint('head_horizontal', -0.1) 
            elif k == ord('W'):
                #Rotate upwards 
                self.move_joint('head_vertical', 0.1)
            elif k == ord('S'):
                #Rotate  downwards
                self.move_joint('head_vertical', -0.1)
            
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
                print("Min-pos: ", self.actuators['wheel_left']['motor'].getMinPosition())
                print("Max-pos: ", self.actuators['wheel_left']['motor'].getMaxPosition())
                print("Acceleration: ", self.actuators['wheel_left']['motor'].getAcceleration())
                print("Velocity: ", self.actuators['wheel_left']['motor'].getVelocity())
                print("Max-Velocity: ", self.actuators['wheel_left']['motor'].getMaxVelocity())
                print("Left Wheel Position: ", self.actuators['wheel_left']['sensor'].getValue())
                print("Right Wheel Position: ", self.actuators['wheel_right']['sensor'].getValue())
                print("Einde info")
                
            elif k == ord("U"):
                print("Head joint horizontal: ", self.actuators['head_horizontal']['sensor'].getValue())
                print("Head joint vertical: ", self.actuators['head_vertical']['sensor'].getValue())
                
            elif k == ord("L"):
                print("Stopping")
                self.wheel_left.setPosition(0)
                self.wheel_right.setPosition(0)
                self.wheel_left.setVelocity(0)
                self.wheel_right.setVelocity(0)

            
            if self.step(self.timeStep) == -1:
                break
                


robot = MyRobot(ext_camera_flag = False)
robot.run_keyboard()
#robot.test_hands()