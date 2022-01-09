import cv2
import io
import warnings
import numpy as np
import math
import time

from PIL import Image
from controller import Robot, Keyboard, Display, Motion, Camera

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
        #self.left_wheel = self.getMotor("wheel_left_joint")
        #self.right_wheel = self.getMotor("wheel_right_joint")
        
        #self.left_wheel_sensor = self.left_wheel.getPositionSensor()
        #self.left_wheel_sensor.enable(self.timeStep)
        #self.right_wheel_sensor = self.right_wheel.getPositionSensor()
        #self.right_wheel_sensor.enable(self.timeStep)
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
            'left_wheel':{
                'motor': self.getDevice('wheel_left_joint'),
                'sensor': self.getDevice('wheel_left_joint_sensor')
            },
            'right_wheel':{
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
            'left_shoulder_yaw': {  # shoulders
                'motor': self.getDevice('arm_left_1_joint'),
                'sensor': self.getDevice('arm_left_1_joint_sensor')
            },       
            'left_shoulder_pitch': {
                'motor': self.getDevice('arm_left_2_joint'),
                'sensor': self.getDevice('arm_left_2_joint_sensor')
            },
            'left_shoulder_roll': {
                'motor': self.getDevice('arm_left_3_joint'),
                'sensor': self.getDevice('arm_left_3_joint_sensor')
            },
            'right_shoulder_yaw': {
                'motor': self.getDevice('arm_right_1_joint'),
                'sensor': self.getDevice('arm_right_1_joint_sensor')
            },       
            'right_shoulder_pitch': {
                'motor': self.getDevice('arm_right_2_joint'),
                'sensor': self.getDevice('arm_right_2_joint_sensor')
            },
            'right_shoulder_roll': {
                'motor': self.getDevice('arm_right_3_joint'),
                'sensor': self.getDevice('arm_right_3_joint_sensor')
            },
            'left_elbow_bend': {  # elbows
                'motor': self.getDevice('arm_left_4_joint'),
                'sensor': self.getDevice('arm_left_4_joint_sensor')
            },         
            'left_elbow_roll': {
                'motor': self.getDevice('arm_left_5_joint'),
                'sensor': self.getDevice('arm_left_5_joint_sensor')
            },
            'right_elbow_bend': {
                'motor': self.getDevice('arm_right_4_joint'),
                'sensor': self.getDevice('arm_right_4_joint_sensor')
            },
            'right_elbow_roll': {
                'motor': self.getDevice('arm_right_5_joint'),
                'sensor': self.getDevice('arm_right_5_joint_sensor')
            },
            'left_wrist_bend': {  # wrists
                'motor': self.getDevice('arm_left_6_joint'),
                'sensor': self.getDevice('arm_left_6_joint_sensor'),
            },         
            'left_wrist_roll': {
                'motor': self.getDevice('arm_left_7_joint'),
                'sensor': self.getDevice('arm_left_7_joint_sensor'),
            },
            'right_wrist_bend': {
                'motor': self.getDevice('arm_right_6_joint'),
                'sensor': self.getDevice('arm_right_6_joint_sensor'),
            },
            'right_wrist_roll': {
                'motor': self.getDevice('arm_right_7_joint'),
                'sensor': self.getDevice('arm_right_7_joint_sensor'),
            }
        }
        # add extra values to actuator dictionary and enable sensors:
        for actuator in self.actuators.values():
            actuator['last_pos'] = 0
            actuator['moving'] = False
            if actuator['sensor']:
                actuator['sensor'].enable(self.timeStep)
            
        # set rotational velocity of the wheels:
        self.actuators['left_wheel']['motor'].setVelocity(1)
        self.actuators['right_wheel']['motor'].setVelocity(1)
        
        self.pill_colours = {    
        'red': [255,0,0],
        'green': [127,255,0],
        'blue': [0,255,255],
        'purple': [127, 0, 255],
        }


    
    def run_keyboard(self):   # change to dictionary if time allows
        # fold in arms
        self.fold_arm_in('left')
        self.fold_arm_in('right')
        #self.drive_straight()
            
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
                self.grab_bottle('green')
            elif k == Keyboard.DOWN:
                self.grab_bottle('purple')
            elif k == Keyboard.LEFT:
                self.grab_bottle('blue')
            elif k == Keyboard.RIGHT:
                self.grab_bottle('red')

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
                print("Min-pos: ", self.actuators['left_wheel']['motor'].getMinPosition())
                print("Max-pos: ", self.actuators['left_wheel']['motor'].getMaxPosition())
                print("Acceleration: ", self.actuators['left_wheel']['motor'].getAcceleration())
                print("Velocity: ", self.actuators['left_wheel']['motor'].getVelocity())
                print("Max-Velocity: ", self.actuators['left_wheel']['motor'].getMaxVelocity())
                print("Left Wheel Position: ", self.actuators['left_wheel']['sensor'].getValue())
                print("Right Wheel Position: ", self.actuators['right_wheel']['sensor'].getValue())
                print("Einde info")
                
            elif k == ord("U"):
                print("Head joint horizontal: ", self.actuators['head_horizontal']['sensor'].getValue())
                print("Head joint vertical: ", self.actuators['head_vertical']['sensor'].getValue())
                
            elif k == ord("L"):
                print("Stopping")
                print("TO DO: CHANGE CODE")
                self.left_wheel.setPosition(0)
                self.right_wheel.setPosition(0)
                self.left_wheel.setVelocity(0)
                self.right_wheel.setVelocity(0)
                
            elif k == ord("B"):
                print("looking for QR code")
                self.look_for_QR()

            
            if self.step(self.timeStep) == -1:
                break


    def printHelp(self):
        print(
            'Commands:\n'
            ' H for displaying the commands\n'
            ' UP for green\n'
            ' Down for yellow\n'
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


    def grab_bottle(self, colour_name):
        # Look for bottle
        if not self.search_bottle(colour_name):
            return False
        
        arm_side = 'left'
        # when centered on bottle, stretch out left arm:
        self.fold_arm_out(arm_side)

        # grab the bottle:
        self.grab(arm_side, colour_name)
        
        # return bottle to human


    def search_bottle(self, colour_name):
        print('starting search for pills')
        # convert colour name to HSV value:
        colour = self.choose_colour(colour_name)
        boundary = 40  # colour boundary
        lower = np.array([colour[0] - boundary, colour[1] - boundary, colour[2] - 2*boundary])
        upper = np.array([colour[0] + boundary, colour[1] + boundary, colour[2] + 2*boundary])

        # retrieve relevant actuators:
        print('lowering head')
        head_yaw = self.actuators['head_vertical']
        
        # move head down
        head_angle = -0.4
        head_yaw['motor'].setPosition(head_angle)
        
        # rotate around and if not found return False
        if not self.rotate_to_bottle(lower, upper):
            return False
        
        # if found, start centering procedure in different function:
        self.center_bottle(colour, colour_name, lower, upper)

        return True


    def grab(self, side, colour_name):
        # rotate wrist to vertical position:
        self.rotate_wrist(side)
        print('rotated wrist')
        # bend wrist to 90 degrees;

        # open hand:
        self.open_hand(side)
        print('opened hand')

        # adjust elbow/wrist 
        self.bend_wrist_in(side)
        print('bent wrist')

        # lower arm:
        self.lower_arm(side)
        print('lowered arm')
        # move to grab bottle

        # close hand

        # lift up arm

        # fold in arm

        return


    def center_bottle(self, colour, colour_name, lower, upper):
        # event loop:

        yaw = self.actuators['head_horizontal']
        pitch = self.actuators['head_vertical']

        correct_pitch = -0.7

        while self.step(self.timeStep) != -1:
            # get ball position in image:
            pos = self.extract_bottle_location(lower, upper)
            # get errors:
            dx, dy = self.get_error(pos, 512, 512, 1)
            #print(f'dx: {dx}, dy: {dy}')
            self.move_head_angles(0, -dy)
            if dx > 0.01:
                self.move_wheels(3, 0)
            elif dx < 0.01:
                self.move_wheels(0, 3)
            elif dx < -0.2 or dx == 0 and dy == 0:
                self.move_wheels(-1, 1)
            else:
                self.move_wheels(3, 3)

            if abs(pitch['sensor'].getValue() - correct_pitch) < 0.05:
                self.stop_actuators(['left_wheel', 'right_wheel'])
                return

                  
            # move the head towards ball:
            

        # print('start grab')
        # self.actuators[side+'_hand_left_finger']['motor'].setPosition(0)
        # # self.actuators[side+'_hand_right_finger']['motor'].setPosition(0)
        # l_pos = self.actuators[side+'_hand_left_finger']['sensor'].getValue()
        # # r_pos = self.actuators[side+'_hand_right_finger']['sensor'].getValue()
        # self.actuators[side+'_hand_left_finger']['last_pos'] = l_pos
        # # self.actuators[side+'_hand_right_finger']['last_pos'] = r_pos
        # time_steps = 0
        # count_time_steps = False
        # while self.step(self.timeStep) != -1: 
        #     if time_steps == 3:
        #         break
        #     if count_time_steps:
        #         time_steps += 1
        #     l_pos = self.actuators[side+'_hand_left_finger']['sensor'].getValue()
        #     # r_pos = self.actuators[side+'_hand_right_finger']['sensor'].getValue()
        #     l_pos_old = self.actuators[side+'_hand_left_finger']['last_pos']
        #     # r_pos_old = self.actuators[side+'_hand_right_finger']['last_pos']
        #     if  l_pos > l_pos_old: # or r_pos > r_pos_old:
        #         self.actuators[side+'_hand_left_finger']['motor'].setPosition(l_pos)
        #         # self.actuators[side+'_hand_right_finger']['motor'].setPosition(r_pos)
        #         print('done grab')
        #         count_time_steps = True
        #     self.actuators[side+'_hand_left_finger']['last_pos'] = l_pos
        #     # self.actuators[side+'_hand_right_finger']['last_pos'] = r_pos

 
    def rotate_to_bottle(self, lower, upper):
        print('start rotation')
        count_down = 1000 / self.timeStep * 30    # 30 seconds
        while self.step(self.timeStep) != -1:
            # decrease timer:
            count_down -= 1
            # move wheels
            self.move_wheels(1, -1)
            # extract potential bottle location:
            pos = self.extract_bottle_location(lower, upper)
            cx, cy = self.get_error(pos, 512, 512, 1)
            if cx != 0 and cy != 0 and abs(cx) < 0.4:  # pills were found
                print('I found the pills!')
                self.stop_actuators(['left_wheel', 'right_wheel'])
                return True
            if count_down == 0:
                print('I did not find the pills you asked for, do you want me to find anything else?')
                return False
                
 
    # TO DO: verwijderen?   
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
                        



    # ----------------- ROBOT RELATED FUNCTIONS -----------------

    def pickup_here(self, side):
        self.lower_arm(side)
        self.grab(side)
        print('grabbed')
        self.lift_arm(side)
        

    def putdown_here(self, side):
        self.lower_arm(side)
        self.release(side)
        self.lift_arm(side)
        

    # set head angle function
    def move_head_angles(self, dx, dy):
        # get current angle:
        pitch = self.actuators['head_vertical']
        yaw =  self.actuators['head_horizontal']
        pitch_angle = pitch['sensor'].getValue()
        yaw_angle = yaw['sensor'].getValue()
        
        # calculate new angle (within bounds):
        new_pitch_angle = min(max(pitch['motor'].getMinPosition(), pitch_angle+dy), pitch['motor'].getMaxPosition())
        new_yaw_angle = min(max(yaw['motor'].getMinPosition(), yaw_angle+dx), yaw['motor'].getMaxPosition())

        # send new angle to motor:
        pitch['motor'].setPosition(new_pitch_angle)
        yaw['motor'].setPosition(new_yaw_angle)
              

    def rotate_wrist(self, side):
        wrist = self.actuators[f'{side}_wrist_roll']
        angle = -1.57
        wrist['motor'].setPosition(angle)
        while self.step(self.timeStep) != -1:
            if wrist['sensor'].getValue() > 0.95*angle:
                continue
            return


    def open_hand(self, side):
        left_finger = self.actuators[f'{side}_hand_left_finger']
        right_finger = self.actuators[f'{side}_hand_right_finger']
        goal = 0.045
        left_finger['motor'].setPosition(goal)
        right_finger['motor'].setPosition(goal)
        while self.step(self.timeStep) != -1:
            if left_finger['sensor'].getValue() < 0.95*goal or right_finger['sensor'].getValue() < 0.95*goal:
                continue
            return


    def swing_arm(self, side, angle):
        joint = self.actuators[f'{side}_shoulder_yaw']
        joint['motor'].setPosition(angle)
        joint['motor'].setVelocity(1)
        start_angle = joint['sensor'].getValue()
        direction = np.sign(angle - start_angle)
        while self.step(self.timeStep) != -1:
            joint_pos = joint['sensor'].getValue()
            if direction > 0:
                if joint_pos >= angle or joint_pos >= joint['motor'].getMaxPosition():
                    joint['motor'].setPosition(joint_pos)
                    joint['motor'].setVelocity(0)
                    break
            else:
                if joint_pos <= angle or joint_pos <= joint['motor'].getMinPosition():
                    joint['motor'].setPosition(joint_pos)
                    joint['motor'].setVelocity(0)
                    break
               
                    
    def drive_straight(self):
        print('driving')
        left_wheel = self.actuators['left_wheel']
        right_wheel = self.actuators['right_wheel']
        counter = 0
        left_wheel['motor'].setVelocity(1)
        right_wheel['motor'].setVelocity(1)
        while self.step(self.timeStep) != -1:
            print('straight')
            
            left_wheel['motor'].setPosition(left_wheel['sensor'].getValue()+1)
            right_wheel['motor'].setPosition(right_wheel['sensor'].getValue()+1)
            
            
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


    def stop_actuators(self, actuator_names):
        print('stopping movement')
        for actuator in actuator_names:
            self.actuators[actuator]['motor'].setPosition(self.actuators[actuator]['sensor'].getValue())
        self.step(self.timeStep)
        
        
    def turn_to_bottle(self, cx, turn_speed = 0.1):
        print(cx)
        if cx >= 250:
            self.turn_body(-1, turn_speed)
        elif cx < 250:
            self.turn_body(1, turn_speed)
        else:
            self.actuators['left_wheel']['motor'].setPosition(0)
            self.actuators['right_wheel']['motor'].setPosition(0)
        

    def move_joint(self, joint, input, middle=False):
        motor = self.actuators[joint]['motor']
        sensor = self.actuators[joint]['sensor']
        if(motor.getMaxPosition() > (sensor.getValue() + input) > motor.getMinPosition()):
                    motor.setPosition(float(sensor.getValue() + input))
                    motor.setVelocity(1)
                    return 0
        else:
            return 1    
            

    def turn_head(self, joint_name, input, middle = False):
        joint = self.actuators[joint_name]
        max_pos = joint['motor'].getMaxPosition()
        min_pos = joint['motor'].getMinPosition()
        position = joint['sensor'].getValue()
        mid_position = (abs(max_pos) + abs(min_pos))/2
        mid_position = max_pos - mid_position
        if middle:
            mid_position -= 0.15

            if mid_position < (position + input):
                joint['motor'].setPosition(float(position+input))
                joint['motor'].setVelocity(1)
                return 0
        elif( max_pos > (position + input) > min_pos):
                    joint['motor'].setPosition(float(position + input))
                    joint['motor'].setVelocity(1)
                    return 0
        return 1 


    def turn_body(self, input, turn_speed = 0.1):
        left_wheel = self.actuators['left_wheel']
        right_wheel = self.actuators['right_wheel']
        left_wheel['motor'].setVelocity(1)
        right_wheel['motor'].setVelocity(1)
        # Turn left
        if input > 0:
            left_wheel['motor'].setPosition(left_wheel['sensor'].getValue()-turn_speed)
            right_wheel['motor'].setPosition(right_wheel['sensor'].getValue()+turn_speed)
        # Turn right
        else:
            left_wheel['motor'].setPosition(left_wheel['sensor'].getValue()+turn_speed)
            right_wheel['motor'].setPosition(right_wheel['sensor'].getValue()-turn_speed)
        return


    def move_wheels(self, left_dir, right_dir):
        left_wheel = self.actuators['left_wheel']
        right_wheel = self.actuators['right_wheel']
        left_wheel['motor'].setPosition(left_wheel['sensor'].getValue() + left_dir)
        right_wheel['motor'].setPosition(right_wheel['sensor'].getValue() + right_dir)


    def fold_arm_in(self, side):
        # retrieve actuators
        arm_actuators = [self.actuators[name] for name in [f'{side}_shoulder_yaw', f'{side}_shoulder_pitch', f'{side}_elbow_bend']]
        for actuator in arm_actuators:
            actuator['motor'].setPosition(actuator['motor'].getMaxPosition())
        while self.step(self.timeStep) != -1:
            for actuator in arm_actuators:
                if actuator['sensor'].getValue() < actuator['motor'].getMaxPosition():
                    continue
            return

    def fold_arm_out(self, side):
        arm_actuators = [self.actuators[name] for name in [f'{side}_shoulder_pitch', f'{side}_elbow_bend']]
        for actuator in arm_actuators:
            actuator['motor'].setPosition(0)
        while self.step(self.timeStep) != -1:
            for actuator in arm_actuators:
                if actuator['sensor'].getValue() > 0:
                    continue
            return
        
    def bend_wrist_in(self, side):
        pass


# ----------------- OTHER FUNCTIONS -----------------

    def get_colour_mask(self, lower, upper): 
        #Credits to Niels Cornelissen on Discord
        img = self.get_camera_image()
        #hsvImg = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        hsvImg = img
        mask = cv2.inRange(hsvImg, lower, upper)
        m = cv2.moments(mask)
        return m
    

    def choose_colour(self, input):
        # choose which colour, which then is set in BGR
        if input in self.pill_colours:
            colour = self.pill_colours[input]
        else:
            print(f'I do not know the colour {input}, can you teach me?')
            return [-1,-1,-1]
            
        # Convert RGB to HSV:
        # BGR and RGB are switched around for some reason
        colour = cv2.cvtColor(np.uint8([[colour]] ), cv2.COLOR_BGR2HSV)[0][0]
        print("Colour: ", input)
        return colour


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
    
    
    # bottle locator function
    def extract_bottle_location(self, lower, upper):  
        # convert image to numpy array:
        img = self.get_camera_image()
        
        # create mask:
        mask = cv2.inRange(img, lower, upper)
       
        # erosion + dilation:
        kernel = np.ones((5,5), np.uint8)
        eroded_mask = cv2.erode(mask, kernel, iterations=1)
        dilated_mask = cv2.dilate(eroded_mask, kernel, iterations=1)

        # get contours:
        contours, hierarchy = cv2.findContours(dilated_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        # get and return center of mass:
        if len(contours) > 0:
            moment = cv2.moments(contours[0])
            cx, cy = int(moment['m10']/ moment['m00']), int(moment['m01']/moment['m00'])
            return cx, cy
        else:
            return -1, -1


    # ball position error function
    def get_error(self, pos, width, height, K):
        # calculate error between ball pos and image center:
        if pos[0] >= 0:
            return K*((pos[0]/width)-0.5), K*((pos[1]/height)-0.5)
        else:
            return 0, 0

        

        
    



robot = MyRobot(ext_camera_flag = False)
robot.run_keyboard()
#robot.test_hands()