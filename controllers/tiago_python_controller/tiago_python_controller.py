import cv2
import io
import warnings
import numpy as np
import math
import time

from PIL import Image
from simple_pid import PID
from controller import Robot, Keyboard, Display, Motion, Camera

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.distributions import constraints, multivariate_normal
from torch.distributions.distribution import Distribution

# turn off warnings
# warnings.filterwarnings('ignore')

class MyRobot(Robot):
    def __init__(self, ext_camera_flag):
        super(MyRobot, self).__init__()
        print('> Starting robot controller')
        
        self.timeStep = int(self.getBasicTimeStep())
        self.state = 0 # idle starts for selecting different states
        
        # execute one step to get the initial position
        self.step(self.timeStep)       

        # keyboard
        self.keyboard.enable(self.timeStep)
        self.keyboard = self.getKeyboard()
        
        # camera
        self.ext_camera = ext_camera_flag
        self.camera = self.getDevice('camera')
        self.camera.enable(self.timeStep)
        self.camera_size = 1024

        # actuators:
        self.actuators = {
            # head
            'head_horizontal':{  
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
            # hands
            'left_hand_left_finger': {  
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
            # shoulders
            'left_shoulder_yaw': { 
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
            # elbows
            'left_elbow_bend': {  
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
            # wrists
            'left_wrist_bend': {  
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

        # add extra values to actuator dictionary and enable sensors
        for actuator in self.actuators.values():
            actuator['last_pos'] = 0
            actuator['moving'] = False
            if actuator['sensor']:
                actuator['sensor'].enable(self.timeStep)
            
        # set rotational velocity of the wheels
        self.actuators['left_wheel']['motor'].setVelocity(1)
        self.actuators['right_wheel']['motor'].setVelocity(1)
        
        # colours of pill bottles to search for
        self.pill_colours = {    
        'red': [255,0,0],
        'green': [127,255,0],
        'blue': [0,255,255],
        'purple': [127, 0, 255],
        }

        # QR decoder
        self.qrDecoder = cv2.QRCodeDetector()



# ----------------- MENU RELATED FUNCTIONS -----------------
    
    def run_keyboard(self): 
        # fold in arms
        self.fold_arm_in('left')
        self.fold_arm_in('right')

        # print greeting
        print('Tiago: Tiago at your service! do you want me to find anything for you?')
            
        # main loop
        while True:
            # deal with the pressed keyboard key
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

            # rotate head up (W), left (A), down (S), right (D)
            elif k == ord('A'):
                # rotate to the left
                self.move_joint('head_horizontal', 0.1)
            elif k == ord('D'):
                # rotate to the right
                self.move_joint('head_horizontal', -0.1) 
            elif k == ord('W'):
                # rotate upwards 
                self.move_joint('head_vertical', 0.1)
            elif k == ord('S'):
                # rotate downwards
                self.move_joint('head_vertical', -0.1)
            
            # turn body left or right
            elif k == ord('O'):
                # rotate body to the left
                print("Turning left")
                self.turn_body(1)  
            elif k == ord('P'):
                # rotate  body to the right
                print("Turning right") 
                self.turn_body(-1)

            # print information              
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
                self.stop_actuators(['left_wheel', 'right_wheel'])
                
            # search QR code
            elif k == ord("B"):
                print("Looking for QR code")
                self.return_to_human('red', 'left')
            
            if self.step(self.timeStep) == -1:
                break


    # printing help for the user
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


# ----------------- TASK RELATED FUNCTIONS -----------------

    # perform all necessary actions to grab the pill bottle and return it to the humand
    def grab_bottle(self, colour_name):
        print(f'Human: Tiago, can you get me the {colour_name} pills?')
        # convert colour name to HSV value
        colour = self.choose_colour(colour_name)
        boundary = 40  # colour boundary
        lower = np.array([colour[0] - boundary, colour[1] - boundary, colour[2] - 2*boundary])
        upper = np.array([colour[0] + boundary, colour[1] + boundary, colour[2] + 2*boundary])
        
        # look for bottle
        arm_side = 'left'
        if not self.search_bottle(arm_side, lower, upper):
            return False      
        
        # grab the bottle
        success = self.grab(arm_side)

        # back up for 6 seconds
        backup_counter = 1000 / self.timeStep * 6  
        while self.step(self.timeStep) != -1:
            self.move_wheels(-1, -1)
            backup_counter -= 1
            if backup_counter == 0:
                self.stop_actuators(['left_wheel', 'right_wheel'])
                break
        
        
        # return bottle to human
        if success:
            self.return_to_human(colour_name, arm_side)
        else:
            self.bend_wrist_out(arm_side)
            print('Tiago: Do you want me to try again, or do you want me to find different pills?')
            return False


    # look around and locate the pill bottle of the specified colour (given in lower and upper colour bounds)
    def search_bottle(self, arm_side, lower, upper):
        # retrieve relevant actuators
        head_yaw = self.actuators['head_vertical']
        
        # move head down
        head_angle = -0.4
        head_yaw['motor'].setPosition(head_angle)
        
        # rotate around and if not found return False
        if not self.rotate_to_bottle(lower, upper):
            return False
        
        # if found, start centering procedure in different function
        self.drive_to_target(1.2, self.extract_bottle_location, self.check_head_pitch, dxdy_func_params=(lower, upper), check_func_params=-0.5)
        self.fold_arm_out(arm_side)
        self.drive_to_target(0.8, self.extract_bottle_location, self.check_head_pitch, dxdy_func_params=(lower, upper), check_func_params=-0.74)
        return True


    # move arm, wrist and hand in correct position and grab + lift pill bottle
    def grab(self, side):
        # rotate wrist to vertical position
        self.rotate_wrist(side)
        # bend wrist to 90 degrees;

        # open hand
        self.open_hand(side)

        # lower arm
        self.lower_arm(side)

        # move shoulder out
        self.adjust_joint(f'{side}_shoulder_yaw', -0.2)

        # adjust elbow/wrist 
        self.bend_wrist_in(side)

        # move shoulder in
        self.adjust_joint(f'{side}_shoulder_yaw', 0.2)
        
        # close hand
        self.close_hand(side)
        
        # lift up arm
        self.lift_arm(side)

        # check if pickup was successful
        return self.check_failed_pickup(side)


    # drive to human and give pill bottle
    def return_to_human(self, colour_name, arm_side):
        
        # turn around until QR code is found, then drive to human
        found = False
        while not found:
            self.look_for_QR()
            self.drive_to_target(1, self.qr_detector, self.check_qr_distance, check_func_params=0.3)
            found = self.drive_to_target(2, self.qr_detector, self.check_qr_distance, check_func_params=0.45)
        
        # robot is at the human, ready to give the pills
        self.give_to_human(colour_name, arm_side)


    # turn around and find pills
    def rotate_to_bottle(self, lower, upper):
        count_down = 1000 / self.timeStep * 30    # 30 seconds
        while self.step(self.timeStep) != -1:
            # decrease timer
            count_down -= 1
            # move wheels
            self.move_wheels(1, -1)
            # extract potential bottle location
            pos = self.extract_bottle_location((lower, upper))
            cx, cy = self.get_error(pos, 1)
            if cx != 0 and cy != 0 and abs(cx) < 0.4:  # pills were found
                print('Tiago: I found the pills!')
                self.stop_actuators(['left_wheel', 'right_wheel'])
                return True
            if count_down == 0:
                print('Tiago: I did not find the pills you asked for, do you want me to find anything else?')
                return False                        


    # extact target location and move towards it
    def drive_to_target(self, speed, dxdy_func, check_func, dxdy_func_params=None, check_func_params=None):
        # initialize PID controller
        pid = PID(1, 0.1, 0.5, setpoint=0)
        # initialize glitch counter
        glitch_counter = 0
        # set large target value for wheels
        self.move_wheels(100000,100000) 
        while self.step(self.timeStep) != -1:
            # retrieve image position of target
            pos = dxdy_func(dxdy_func_params)
            # evaluate target
            result = check_func(pos, check_func_params)
            # handle evaluation
            if not result is None:
                self.set_wheel_speed()
                if result == False:   # we lost the target. Try again for max of 6 times before trying to find it again
                    if glitch_counter < 5:
                        glitch_counter += 1 
                        continue
                else:   # we reached our goal
                    self.stop_actuators(['left_wheel', 'right_wheel'])
                return result
            else:       # target was briefly lost
                glitch_counter = 0
            # convert target position to error
            dx, dy = self.get_error(pos[0:2], 1)
            # adjust head position
            self.move_head(0, -dy)
            # adjust wheel speeld
            d_speed = min(2*speed, 4*pid(dx))
            self.set_wheel_speed(l_speed=speed-d_speed, r_speed=speed+d_speed)


    # bend wrist to offer the pills to the human
    def give_to_human(self, colour_name, arm_side):
        self.bend_wrist_out(arm_side)
        print(f'Tiago: Here are the {colour_name} pills!')
            


    # ----------------- ROBOT RELATED FUNCTIONS -----------------

    # set head angle function
    def move_head(self, dx, dy):
        # get current angle
        pitch = self.actuators['head_vertical']
        yaw =  self.actuators['head_horizontal']
        pitch_angle = pitch['sensor'].getValue()
        yaw_angle = yaw['sensor'].getValue()
        
        # calculate new angle (within bounds)
        new_pitch_angle = min(max(pitch['motor'].getMinPosition(), pitch_angle+dy), pitch['motor'].getMaxPosition())
        new_yaw_angle = min(max(yaw['motor'].getMinPosition(), yaw_angle+dx), yaw['motor'].getMaxPosition())

        # send new angle to motor
        pitch['motor'].setPosition(new_pitch_angle)
        yaw['motor'].setPosition(new_yaw_angle)

    
    # see if head pitch is within wanted bounds
    def check_head_pitch(self, pos, params):
        target_angle = params
        if abs(self.actuators['head_vertical']['sensor'].getValue() - target_angle) < 0.01:
            return True
        return None


    # lift arm up, on the specified side
    def lift_arm(self, side):
        # set positions
        self.actuators[f'{side}_elbow_bend']['motor'].setPosition(0)
        self.actuators[f'{side}_shoulder_pitch']['motor'].setPosition(0)
        # move until positions are reached
        while self.step(self.timeStep) != -1:
            if self.actuators['left_elbow_bend']['sensor'].getValue() <= 0:
                break


    # lower arm down, on the specified side
    def lower_arm(self, side):
        # lower joint speeds for better precision
        self.change_actuator_speed([f'{side}_elbow_bend', f'{side}_shoulder_pitch'], 0.6)
        # set positions
        self.actuators[f'{side}_elbow_bend']['motor'].setPosition(0.45)
        self.actuators[f'{side}_shoulder_pitch']['motor'].setPosition(0.55)
        # move until positions are reached
        while self.step(self.timeStep) != -1:
            if self.actuators[f'{side}_shoulder_pitch']['sensor'].getValue() >= 0.50: 
                break
                

    # fold the arm in towards robot body, on the specified side
    def fold_arm_in(self, side):
        # retrieve actuators
        arm_actuators = [self.actuators[name] for name in [f'{side}_shoulder_yaw', f'{side}_shoulder_pitch', f'{side}_elbow_bend']]
        # set positions
        for actuator in arm_actuators:
            actuator['motor'].setPosition(actuator['motor'].getMaxPosition())
        # move until positions are reached
        while self.step(self.timeStep) != -1:
            for actuator in arm_actuators:
                if actuator['sensor'].getValue() < actuator['motor'].getMaxPosition():
                    continue
            return


    # fold the arm out away from robot body, on the specified side
    def fold_arm_out(self, side):
        # retrieve actuators
        arm_actuators = [self.actuators[name] for name in [f'{side}_shoulder_pitch', f'{side}_elbow_bend']]
        # set positions
        for actuator in arm_actuators:
            actuator['motor'].setPosition(0)
        # move until positions are reached
        while self.step(self.timeStep) != -1:
            for actuator in arm_actuators:
                if actuator['sensor'].getValue() > 0:
                    continue
            return
        

    # rotate the wrist, on the specified side
    def rotate_wrist(self, side):
        # retrieve actuator
        wrist = self.actuators[f'{side}_wrist_roll']
        # set position
        angle = -1.75
        wrist['motor'].setPosition(angle)
        # move until position is reached
        while self.step(self.timeStep) != -1:
            if wrist['sensor'].getValue() > 0.95*angle:
                continue
            return

    
    # bend wrist in, on the specified side
    def bend_wrist_in(self, side):
        # retrieve actuator
        wrist = self.actuators[f'{side}_wrist_bend']
        # set position
        angle = -1.34
        wrist['motor'].setPosition(angle)
        # move until position is reached
        while self.step(self.timeStep) != -1:
            if wrist['sensor'].getValue() > 0.95*angle:
                continue
            return


    # bend wrist out, on the specified side
    def bend_wrist_out(self, side):
        # retrieve actuator
        wrist = self.actuators[f'{side}_wrist_bend']
        # set position
        angle = 0
        wrist['motor'].setPosition(angle)
        # move until position is reached
        while self.step(self.timeStep) != -1:
            if abs(wrist['sensor'].getValue() - angle) < 0.01:
                return


    # open hand, on the specified side
    def open_hand(self, side):
        # retrieve actuators
        left_finger = self.actuators[f'{side}_hand_left_finger']
        right_finger = self.actuators[f'{side}_hand_right_finger']
        # set positions
        goal = 0.045
        left_finger['motor'].setPosition(goal)
        right_finger['motor'].setPosition(goal)
        # move until positions are reached
        while self.step(self.timeStep) != -1:
            if left_finger['sensor'].getValue() < 0.95*goal or right_finger['sensor'].getValue() < 0.95*goal:
                continue
            return


    # close hand, on the specified side
    def close_hand(self, side):
        print('Tiago: I will now grab the pills')
        # set positions and adjust
        self.adjust_joint(f'{side}_hand_right_finger', -0.003)
        self.actuators[f'{side}_hand_left_finger']['motor'].setPosition(0)
        l_pos = self.actuators[f'{side}_hand_left_finger']['sensor'].getValue()
        self.actuators[f'{side}_hand_left_finger']['last_pos'] = l_pos
        # close hand until no movement possible because object is grasped
        time_steps = 0
        count_time_steps = False
        while self.step(self.timeStep) != -1: 
            if time_steps == 5:  # 5
                break
            if count_time_steps:
                time_steps += 1
            l_pos = self.actuators[f'{side}_hand_left_finger']['sensor'].getValue()
            l_pos_old = self.actuators[f'{side}_hand_left_finger']['last_pos']
            if  l_pos > l_pos_old: # no movement possible anymore
                self.actuators[f'{side}_hand_left_finger']['motor'].setPosition(l_pos)
                count_time_steps = True
            self.actuators[f'{side}_hand_left_finger']['last_pos'] = l_pos

        # stop moving fingers
        self.stop_actuators([f'{side}_hand_left_finger', f'{side}_hand_right_finger'])
        
    # check for successful pickup
    def check_failed_pickup(self, side):
        r_finger = self.actuators[f'{side}_hand_right_finger']
        l_finger = self.actuators[f'{side}_hand_left_finger']
        # extract positions
        r_pos = r_finger['sensor'].getValue()
        l_pos = l_finger['sensor'].getValue()
        r_finger['motor'].setPosition(r_pos - 0.01)
        for i in range(10):
            self.step(self.timeStep)
        # extract new positions
        r_new_pos = r_finger['sensor'].getValue()
        l_new_pos = l_finger['sensor'].getValue()
        # stop movement
        r_finger['motor'].setPosition(r_new_pos)
        for i in range(3):
            self.step(self.timeStep)
        # check for movement
        if abs(r_new_pos - r_pos) > 0.005 and abs(l_pos - l_new_pos) < 0.00001:
            print('Tiago: Oh no, I dropped the pills. I\'m so clumsy!')
            return False
        else:
            print('Tiago: I grabbed your pills')
            return True
        

        


    # rotate body to input side (clockwise or counter clockwise), with certain turning speed
    def turn_body(self, input, turn_speed = 0.1):
        # retrieve actuators
        left_wheel = self.actuators['left_wheel']
        right_wheel = self.actuators['right_wheel']
        # set speed of motors
        left_wheel['motor'].setVelocity(1)
        right_wheel['motor'].setVelocity(1)
        # turn left
        if input > 0:
            left_wheel['motor'].setPosition(left_wheel['sensor'].getValue()-turn_speed)
            right_wheel['motor'].setPosition(right_wheel['sensor'].getValue()+turn_speed)
        # turn right
        else:
            left_wheel['motor'].setPosition(left_wheel['sensor'].getValue()+turn_speed)
            right_wheel['motor'].setPosition(right_wheel['sensor'].getValue()-turn_speed)
        return


    # move wheels in specified direction
    def move_wheels(self, left_dir, right_dir):
        # retrieve actuators
        left_wheel = self.actuators['left_wheel']
        right_wheel = self.actuators['right_wheel']
        # move to new position
        left_wheel['motor'].setPosition(left_wheel['sensor'].getValue() + left_dir)
        right_wheel['motor'].setPosition(right_wheel['sensor'].getValue() + right_dir)


    # change speed at which the wheels turn
    def set_wheel_speed(self, l_speed=1, r_speed=1):
        self.actuators['left_wheel']['motor'].setVelocity(max(0, l_speed))
        self.actuators['right_wheel']['motor'].setVelocity(max(0, r_speed))

    
    # move joint to specified position
    def move_joint(self, joint, input):
        # retrieve actuators
        motor = self.actuators[joint]['motor']
        sensor = self.actuators[joint]['sensor']
        # move to new position
        if(motor.getMaxPosition() > (sensor.getValue() + input) > motor.getMinPosition()):
                    motor.setPosition(float(sensor.getValue() + input))
                    return 0
        else:
            return 1    
    

    # change joint postion with specified input
    def adjust_joint(self, joint_name, input):
        # retrieve actuators
        joint = self.actuators[joint_name]
        last_pos = joint['sensor'].getValue()
        # check for boundaries + define goal
        goal = max(min((last_pos + input), joint['motor'].getMaxPosition()), joint['motor'].getMinPosition())
        joint['motor'].setPosition(goal)
        # move joint towards new postion
        while self.step(self.timeStep) != -1:
            if abs(joint['sensor'].getValue() - goal) < 0.01:
                return

    
    # stop the specified actuators at current postion
    def stop_actuators(self, actuator_names):
        for actuator in actuator_names:
            joint = self.actuators[actuator]
            if actuator in ['left_wheel', 'right_wheel']:
                joint['motor'].setPosition(joint['sensor'].getValue())
            else:
                joint['motor'].setPosition(max(min(joint['sensor'].getValue(), joint['motor'].getMaxPosition()), joint['motor'].getMinPosition()))
        self.step(self.timeStep)

    # change actuator speed for better precision:
    def change_actuator_speed(self, actuator_names, new_speed):
        for name in actuator_names:
            self.actuators[name]['motor'].setVelocity(new_speed)
               

# ----------------- OTHER FUNCTIONS -----------------   

    # bottle locator function
    def extract_bottle_location(self, bounds):  
        lower = bounds[0]
        upper = bounds[1]

        # convert image to numpy array
        img = self.get_camera_image()
        
        # create mask
        mask = cv2.inRange(img, lower, upper)
       
        # erosion + dilation
        kernel = np.ones((5,5), np.uint8)
        eroded_mask = cv2.erode(mask, kernel, iterations=1)
        dilated_mask = cv2.dilate(eroded_mask, kernel, iterations=1)

        # get contours
        contours, _ = cv2.findContours(dilated_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        # get and return center of mass
        if len(contours) > 0:
            moment = cv2.moments(contours[0])
            cx, cy = int(moment['m10']/ moment['m00']), int(moment['m01']/moment['m00'])
            return cx, cy
        else:
            return -1, -1


    # retrieve image seen in camera of robot
    def get_camera_image(self):
        # credits to Niels Cornelissen on the discord
        w = self.camera.getWidth()
        h = self.camera.getHeight()

        img = self.camera.getImage()
        img = Image.frombytes('RGBA', (w, h), img, 'raw', 'BGRA') 
        img = np.array(img.convert('RGB'))

        img = cv2.resize(img, None, fx=2, fy=2)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        return img

    
    # pick colour from known colours
    def choose_colour(self, input):
        # choose which colour to retrieve
        if input in self.pill_colours:
            colour = self.pill_colours[input]
        else:
            print(f'Tiago: I do not know the colour {input}, can you teach me?')
            return [-1,-1,-1]
            
        # convert RGB to HSV:
        # BGR and RGB are switched around for some reason
        colour = cv2.cvtColor(np.uint8([[colour]] ), cv2.COLOR_BGR2HSV)[0][0]
        print(f'Tiago: I will search for {input} pills now')
        return colour


    # get position error
    def get_error(self, pos, K):
        # calculate error between pos (the position) and image center
        if pos[0] >= 0:
            return K*((pos[0]/self.camera_size)-0.5), K*((pos[1]/self.camera_size)-0.5)
        else:
            return 0, 0


    # turn around to find a QR code
    def look_for_QR(self):
        head_position = self.actuators['head_vertical']['sensor'].getValue()
        self.adjust_joint('head_vertical', -head_position)
        while self.step(self.timeStep) != -1:
            self.move_wheels(-1, 1)
            pos = self.qr_detector()
            if not pos[2] is None:
                print("Tiago: I found you! I will bring the pills to you now")
                self.stop_actuators(['left_wheel', 'right_wheel'])
                break
        return 

    
    # finds QR code in image of camera
    def qr_detector(self, *args):
        img = self.get_camera_image()
        data, bbox, _ = self.qrDecoder.detectAndDecode(img)
        if bbox is None or len(bbox) == 0 or data != 'Hello :)':
            return [-1, -1, None]

        x = (bbox[0][1][0] + bbox[0][0][0]) / 2
        y = (bbox[0][1][1] + bbox[0][2][1]) / 2
        return [x, y, bbox[0]]
                
    
    # defines the distance between the robot and QR code
    def check_qr_distance(self, pos, target):
        bbox = pos[2]
        if bbox is None:
            return False   # signal we lost the qr code
        
        target_size = target*self.camera_size
        upper_left = bbox[0]
        bottom_right = bbox[2]
        if (bottom_right[0] - upper_left[0]) > target_size:
            return True   # signal that we are close to the human
        return None    # signal we still have the qr, but we are not close enough
 

robot = MyRobot(ext_camera_flag = False)
robot.run_keyboard()
