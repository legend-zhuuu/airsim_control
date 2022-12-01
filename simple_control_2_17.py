import math
import pprint
import numpy as np
import os
import shutil
import cv2
import airsim
import time
from KeyController import KeyController
from obstacle_move import DroneController
import threading

# TIMEOUT
from airsim_functions.orbit import OrbitNavigator

TIMEOUT = 1200  # 20 miniuts
MIN_DEPTH_METERS = 0
MAX_DEPTH_METERS = 230

# Mesh ID's
OBSTACLE = 100
DRONE = 200

# Commands:
ARM = "arm"
DISARM = "disarm"
MOVE = "move"
MOVE_PATH = "moveonpath"
DRONE_MOVE = "dm"
HOME = "home"
STATE = "state"
TAKEOFF = "takeoff"
LANDING = "land"
STOP = "stop"
KEYBOARD_CONTROL = "kc"
ORBIT = "inspect"
FORWARD_FORCE = 1
BACKWARD_FORCE = -1
RIGHT_FORCE = 1
LEFT_FORCE = -1

#Savedir
DIR = r"C:\Users\1328301164\Documents"


class SimpleTerminalController:
    def __init__(self,
                 verbatim: bool = True,
                 maxmin_velocity: float = 10,
                 drive_type: airsim.DrivetrainType = airsim.DrivetrainType.ForwardOnly):
        # Should this class print to terminal
        self.verbatim = verbatim
        self.DriveType = drive_type
        # connect to the AirSim simulator
        self.client = airsim.MultirotorClient()
        self.confirm_connection()
        self.client.landAsync().join()
        # self.client.simEnableWeather(True)
        # self.client.simSetWeatherParameter(airsim.WeatherParameter.Rain, 0.1)
        # self.yaw_mode = airsim.YawMode(True, 0)
        # connect to the AirSim simulator
        self.client = airsim.MultirotorClient()

        self.imgclient = airsim.MultirotorClient()
        self.imgclient.confirmConnection()
        self.imgclient.enableApiControl(True)  # 获取控制权
        self.imgclient.armDisarm(True)
        self.img_callback_thread = threading.Thread(target=self.repeat_timer_img_callback,
                                                         args=(self.img_callback, 1), daemon=True)
        self.is_img_thread_active = False

        self.imu_existing_data_cleared = False
        self.count = 0

        self.confirm_connection()
        # Segmentation setup
        self.setup_segmentation_colors()

        # Movement and constraints:
        self.vx = 0
        self.vy = 0
        self.vz = 0
        self.height = -100
        self.yaw = 0
        self.nav = None
        self.imgcount = 0
        self.saveimgHz = 5
        self.maxmin_vel = maxmin_velocity
        # self.dc = DroneController("Drone2")
        self.dcrun = 0

    def confirm_connection(self):
        self.client.confirmConnection()
        self.client.enableApiControl(True)
        self.yaw_rate = 0
        self.nav = None

        # camera_pose = airsim.Pose(airsim.Vector3r(0, 0, 0), airsim.to_quaternion(-1, 0, -0.1))
        # self.client.simSetCameraPose("0", camera_pose)

    def confirm_connection(self):
        self.client.confirmConnection()
        self.client.enableApiControl(True, vehicle_name='Drone1')

    def repeat_timer_img_callback(self, task, period):
        while self.is_img_thread_active:
            task()
            self.count += 1
            if self.count % 10 == 0:
                print("********************save img count:", self.count, "********************")
            time.sleep(period)

    def img_callback(self):
        self.saveimg()

    def start_img_callback_thread(self):
        if not self.is_img_thread_active:
            print("daemon state:", self.img_callback_thread.isDaemon())
            self.is_img_thread_active = True
            self.img_callback_thread.start()
            print("Started img save callback thread")

    def stop_img_callback_thread(self):
        if self.is_img_thread_active:
            self.is_img_thread_active = False
            self.img_callback_thread.join()
            print("Stopped img save callback thread.")

    def setup_segmentation_colors(self):

        # Finding regexp GameObject name and set the ID
        # success = self.client.simSetSegmentationObjectID("SM_Floor20m[\w]*", 100, True)
        # print("Change of color =", success)

        # self.change_color("", OBSTACLE)
        self.change_color("DRONE", DRONE)

    def change_color(self, name, id):
        success = self.client.simSetSegmentationObjectID(name + r"[\w]*", id, True)
        print("Change of color on", name, "=", success)

    def takeoff(self):
        state = self.client.getMultirotorState()
        print("Takeoff received")
        if state.landed_state == airsim.LandedState.Landed:
            print("taking off...")
            self.client.takeoffAsync().join()
        else:
            self.client.hoverAsync().join()

    def land(self):
        state = self.client.getMultirotorState()
        print("Land received")
        if state.landed_state != airsim.LandedState.Landed:
            print("landing...")
            self.client.landAsync()

    def arm(self):
        print("Arm received")
        self.client.armDisarm(True, "Drone1")
        # self.client.armDisarm(True, "Drone2")

    def disarm(self):
        print("Disarm received")
        self.client.armDisarm(False)

    def move_to_position(self, args: list):
        print("Move received")
        if len(args) != 5:
            print("Move needs 5 args")
            return
        self.client.enableApiControl(True)
        print("Move args:", float(args[1]), float(args[2]), float(args[3]), float(args[4]))
        self.client.moveToPositionAsync(x=float(args[1]), y=float(args[2]), z=float(args[3]),
                                        velocity=float(args[4]), drivetrain=airsim.DrivetrainType.ForwardOnly,
                                        yaw_mode=airsim.YawMode(False, 0)).join()
        self.client.hoverAsync().join()
        print("Moved!")

    def move_on_path(self, args: list):
        print("MoveOnPath received")
        if len(args) % 3 != 2:
            print("Move needs 3 args per position args")
            return
        # Have to make sure it is enabled:
        self.client.enableApiControl(True, vehicle_name="Drone2")
        iterations = (len(args) - 2) / 3
        path = []
        for i in range(int(iterations)):
            point = airsim.Vector3r(float(args[(i * 3) + 1]),
                                    float(args[(i * 3) + 2]),
                                    float(args[(i * 3) + 3]))
            path.append(point)
            if self.verbatim:
                print("path point added", str(point))
        try:
            result = self.client.moveOnPathAsync(path, float(args[-1]), TIMEOUT,
                                                 airsim.DrivetrainType.ForwardOnly, airsim.YawMode(False, 0),
                                                 20,
                                                 1, vehicle_name="Drone2")
        except:
            errorType, value, traceback = airsim.sys.exc_info()
            print("moveOnPath threw exception: " + str(value))
            pass

        self.client.hoverAsync().join()
        print("Path moved!")

    def rect_path(self, round, length, height, radius, step=50):
        path = []
        for r in range(round):
            path_line1 = [airsim.Vector3r(i * length / step, -4 * radius * r, height) for i in range(step + 1)]
            path_round1 = [airsim.Vector3r(length + radius * (math.sin(theta * math.pi / step)),
                                           -4 * radius * r - (radius * (1 - math.cos(theta * math.pi / step))), height)
                           for theta in range(step)]
            path_line2 = [airsim.Vector3r(length - i * length / step, -4 * radius * r - 2 * radius, height) for i in
                          range(step + 1)]
            path_round2 = [airsim.Vector3r(-radius * (math.sin(theta * math.pi / step)),
                                           -4 * radius * r - 2 * radius - (
                                                       radius * (1 - math.cos(theta * math.pi / step))), height)
                           for theta in range(step)]
            path = path + path_line1 + path_round1 + path_line2 + path_round2
        path_line1 = [airsim.Vector3r(i * length / step, -4 * radius * round, height) for i in range(step + 1)]
        path_round1 = [airsim.Vector3r(length + radius * (math.sin(theta * math.pi / step)),
                                       -4 * radius * round - (radius * (1 - math.cos(theta * math.pi / step))), height)
                       for theta in range(step)]
        path_line2 = [airsim.Vector3r(length - i * length / step, -4 * radius * round - 2 * radius, height) for i in
                      range(step + 1)]
        path_round2 = [airsim.Vector3r(length + radius * (math.sin(theta * math.pi / step / 2)),
                                       -4 * radius * round + (radius * (1 - math.cos(theta * math.pi / step / 2))),
                                       height)
                       for theta in range(step)]
        start = (0, 0)
        end = (length + radius, -4 * radius * round + radius)
        path_back = [
            airsim.Vector3r((end[0] - start[0]) * (step - i) / step, (end[1] - start[1]) * (step - i) / step, height)
            for i in range(step + 1)]
        path = path + path_line1 + path_round2 + path_back
        return path

    def archi_path(self):
        N = 12
        theta = np.linspace(0, N * np.pi, 2000*N)
        a = 1
        b = 10
        x = (a * b * theta) * np.cos(theta)
        y = (a * b * theta) * np.sin(theta)
        start = [x[0], y[0]]
        end = [x[-1], y[-1]]
        dist = math.sqrt((start[0] - end[0]) ** 2 + (start[1] - end[1]) ** 2)
        theta = np.linspace(0, np.pi, 2000)
        x2 = dist / 2 + dist / 2 * np.cos(theta)
        y2 = dist / 2 * np.sin(theta)
        x = np.append(x, x2)
        y = np.append(y, y2)
        return [airsim.Vector3r(x[i], y[i], self.height) for i in range(len(x))]

    def print_path(self, path):
        self.client.simPlotLineList(path, color_rgba=[1, 0, 0, 1], is_persistent=True)

    def drone_move(self, args: list):
        print("MoveOnPath received")
        dronevel = 2
        step = 50
        radius = 25
        length = 400
        height = self.height
        round = 5
        path = []

        self.client.moveToPositionAsync(x=0, y=0, z=height, velocity=10, timeout_sec=30, vehicle_name="Drone1").join()
        self.start_img_callback_thread()
        airsim.time.sleep(1)

        path = self.rect_path(round=round, length=length, height=height, radius=radius)
        # path = self.archi_path()
        # self.print_path(path)
        # print("path_len:\n", len(path))
        self.client.moveOnPathAsync(path, velocity=dronevel).join()

        # self.client.moveByVelocityAsync(vx=0, vy=dronevel, vz=0, duration=20, vehicle_name="Drone1").join()
        # self.client.moveByVelocityAsync(vx=-dronevel, vy=0, vz=0, duration=20, vehicle_name="Drone1").join()
        # self.client.moveByVelocityAsync(vx=0, vy=dronevel, vz=0, duration=20, vehicle_name="Drone1").join()
        self.stop_img_callback_thread()

    def host_move(self, args: list):
        # Have to make sure it is enabled:
        self.client.enableApiControl(True, args[1])
        iterations = (len(args) - 3) / 3
        path = []
        for i in range(int(iterations)):
            point = airsim.Vector3r(float(args[(i * 3) + 2]),
                                    float(args[(i * 3) + 3]),
                                    float(args[(i * 3) + 4]))
            path.append(point)
            if self.verbatim:
                print("path point added", str(point))
        try:
            result = self.client.moveOnPathAsync(path, float(args[-1]), TIMEOUT,
                                                 airsim.DrivetrainType.ForwardOnly, airsim.YawMode(False, 0),
                                                 20,
                                                 1,
                                                 vehicle_name = args[1])

        except:
            errorType, value, traceback = airsim.sys.exc_info()
            print("moveOnPath threw exception: " + str(value))
            pass

    def track(self, velocity, time):
        kp = 0
        kv = 1
        drone1_state = self.client.getMultirotorState(vehicle_name="Drone1")
        drone2_state = self.client.getMultirotorState(vehicle_name="Drone2")
        drone2_pos = drone2_state.kinematics_estimated.position
        x = drone2_pos.x_val + drone2_state.kinematics_estimated.linear_velocity.x_val * time
        y = drone2_pos.y_val + drone2_state.kinematics_estimated.linear_velocity.y_val * time
        z = drone2_pos.z_val + drone2_state.kinematics_estimated.linear_velocity.z_val * time
        dx = x - drone1_state.kinematics_estimated.position.x_val
        dy = y - drone1_state.kinematics_estimated.position.y_val
        dz = z - drone1_state.kinematics_estimated.position.z_val
        vel_x = dx / time
        vel_y = dy / time
        vel_z = dz / time
        ddis = np.array([dx, dy, dz])
        dvel = np.array([vel_x, vel_y, vel_z])
        vel = kp*ddis + kv*dvel
        self.client.moveByVelocityAsync(vx=vel[0], vy=vel[1], vz=vel[2],duration=time,
                                        drivetrain=airsim.DrivetrainType.MaxDegreeOfFreedom,
                                        yaw_mode=airsim.YawMode(False, 90), vehicle_name="Drone1")

    def home(self):
        print("Home received")
        self.client.goHomeAsync()
        self.client.armDisarm(False)

    def stop(self):
        self.client.goHomeAsync()
        self.client.armDisarm(False)
        self.client.reset()

    def orbit(self, args):  # name, speed, x,y
        if len(args) < 3:
            print("need at least speed parameter and iterations")
            return
        if len(args) != 4:  # Name, x,y
            target_x = float(72.38)  # X coordinate of turbine 1
            target_y = float(48.92)  # Y coordinate of turbine 1

            self.client.enableApiControl(True)
            self.client.moveToPositionAsync(x=float(36.33), y=float(24.32), z=-float(17.33),
                                            velocity=2, drivetrain=airsim.DrivetrainType.ForwardOnly,
                                            yaw_mode=airsim.YawMode(False, 0)).join()
            self.client.hoverAsync().join()
            airsim.time.sleep(2)
        else:
            target_x = float(args[3])
            target_y = float(args[4])
        speed = float(args[1])
        iterations = int(args[2])
        for i in range(iterations):
            current_pos = self.client.getMultirotorState().kinematics_estimated.position
            look_at_point = np.array([target_x, target_y])
            current_pos_np = np.array([current_pos.x_val, current_pos.y_val])
            angle = self.lookAt(look_at_point, np.array([1, 0]))
            l = look_at_point - current_pos_np
            radius = np.linalg.norm(l)
            print("Radius:", radius)
            # Have to make sure it is enabled:
            self.client.enableApiControl(True)
            self.client.rotateToYawAsync(angle, 20, 0).join()
            print(self.client.getMultirotorState().kinematics_estimated.orientation)

            self.nav = OrbitNavigator(self.client,
                                      radius=radius,
                                      altitude=float(current_pos.z_val),
                                      speed=speed,
                                      iterations=1,
                                      center=l)

            self.nav.start()
            print("Orbit ", i, "is done, climb to:", current_pos.x_val, current_pos.y_val, current_pos.z_val - radius)
            self.client.moveToPositionAsync(current_pos.x_val, current_pos.y_val, current_pos.z_val - radius, speed,
                                            10).join()

    def world_to_drone(self, velocity):
        vx, vy, vz = velocity.x_val, velocity.y_val, velocity.z_val
        ori = self.client.getMultirotorState(vehicle_name="Drone1").kinematics_estimated.orientation
        q0, q1, q2, q3 = ori.w_val, ori.x_val, ori.y_val, ori.z_val
        R_drone_to_world = np.array([[q0*q0 + q1*q1 - q2*q2 - q3*q3, 2*(q1*q2 - q0*q3), 2*(q1*q3 + q0*q2)],
                      [2*(q1*q2 + q0*q3), q0*q0 - q1*q1 + q2*q2 - q3*q3, 2*(q2*q3 - q0*q1)],
                      [2*(q1*q3 - q0*q2), 2*(q2*q3 - q0*q1), q0*q0 - q1*q1 - q2*q2 + q3*q3]])
        vel = np.array([[vx], [vy], [vz]])
        vel_in_drone = np.matmul(np.linalg.inv(R_drone_to_world), vel)
        velocity.x_val, velocity.y_val, velocity.z_val = map(float, vel_in_drone)
        return velocity

    def lookAt(self, target_pos, current_pos):
        dx = target_pos[0] - current_pos[0]
        dy = target_pos[1] - current_pos[1]
        angle = np.arctan2(dy, dx) * 180 / np.math.pi
        return angle

    def handle_gimble_key(self, keys_to_check: list, pressed_keys: list, current_ori=(0, 0, 0)):
        if keys_to_check[0] in pressed_keys:
            pitch = round(number=float(np.clip(current_ori[0] + 0.1, -1.0, 1.0)), ndigits=2)
            return airsim.to_quaternion(pitch, current_ori[1], current_ori[2]), True
        if keys_to_check[1] in pressed_keys:
            pitch = round(number=float(np.clip(current_ori[0] - 0.1, -1.0, 1.0)), ndigits=2)
            return airsim.to_quaternion(pitch, current_ori[1], current_ori[2]), True
        if keys_to_check[2] in pressed_keys:
            yaw = round(number=float(np.clip(current_ori[2] + 0.1, -1.0, 1.0)), ndigits=2)
            return airsim.to_quaternion(current_ori[0], current_ori[1], yaw), True
        if keys_to_check[3] in pressed_keys:
            yaw = round(number=float(np.clip(current_ori[2] - 0.1, -1.0, 1.0)), ndigits=2)
            return airsim.to_quaternion(current_ori[0], current_ori[1], yaw), True

        return airsim.Quaternionr(), False

    def handle_key_pressed(self, keys_to_check: list, pressed_keys: list, current_vel: float) -> float:
        new_vel = current_vel
        positive_axis_press = keys_to_check[0] in pressed_keys
        negative_axis_press = keys_to_check[1] in pressed_keys

        # if keys_to_check[0] == 'z' or keys_to_check[1] == 'c':
        #     if positive_axis_press:
        #         return round(new_vel + 360, ndigits=2)
        #     if negative_axis_press:
        #         return round(new_vel - 360, ndigits=2)
        #     return None
        # else:
        if positive_axis_press and negative_axis_press:
            return new_vel

        if positive_axis_press:
            return round(number=float(np.clip(new_vel + 1, - self.maxmin_vel, self.maxmin_vel)), ndigits=2)

        if negative_axis_press:
            return round(number=float(np.clip(new_vel - 1, - self.maxmin_vel, self.maxmin_vel)), ndigits=2)

        # nothing is pressed, smoothly lowering the value
        return round(number=float(np.clip(new_vel * 0.75, - self.maxmin_vel, self.maxmin_vel)), ndigits=2)

    def save_key_pressed(self, keys_to_check: list, pressed_keys: list):
        saveimg_press = keys_to_check[0] in pressed_keys
        if saveimg_press:
            self.saveimg()
            print("IMG has saved!")

    def drone_track_pressed(self, keys_to_check: list, pressed_keys: list, dc):
        track_press = keys_to_check[0] in pressed_keys
        if track_press:
            print("Drone2 track!")

    def enter_keyboard_control(self):
        print("You entered the keyboard mode. Press 't' to return.")
        kc = KeyController()
        self.client.enableApiControl(True)
        while kc.thread.is_alive():
            self.client.cancelLastTask()
            self.client.enableApiControl(True)
            keys = kc.get_key_pressed()
            quad_vel = self.client.getMultirotorState().kinematics_estimated.linear_velocity
            quad_vel = self.world_to_drone(quad_vel)
            ori = airsim.to_eularian_angles(self.client.getMultirotorState().kinematics_estimated.orientation)
            self.vx = self.handle_key_pressed(keys_to_check=['w', 's'], pressed_keys=keys, current_vel=quad_vel.x_val)
            self.vy = self.handle_key_pressed(keys_to_check=['d', 'a'], pressed_keys=keys, current_vel=quad_vel.y_val)
            self.vz = self.handle_key_pressed(keys_to_check=['e', 'q'], pressed_keys=keys, current_vel=quad_vel.z_val)
            self.yaw = self.handle_key_pressed(keys_to_check=['[', ']'], pressed_keys=keys, current_vel=self.yaw)
            self.save_key_pressed(keys_to_check=['o'], pressed_keys=keys)

            # angular_vel = self.client.getMultirotorState().kinematics_estimated.angular_velocity
            ori = airsim.to_eularian_angles(self.client.simGetCameraInfo(camera_name="0").pose.orientation)
            print(ori)
            self.vx = self.handle_key_pressed(keys_to_check=['w', 's'], pressed_keys=keys, current_vel=quad_vel.x_val)
            self.vy = self.handle_key_pressed(keys_to_check=['d', 'a'], pressed_keys=keys, current_vel=quad_vel.y_val)
            self.vz = self.handle_key_pressed(keys_to_check=['e', 'q'], pressed_keys=keys, current_vel=quad_vel.z_val)
            self.yaw = self.handle_key_pressed(keys_to_check=['[', ']'], pressed_keys=keys, current_vel=self.yaw)
            camera_ori, new = self.handle_gimble_key(keys_to_check=['key.up', 'key.down', 'key.right', 'key.left'],
                                                     pressed_keys=keys, current_ori=ori)

            if new:
                camera_pose = airsim.Pose(airsim.Vector3r(0, 0, 0.2), camera_ori)
                self.client.simSetCameraPose("0", camera_pose)

            # self.yaw_rate = self.handle_key_pressed(keys_to_check=['z', 'c'], pressed_keys=keys,current_vel=angular_vel.z_val)
            print(
                "current vel: \n vx:{0}, nvx:{1}\n vy:{2}, nvy:{3}\n vz:{4}, nvz:{5}\n".format(quad_vel.x_val, self.vx,
                                                                                               quad_vel.y_val, self.vy,
                                                                                               quad_vel.z_val, self.vz))
            current_pos = self.client.getMultirotorState().kinematics_estimated.position
            self.client.moveByVelocityBodyFrameAsync(self.vx, self.vy, self.vz, 0.1,
                                                     airsim.DrivetrainType.MaxDegreeOfFreedom,
                                                     airsim.YawMode(True, -1*self.yaw)).join()
            # self.client.moveByVelocityZBodyFrameAsync(vx=self.vx, vy=self.vy, z=self.height, duration=0.1,
            #                                             drivetrain=airsim.DrivetrainType.MaxDegreeOfFreedom,
            #                                             yaw_mode=airsim.YawMode(True, -1 * self.yaw)).join()
            # print('self.yaw_rate: ', self.vz)
            current_pos = self.client.getMultirotorState().kinematics_estimated.position
            # height = current_pos.z_val
            # self.vz = np.clip(-height+self.height, -self.maxmin_vel, self.maxmin_vel)
            print("current pos: \n x:{0}, y:{1}\n z:{2}\n".format(current_pos.x_val, current_pos.y_val,
                                                                  current_pos.z_val))

            self.client.moveByVelocityBodyFrameAsync(self.vx, self.vy, self.vz, 0.1,
                                                     airsim.DrivetrainType.MaxDegreeOfFreedom,
                                                     airsim.YawMode(True, -1*self.yaw)).join()


            # if self.yaw_rate != 0:
            # self.client.rotateByYawRateAsync(self.yaw_rate, 0.1).join()
            # if self.yaw_rate:
            #     # self.client.rotateToYawAsync(90).join()
            #     self.client.rotateByYawRateAsync(30, 1).join()
            # airsim.time.sleep(0.2)
        print("'t' has been pressed and the console control is back")
        self.client.hoverAsync().join()

    def print_stats(self):
        state = self.client.getMultirotorState()
        s = pprint.pformat(state)
        print("state: %s" % s)

        imu_data = self.client.getImuData()
        s = pprint.pformat(imu_data)
        print("imu_data: %s" % s)

        barometer_data = self.client.getBarometerData()
        s = pprint.pformat(barometer_data)
        print("barometer_data: %s" % s)

        magnetometer_data = self.client.getMagnetometerData()
        s = pprint.pformat(magnetometer_data)
        print("magnetometer_data: %s" % s)

        gps_data = self.client.getGpsData()
        s = pprint.pformat(gps_data)
        print("gps_data: %s" % s)

    def saveimg(self):
        # take images
        # get camera images from the car
        time1 = time.time()
        responses = self.imgclient.simGetImages([
            airsim.ImageRequest("3", airsim.ImageType.DepthPlanar, True, False),  #depth visualization image
            #airsim.ImageRequest("1", airsim.ImageType.DepthPerspective, True), #depth in perspective projection
            #airsim.ImageRequest("0", airsim.ImageType.Scene), #scene vision image in png format
            airsim.ImageRequest("3", airsim.ImageType.Scene, False, False),  #scene vision image in uncompressed RGBA array
            airsim.ImageRequest("3",airsim.ImageType.Segmentation, False, False)  #segmentation vision image
            ], vehicle_name="Drone1")
        # print('Retrieved images: %d' % len(responses))
        time2 = time.time()
        tmp_dir = os.path.join(DIR, "picture")
        # print ("Saving images to %s" % tmp_dir)
        try:
            os.makedirs(tmp_dir)
            os.makedirs(os.path.join(tmp_dir, "DepthVis"))
            os.makedirs(os.path.join(tmp_dir, "Scene"))
            os.makedirs(os.path.join(tmp_dir, "Segmentation"))
        except OSError:
            if not os.path.isdir(tmp_dir):
                raise

        states = self.imgclient.getMultirotorState()
        pos = states.kinematics_estimated.position
        orien = states.kinematics_estimated.orientation
        imu_data = self.imgclient.getImuData()
        gps_data = self.imgclient.getGpsData()
        timestamp = states.timestamp
        time3 = time.time()
        for idx, response in enumerate(responses):
            if idx == 0:
                out_dir = os.path.join(tmp_dir, "DepthVis")
                filename = os.path.join(out_dir, str(timestamp))
            elif idx == 1:
                out_dir = os.path.join(tmp_dir, "Scene")
                filename = os.path.join(out_dir, str(timestamp))
            else:
                out_dir = os.path.join(tmp_dir, "Segmentation")
                filename = os.path.join(out_dir, str(timestamp))

            if response.pixels_as_float:
                # print("Type %d, size %d" % (response.image_type, len(response.image_data_float)))
                # airsim.write_pfm(os.path.normpath(filename + '.pfm'), airsim.get_pfm_array(response))
                # get numpy array

                # Reshape to a 2d array with correct width and height
                # depth_img_in_meters = airsim.list_to_2d_float_array(response.image_data_float, response.width,
                #                                                     response.height)
                # depth_img_in_meters = depth_img_in_meters.reshape(response.height, response.width, 1)
                # print("depth_img_in_meters:\n", depth_img_in_meters)
                # # Lerp 0..230m to 0..255 gray values
                # # depth_8bit_lerped = np.interp(depth_img_in_meters, (MIN_DEPTH_METERS, MAX_DEPTH_METERS), (0, 255))
                # # cv2.imwrite("depth_visualization.png", depth_8bit_lerped.astype('uint8'))
                # # Convert depth_img to millimeters to fill out 16bit unsigned int space (0..65535). Also clamp large values (e.g. SkyDome) to 65535
                # depth_img_in_millimeters = depth_img_in_meters * 500
                # depth_16bit = np.clip(depth_img_in_millimeters, 0, 65535)
                # path = os.path.join(filename + '.png')
                # cv2.imwrite(path, depth_16bit.astype('uint16'))


                img1d = np.array(response.image_data_float, dtype=float)
                # print("img1d:\n", img1d)
                # print("shape:", img1d.shape)
                img_depth = img1d.reshape(response.height, response.width)
                img_depth = np.flipud(img_depth)
                # to visualize
                img_depth = np.clip(img_depth, MIN_DEPTH_METERS, MAX_DEPTH_METERS)

                # print("\nimg_depth1\n", img_depth)
                img_depth[img_depth>229] = 0
                # print("dist_max:", np.max(img_depth))
                # print(img_depth.shape)
                img_depth = (img_depth * 255).astype(np.uint16)
                # print("\nimg_depth2\n", img_depth)
                # print(max(max(img_depth.tolist())), min(min(img_depth.tolist())))
                path = os.path.join(filename + '.png')
                airsim.write_png(os.path.normpath(path), cv2.flip(img_depth, 0))

            # elif response.compress: #png format
            #     # print("Type %d, size %d" % (response.image_type, len(response.image_data_uint8)))
            #     # airsim.write_file(os.path.normpath(filename + '.png'), response.image_data_uint8)
            #     pass
            else: #uncompressed array
                # # print("Type %d, size %d" % (response.image_type, len(response.image_data_uint8)))
                # img1d = np.fromstring(response.image_data_float, dtype=np.uint8) # get numpy array
                # img_rgb = img1d.reshape(response.height, response.width, 3) # reshape array to 4 channel image array H X W X 3
                # cv2.imwrite(os.path.normpath(filename + '.png'), img_rgb) # write to png
                # get numpy array
                img1d = np.frombuffer(response.image_data_uint8, dtype=np.uint8)
                # reshape array to 4 channel image array H X W X 4
                img_rgb = img1d.reshape(response.height, response.width, 3)
                # original image is fliped vertically
                img_rgb = np.flipud(img_rgb)
                # write to png
                path = os.path.join(filename + '.png')
                airsim.write_png(os.path.normpath(path), cv2.flip(img_rgb, 0))
        time4 = time.time()
        vehicle_name = "Drone1"
        imu_name = "ImuSensor1"
        try:
            filename = os.path.join(DIR, vehicle_name + "_" + imu_name + ".txt")
            if not self.imu_existing_data_cleared:
                f = open(filename, 'w')
                self.imu_existing_data_cleared = True
            else:
                f = open(filename, 'a')
            linear_acceleration = imu_data.linear_acceleration
            angular_velocity = imu_data.angular_velocity
            latitude = gps_data.gnss.geo_point.latitude
            longitude = gps_data.gnss.geo_point.longitude
            altitude = gps_data.gnss.geo_point.altitude

            f.write("Timestamp: %d: \t POS xyz: %f %f %f \t orien QW,QX,QY,QZ: %f %f %f %f\t"
                    "IMU_GPS linear_acceleration xyz: %f %f %f \t angular_velocity xyz: %f %f %f \t latitude: %f, longitude: %f, altitude: %f\n"
                    % (
                        timestamp, pos.x_val, pos.y_val, pos.z_val, orien.w_val, orien.x_val, orien.y_val, orien.z_val,
                        linear_acceleration.x_val, linear_acceleration.y_val, linear_acceleration.z_val,
                        angular_velocity.x_val, angular_velocity.y_val, angular_velocity.z_val,
                        latitude, longitude, altitude))
            f.close()
        except KeyboardInterrupt:
            airsim.wait_key('Press any key to stop running this script')
            print("Done!\n")
        time5 = time.time()
        print("times:", time2-time1, time3-time2, time4-time3, time5-time4, sep=' ')

    def run(self):
        while True:
            command = input()
            args = command.split(" ")
            print("Args given", args)
            command_type = args[0]
            if command_type.lower() == ARM:
                self.arm()
            elif command_type.lower() == DISARM:
                self.disarm()
            elif command_type.lower() == MOVE:
                self.move_to_position(args)
            elif command_type.lower() == MOVE_PATH:
                self.move_on_path(args)
            elif command_type.lower() == DRONE_MOVE:
                self.drone_move(args)
            elif command_type.lower() == HOME:
                self.home()
            elif command_type.lower() == TAKEOFF:
                self.takeoff()
            elif command_type.lower() == STATE:
                self.print_stats()
            elif command_type.lower() == KEYBOARD_CONTROL:
                self.enter_keyboard_control()
            elif command_type.lower() == STOP:
                self.stop()
                break
            elif command_type.lower() == ORBIT:
                self.orbit(args)
            elif command_type.lower() == LANDING:
                self.land()
            else:
                print("The command given is not a valid command.")

        # that's enough fun for now. let's quit cleanly
        airsim.wait_key("When ready to kill")
        self.client.enableApiControl(False)


if __name__ == '__main__':
    controller = SimpleTerminalController(maxmin_velocity=8)
    controller.run()
