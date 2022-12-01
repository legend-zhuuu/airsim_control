import pprint
import numpy as np
import os
import cv2
import airsim
import time
from KeyController import KeyController
from obstacle_move import DroneController
import threading

# TIMEOUT
from airsim_functions.orbit import OrbitNavigator

TIMEOUT = 1200  # 20 miniuts

# Mesh ID's
OBSTACLE = 100
DRONE = 200

# Commands:
ARM = "arm"
DISARM = "disarm"
MOVE = "move"
MOVE_PATH = "moveonpath"
DRONE_MOVE = "dronemove"
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
DIR = r"C:\Users\13283\Documents"


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
        self.sensors_client = airsim.MultirotorClient()
        self.image_client = airsim.MultirotorClient()
        self.confirm_connection()
        # self.client.simEnableWeather(True)
        # self.client.simSetWeatherParameter(airsim.WeatherParameter.Rain, 0.1)
        # self.yaw_mode = airsim.YawMode(True, 0)
        # connect to the AirSim simulator
        # Segmentation setup
        # self.setup_segmentation_colors()

        # Movement and constraints:
        self.vx = 0
        self.vy = 0
        self.vz = 0
        self.yaw = 0
        self.nav = None
        self.imgcount = 0
        self.saveimgHz = 5
        self.maxmin_vel = maxmin_velocity

        #thread -get imu, lidar, camera data
        self.sensors_callback_thread = threading.Thread(target=self.repeat_timer_sensors_callback,
                                                      args=(self.sensors_callback, 0.1))
        self.is_sensors_thread_active = False
        self.image_callback_thread = threading.Thread(target=self.repeat_timer_image_callback,
                                                      args=(self.image_callback, 0.1))
        self.is_image_thread_active = False

        self.imu_existing_data_cleared = False
        self.lidar_existing_data_cleared = False

    def confirm_connection(self):
        self.client.confirmConnection()
        self.client.enableApiControl(True)
        self.sensors_client.confirmConnection()
        self.sensors_client.enableApiControl(True)
        self.image_client.confirmConnection()
        self.image_client.enableApiControl(True)
        self.client.armDisarm(True)
        self.yaw_rate = 0
        self.nav = None

        # camera_pose = airsim.Pose(airsim.Vector3r(0, 0, 0), airsim.to_quaternion(-1, 0, -0.1))
        # self.client.simSetCameraPose("0", camera_pose)

    def setup_segmentation_colors(self):

        # Finding regexp GameObject name and set the ID
        # success = self.client.simSetSegmentationObjectID("SM_Floor20m[\w]*", 100, True)
        # print("Change of color =", success)

        self.change_color("", OBSTACLE)
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

    def drone_move(self, args: list):
        print("MoveOnPath received")
        if len(args) % 3 != 0:
            print("DroneMove needs vehicle_name, position, velocity.")
            return
        self.host_move(args)
        drone2_state = self.client.getMultirotorState(vehicle_name="Drone2")
        previous_time = time.time()
        saveimg_time = previous_time
        airsim.time.sleep(1)
        dist = 10
        # Have to make sure it is enabled:
        self.client.enableApiControl(True, vehicle_name = "Drone1")
        t = 1
        while dist > 1:
            current_time = time.time()
            if current_time - previous_time > t:
                drone2_state = self.client.getMultirotorState(vehicle_name="Drone2")
                current_pos = drone2_state.kinematics_estimated.position
                dist = np.linalg.norm((current_pos.x_val - float(args[-4]),
                                       current_pos.y_val - float(args[-3]),
                                       current_pos.z_val - float(args[-2])))
                previous_time = current_time
                self.track(args[-1], t)
                print("dist:", dist)
            if current_time - saveimg_time > 1/self.saveimgHz:
                saveimg_time = current_time
                self.saveimg()
        print("track end")

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
        self.stop_sensors_callback_thread()
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
                # self.saveimg()
            # angular_vel = self.client.getMultirotorState().kinematics_estimated.angular_velocity
            ori = airsim.to_eularian_angles(self.client.simGetCameraInfo('0').pose.orientation)
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
            print("current pos: \n x:{0}, y:{1}\n z:{2}\n".format(current_pos.x_val, current_pos.y_val,
                                                                  current_pos.z_val))
            self.client.moveByVelocityBodyFrameAsync(self.vx, self.vy, self.vz, 0.1,
                                                     airsim.DrivetrainType.MaxDegreeOfFreedom,
                                                     airsim.YawMode(True, -10*self.yaw)).join()
            # print('self.yaw_rate: ', self.vz)
            current_pos = self.client.getMultirotorState().kinematics_estimated.position
            # print("current pos: \n x:{0}, y:{1}\n z:{2}\n".format(current_pos.x_val, current_pos.y_val,
            #                                                       current_pos.z_val))
            self.client.moveByVelocityBodyFrameAsync(self.vx, self.vy, self.vz, 0.1,
                                                     airsim.DrivetrainType.MaxDegreeOfFreedom,
                                                     airsim.YawMode(True, -10*self.yaw)).join()
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

    def sensor_move(self):
        self.client.moveToPositionAsync(5, 0, -5, 2).join()
        self.client.moveToPositionAsync(5, -60, -5, 2).join()

    def saveimg(self):
        # take images
        # get camera images from the car
        responses = self.image_client.simGetImages([
            airsim.ImageRequest("0", airsim.ImageType.DepthVis),  #depth visualization image
            #airsim.ImageRequest("1", airsim.ImageType.DepthPerspective, True), #depth in perspective projection
            #airsim.ImageRequest("0", airsim.ImageType.Scene), #scene vision image in png format
            airsim.ImageRequest("1", airsim.ImageType.Scene, False, False),  #scene vision image in uncompressed RGBA array
            airsim.ImageRequest("0",airsim.ImageType.Segmentation)  #segmentation vision image
            ])
        # print('Retrieved images: %d' % len(responses))

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

        for idx, response in enumerate(responses):
            if idx == 0:
                out_dir = os.path.join(tmp_dir, "DepthVis")
                filename = os.path.join(out_dir, str(self.imgcount) + "_" + str(idx))
            elif idx == 1:
                out_dir = os.path.join(tmp_dir, "Scene")
                filename = os.path.join(out_dir, str(self.imgcount) + "_" + str(idx))
            else:
                out_dir = os.path.join(tmp_dir, "Segmentation")
                filename = os.path.join(out_dir, str(self.imgcount) + "_" + str(idx))

            if response.pixels_as_float:
                # print("Type %d, size %d" % (response.image_type, len(response.image_data_float)))
                airsim.write_pfm(os.path.normpath(filename + '.pfm'), airsim.get_pfm_array(response))
            elif response.compress: #png format
                # print("Type %d, size %d" % (response.image_type, len(response.image_data_uint8)))
                airsim.write_file(os.path.normpath(filename + '.png'), response.image_data_uint8)
            else: #uncompressed array
                # print("Type %d, size %d" % (response.image_type, len(response.image_data_uint8)))
                img1d = np.fromstring(response.image_data_uint8, dtype=np.uint8) # get numpy array
                img_rgb = img1d.reshape(response.height, response.width, 3) # reshape array to 4 channel image array H X W X 3
                cv2.imwrite(os.path.normpath(filename + '.png'), img_rgb) # write to png
        self.imgcount += 1

    def save_imu_data(self):
        vehicle_name = "Drone1"
        imu_name = "ImuSensor1"
        try:
            filename = os.path.join(DIR, vehicle_name + "_" + imu_name + ".txt")
            if not self.imu_existing_data_cleared:
                f = open(filename, 'w')
                self.imu_existing_data_cleared = True
            else:
                f = open(filename, 'a')
            imu_data = self.sensors_client.getImuData(imu_name=imu_name, vehicle_name=vehicle_name)
            linear_acceleration = imu_data.linear_acceleration
            angular_velocity = imu_data.angular_velocity
            orientation = imu_data.orientation
            f.write("id: %d: \n linear_acceleration xyz: %f %f %f \n "
                    "angular_velocity xyz: %f %f %f \n "
                    "orientation wxyz: %f %f %f %f\n"
                    % (self.imgcount, linear_acceleration.x_val, linear_acceleration.y_val, linear_acceleration.z_val,
                       angular_velocity.x_val, angular_velocity.y_val, angular_velocity.z_val,
                       orientation.w_val, orientation.x_val, orientation.y_val, orientation.z_val))
            f.close()
        except KeyboardInterrupt:
            airsim.wait_key('Press any key to stop running this script')
            print("Done!\n")

    def save_lidar_data(self):
        vehicle_name = "Drone1"
        lidar_name = "LidarSensor1"
        try:
            filename = os.path.join(DIR, vehicle_name + "_" + lidar_name + "_" + "pointcloud.txt")
            if not self.lidar_existing_data_cleared:
                f = open(filename, 'w')
                self.lidar_existing_data_cleared = True
            else:
                f = open(filename, 'a')
            lidar_data = self.sensors_client.getLidarData(lidar_name=lidar_name, vehicle_name=vehicle_name)

            for i in range(0, len(lidar_data.point_cloud), 3):
                xyz = lidar_data.point_cloud[i:i + 3]

                f.write("id: %d, XYZ: %f %f %f \n" % (self.imgcount, xyz[0], xyz[1], xyz[2]))
            f.close()
        except KeyboardInterrupt:
            airsim.wait_key('Press any key to stop running lidar script')
            print("Done!\n")

    def sensors_callback(self):
        self.saveimg()
        self.save_lidar_data()
        self.save_imu_data()

    def repeat_timer_sensors_callback(self, task, period):
        while self.is_sensors_thread_active:
            task()
            time.sleep(period)

    def start_sensors_callback_thread(self):
        if not self.is_sensors_thread_active:
            self.is_sensors_thread_active = True
            self.sensors_callback_thread.start()
            print("Started sensors callback thread")

    def stop_sensors_callback_thread(self):
        if self.is_sensors_thread_active:
            self.is_sensors_thread_active = False
            self.sensors_callback_thread.join()
            print("Stopped sensors callback thread.")

    def image_callback(self):
        self.saveimg()

    def repeat_timer_image_callback(self, task, period):
        while self.is_image_thread_active:
            task()
            time.sleep(period)

    def start_image_callback_thread(self):
        if not self.is_image_thread_active:
            self.is_image_thread_active = True
            self.image_callback_thread.start()
            print("Started image callback thread")

    def stop_image_callback_thread(self):
        if self.is_image_thread_active:
            self.is_image_thread_active = False
            self.image_callback_thread.join()
            print("Stopped image callback thread.")



    def run(self):
        # self.start_sensors_callback_thread()
        # self.start_image_callback_thread()
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
                self.sensor_move()
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
    controller = SimpleTerminalController(maxmin_velocity=20)
    controller.run()
