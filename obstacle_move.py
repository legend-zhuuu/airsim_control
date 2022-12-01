import airsim
import numpy as np
import time

MAINTAIN = 0
ACROSS1 = 1  # from start point to center point
ACROSS2 = 2  # from center point to final point
END = 3  # across end


class DroneController:
    def __init__(self, drone_name):
        # argument
        self.drone_name = drone_name
        self.distance = 10
        self.time = 1.
        self.gaussian_coeff = 0.6
        self.tolerate = 3
        self.settingsCoordinate_Drone2ToDrone1_x = 6
        self.settingsCoordinate_Drone2ToDrone1_y = 15
        self.norm_mean = (0, 0, 0)
        self.norm_cov = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        self.coeff_A = [[0., 0., 0., 0.],
                        [0., 0., 0., 0.],
                        [0., 0., 0., 0.]]
        self.R_world_to_drone = None
        self.max_velocity = 12.
        if np.random.random() > 0.5:
            self.state = MAINTAIN
        else:
            self.state = MAINTAIN
            self.max_velocity = 8.

        # init function
        self.client = airsim.MultirotorClient()
        self.confirm_connection()
        self.arm_api()
        self.drone2takeoff()

        # others
        drone1state = self.client.getMultirotorState(vehicle_name="Drone1")
        drone1_pos = drone1state.kinematics_estimated.position
        drone2state = self.client.getMultirotorState(vehicle_name="Drone2")
        drone2_pos = drone2state.kinematics_estimated.position
        self.host_position = np.array([drone1_pos.x_val, drone1_pos.y_val, drone1_pos.z_val])
        self.my_position = np.array([drone2_pos.x_val, drone2_pos.y_val, drone2_pos.z_val])
        self.host_pos_end = None
        self.my_velocity = None
        self.previous_time = time.time()
        self.current_time = time.time()

    def confirm_connection(self):
        self.client.confirmConnection()

    def arm_api(self):
        self.client.enableApiControl(True, vehicle_name=self.drone_name)
        self.client.armDisarm(True, vehicle_name=self.drone_name)

    def drone2takeoff(self):
        self.client.takeoffAsync(vehicle_name="Drone2")

    def get_R_world_to_drone(self):
        ori = self.client.getMultirotorState(vehicle_name="Drone1").kinematics_estimated.orientation
        q0, q1, q2, q3 = ori.w_val, ori.x_val, ori.y_val, ori.z_val
        R_drone_to_world = np.array([[q0*q0 + q1*q1 - q2*q2 - q3*q3, 2*(q1*q2 - q0*q3), 2*(q1*q3 + q0*q2)],
                      [2*(q1*q2 + q0*q3), q0*q0 - q1*q1 + q2*q2 - q3*q3, 2*(q2*q3 - q0*q1)],
                      [2*(q1*q3 - q0*q2), 2*(q2*q3 - q0*q1), q0*q0 - q1*q1 - q2*q2 + q3*q3]])
        self.R_world_to_drone = (R_drone_to_world)

    def coordinate_change(self):
        self.host_position[0] -= self.settingsCoordinate_Drone2ToDrone1_x
        self.host_position[1] -= self.settingsCoordinate_Drone2ToDrone1_y

    def host_position_predict(self, drone1state):
        drone1_velocity = drone1state.kinematics_estimated.linear_velocity
        self.host_position[0] += self.time * drone1_velocity.x_val
        self.host_position[1] += self.time * drone1_velocity.y_val
        self.host_position[2] += self.time * drone1_velocity.z_val

    def add_gaussian_noise(self):
        nx, ny, nz = self.gaussian_coeff * np.random.multivariate_normal(self.norm_mean, self.norm_cov, size=1)[0]
        self.host_position[0] += nx
        self.host_position[1] += ny
        self.host_position[2] += nz

    def get_my_position(self):
        drone2_state = self.client.getMultirotorState(vehicle_name=self.drone_name)
        drone2_pos = drone2_state.kinematics_estimated.position
        self.my_position = np.array([drone2_pos.x_val, drone2_pos.y_val, drone2_pos.z_val])

    def run(self):
        drone1state = self.client.getMultirotorState(vehicle_name="Drone1")
        drone1_pos = drone1state.kinematics_estimated.position
        self.current_time = time.time()
        if self.state == MAINTAIN:
            # compute gaussian noise per second
            if self.current_time - self.previous_time > self.time:
                self.host_position = np.array([drone1_pos.x_val, drone1_pos.y_val, drone1_pos.z_val])
                self.coordinate_change()
                self.host_position_predict(drone1state)
                self.add_gaussian_noise()
                distance_position = np.array([[self.distance], [0], [0]])
                self.get_R_world_to_drone()
                self.host_position += np.matmul(self.R_world_to_drone, distance_position).reshape(3,)
                self.get_my_position()
                # polynomial fitting A: coefficient 3*4
                # self.coeff_A = self.compute_coeff()
                self.previous_time = self.current_time
            self.track()
        elif self.state == ACROSS1:  # state1: Drone2 move to center point(x+10, y, z)
            if self.current_time - self.previous_time > self.time:
                self.host_position = np.array([drone1_pos.x_val, drone1_pos.y_val, drone1_pos.z_val])
                self.coordinate_change()
                self.host_position_predict(drone1state)
                self.add_gaussian_noise()
                self.host_position[0] += self.distance
                self.get_my_position()
                # polynomial fitting A: coefficient 3*4
                # self.coeff_A = self.compute_coeff()
                self.previous_time = self.current_time
                if np.linalg.norm(
                        self.host_position - self.my_position) < 1 * self.tolerate:
                    self.state = ACROSS2
                    self.client.hoverAsync().join()
                    airsim.time.sleep(1)
            self.track()
        elif self.state == ACROSS2:  # state2: Drone2 move to end point(x+10, y-10, z)
            self.host_position = np.array([drone1_pos.x_val, drone1_pos.y_val, drone1_pos.z_val])
            self.coordinate_change()
            self.host_position[0] = self.host_position[0] + 2*self.distance
            self.host_position[1] = self.host_position[1] - 3*self.distance
            if self.host_pos_end is None:
                self.host_pos_end = self.host_position
            self.get_my_position()
            if np.linalg.norm(self.host_pos_end - self.my_position) < self.tolerate:
                self.state = END
            self.track()
        elif self.state == END:
            self.client.hoverAsync()
        else:
            print("state error!")
        print("goal: ", self.host_position, "state: ", self.state)

    def compute_coeff(self, position1=None, position2=None, velocity1=None, velocity2=0.1, t=None):
        # get the position of Drone2
        if position1 is None:
            position1 = self.my_position
        if position2 is None:
            position2 = self.host_position
        if velocity1 is None:
            velocity1 = self.my_velocity
        if t is None:
            t = self.time
        drone2_state = self.client.getMultirotorState(vehicle_name=self.drone_name)
        drone2_pos = drone2_state.kinematics_estimated.position
        velocity = drone2_state.kinematics_estimated.linear_velocity
        velocity1 = np.array([velocity.x_val, velocity.y_val, velocity.z_val])
        position1 = np.array([drone2_pos.x_val, drone2_pos.y_val, drone2_pos.z_val])
        A_xyz = []
        for i in range(3):
            A = np.array([[1, 0, 0, 0],
                          [1, t, pow(t, 2), pow(t, 3)],
                          [0, 1, 0, 0],
                          [0, 1, 2 * t, 3 * pow(t, 2)]])
            velocity2 = np.clip((position2[i] - position1[i]), -1, 1)
            A_i = np.matmul(np.linalg.inv(A),
                            np.array([[position1[i], position2[i], velocity1[i], velocity2]]).T)
            A_xyz.append([A_i[0][0], A_i[1][0], A_i[2][0], A_i[3][0]])
        return A_xyz

    def track_A(self):
        vel = np.array([0., 0., 0.])
        t = self.current_time - self.previous_time
        # print("t:", t)
        print("host_position: ", self.host_position)
        if t < self.time:
            for i in range(3):
                A_i = self.coeff_A[i]
                vel_i = A_i[1] + 2 * A_i[2] * t + 3 * A_i[3] * pow(t, 2)
                vel_i = np.clip((self.host_position[i] - self.my_position[i]) / self.time, -self.max_velocity, self.max_velocity)
                vel[i] = vel_i
        print("Drone2_vel: ", vel)
        self.client.moveByVelocityAsync(vx=vel[0], vy=vel[1], vz=vel[2], duration=self.time,
                                        drivetrain=airsim.DrivetrainType.MaxDegreeOfFreedom,
                                        yaw_mode=airsim.YawMode(False, 0), vehicle_name=self.drone_name)

    def track(self):
        vel = np.clip((self.host_position - self.my_position) / self.time, -self.max_velocity, self.max_velocity)
        print("Drone2_vel: ", vel)
        self.client.moveByVelocityAsync(vx=vel[0], vy=vel[1], vz=vel[2], duration=self.time,
                                        drivetrain=airsim.DrivetrainType.MaxDegreeOfFreedom,
                                        yaw_mode=airsim.YawMode(False, 0), vehicle_name=self.drone_name)


if __name__ == '__main__':
    Dcontrol = DroneController("Drone2")
