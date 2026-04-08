#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from copy import deepcopy
import math
import time

# Navigation and Message Imports
from geometry_msgs.msg import PoseStamped, Pose
from sensor_msgs.msg import LaserScan
from nav2_simple_commander.robot_navigator import BasicNavigator, TaskResult

# MoveIt2 Action Imports
from moveit_msgs.action import MoveGroup
from moveit_msgs.msg import Constraints, PositionConstraint, JointConstraint, OrientationConstraint
from shape_msgs.msg import SolidPrimitive

class TB3FinalMission(Node):
    def __init__(self):
        # Initialize as a ROS 2 Node so we can use Subscribers/ActionClients
        super().__init__('tb3_final_mission')

        # --- 1. NAVIGATION SETUP ---
        # BasicNavigator handles the high-level wheel movement
        self.navigator = BasicNavigator()

        # --- 2. LIDAR SETUP ---
        # Subscribe to /scan to find the closest object in front of the robot
        self.scan_sub = self.create_subscription(LaserScan, '/scan', self.lidar_callback, 10)
        self.closest_dist = 1.0  # Variable for object distance
        self.closest_angle = 0.0 # Variable for object angle

        # --- 3. MOVEIT SETUP ---
        # Connect to the MoveGroup action server to plan and move the arm
        self.move_group_client = ActionClient(self, MoveGroup, 'move_action')

                
        # NAVIGATE ROUTE [x, y, z_val, w_val]
        self.routes = {
            "red": [
                [4.6, 0.0, 0.0, 1.0],      # infeed
                [4.5, -2.0, 0.707, -0.707],# station 1
                [4.6, 0.0, 0.0, 1.0],      # return
            ],
            "green": [
                [4.6, 0.0, 0.0, 1.0],
                [2.0, 2.0, 1.0, 1.0],      # station 2
                [4.6, 0.0, 0.0, 1.0],
            ],
            "blue": [
                [4.6, 0.0, 0.0, 1.0],
                [2.0, -2.0, 0.707, 0.707], # station 3
                [4.6, 0.0, 0.0, 1.0],
            ]
        }


    def lidar_callback(self, msg):
        """Finds the closest object in a 180-degree front arc."""
        # 180° arc: Indices 0-90 (left-front) and 270-359 (right-front)
        front_indices = list(range(0, 91)) + list(range(270, 360))
        
        valid_hits = []
        for i in front_indices:
            dist = msg.ranges[i]
            # Filter: Ignore 0, infinity, and anything closer than 15cm (robot's own body)
            if 0.15 < dist < 1.5:
                angle = msg.angle_min + (i * msg.angle_increment)
                valid_hits.append((dist, angle))

        if valid_hits:
            # Sort the list by distance and pick the smallest one
            self.closest_dist, self.closest_angle = min(valid_hits, key=lambda x: x[0])

    def send_gripper_goal(self, close=True):
        """Sends a goal to the gripper group using the confirmed joint name."""
        goal_msg = MoveGroup.Goal()
        
        # Ensure this matches your MoveIt SRDF (usually 'gripper')
        goal_msg.request.group_name = "gripper"
        
        joint_con = JointConstraint()
        # Use the exact name from your /joint_states (the right is a mirror of the left)
        # We need to actually control the left
        joint_con.joint_name = "gripper_left_joint" 
        
        # Simulation Values for OpenManipulator-X:
        # Open: 0.019 (positive value)
        # Closed: -0.01 (negative value creates the clamping force)
        joint_con.position = 0.000 if close else 0.019
        
        joint_con.tolerance_above = 0.0005
        joint_con.tolerance_below = 0.0005
        joint_con.weight = 1.0

        constraints = Constraints()
        constraints.joint_constraints.append(joint_con)
        goal_msg.request.goal_constraints.append(constraints)

        self.get_logger().info(f"{'Closing' if close else 'Opening'} gripper...")
        
        # Check for the action server
        if not self.move_group_client.wait_for_server(timeout_sec=2.0):
            self.get_logger().error("MoveGroup action server not available!")
            return

        return self.move_group_client.send_goal_async(goal_msg)

    def send_pick_goal(self, z_height=0.10, pos = None, approach_offset=0.0):
        """Calculates a Cartesian goal and sends it to MoveIt.
        Parameters:
            z_height: Height in Z relative to base
            pos: X and Y position coordinates to force a specific position.
            approach_offset: How many meters to stay BACK from the object 
                            along the line of sight."""

        # Distance from LiDAR center to Arm 'link1' center
        LIDAR_TO_ARM_X = 0.10

        if pos is None:
            # Convert LiDAR polar (dist, angle) to LiDAR-frame Cartesian
            x_lidar = self.closest_dist * math.cos(self.closest_angle)
            y_lidar = self.closest_dist * math.sin(self.closest_angle)

            # Shift the X coordinate because the arm is behind the lidar
            # target_x is now the distance from the Arm Base to the object
            x_arm_total = x_lidar + LIDAR_TO_ARM_X
            y_arm_total = y_lidar

            # Calculate the direct distance (Hypotenuse) from arm to object
            dist_to_obj = math.sqrt(x_arm_total**2 + y_arm_total**2)

            # Preserve the Angle (Aspect Ratio)
            # Find the 'ratio' of the new distance to the total distance
            if dist_to_obj > approach_offset:
                scale = (dist_to_obj - approach_offset) / dist_to_obj
                target_x = x_arm_total * scale
                target_y = y_arm_total * scale
            else:
                # If offset is larger than distance, just stay at the arm base
                target_x, target_y = 0.05, 0.0
        else:
            # Use provided manual position (for the lift/stow phase)
            target_x = pos[0]
            target_y = pos[1]

        # Create the Pose
        target_p = Pose()
        target_p.position.x = float(target_x)
        target_p.position.y = float(target_y)
        target_p.position.z = float(z_height)

        target_p.orientation.x = 0.0
        target_p.orientation.y = 0.0 # was 0.0
        target_p.orientation.z = 0.0
        target_p.orientation.w = 1.0 # was 1.0

        # position constraint, define the target link and tolerance
        pos_con = PositionConstraint()
        pos_con.header.frame_id = "link1" # Arm base
        pos_con.link_name = "end_effector_link" # Standard TB3 link name
        
        box = SolidPrimitive()
        box.type, box.dimensions = SolidPrimitive.BOX, [0.02, 0.02, 0.02]
        
        pos_con.constraint_region.primitives.append(box)
        pos_con.constraint_region.primitive_poses.append(target_p)
        pos_con.weight = 1.0

        # Orientation constraint, define the target link and tolerance
        ori_con = OrientationConstraint()
        ori_con.header.frame_id = "link1" # same frame as position
        ori_con.link_name = "end_effector_link"

        ori_con.orientation = target_p.orientation # enforce this orientation

        # Tight tolerances so it stays horizontal
        ori_con.absolute_x_axis_tolerance = 0.05
        ori_con.absolute_y_axis_tolerance = 0.05
        ori_con.absolute_z_axis_tolerance = 3.14

        ori_con.weight = 1.0

        my_constraints = Constraints()
        my_constraints.position_constraints.append(pos_con)
        my_constraints.orientation_constraints.append(ori_con)

        # Pack and send goal
        goal_msg = MoveGroup.Goal()
        goal_msg.request.group_name = "arm" 
        goal_msg.request.num_planning_attempts = 10
        goal_msg.request.allowed_planning_time = 5.0
        # Now append the fully built constraints object to the list
        goal_msg.request.goal_constraints.append(my_constraints)

        self.get_logger().info(f"Targeting Object at Arm-X: {target_x:.2f}, Arm-Y: {target_y:.2f}, Arm-Z: {z_height:.2f}")
        
        # Ensure MoveIt is actually there before sending
        if not self.move_group_client.wait_for_server(timeout_sec=2.0):
            self.get_logger().error("MoveGroup action server not available! Did you launch MoveIt?")
            return

        # Send the goal asynchronously
        send_goal_future = self.move_group_client.send_goal_async(goal_msg)
        rclpy.spin_until_future_complete(self, send_goal_future)
        goal_handle = send_goal_future.result()

        if not goal_handle.accepted:
            self.get_logger().error("Goal rejected")
            return False

        # Wait for the actual execution/planning result
        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future)
        result = result_future.result()

        # CHECK ERROR CODES
        # MoveIt returns a MoveItErrorCodes message inside the result
        error_code = result.result.error_code.val
        
        if error_code == 1:  # SUCCESS = 1
            self.get_logger().info("Motion successful!")
            return True
        else:
            # Map common codes to readable messages
            error_map = {
                -31: "NO_IK_SOLUTION",
                -1: "PLANNING_FAILED",
                -12: "GOAL_IN_COLLISION",
                -14: "GOAL_CONSTRAINTS_VIOLATED",
                -27: "INVALID_MOTION_PLAN (Goal or Start in collision)",
                99999: "Beyond arm reach"
            }
            err_msg = error_map.get(error_code, f"Unknown Error Code: {error_code}")
            self.get_logger().error(f"MoveGroup failed with status: {err_msg}")
            return False

    def run(self):
        while rclpy.ok():

            # 1. Go to infeed
            self.get_logger().info("Going to infeed...")
            self.go_to_pose([4.6, 0.0, 0.0, 0.0])

            # 2. Pick object
            if self.closest_dist < 1.0:
                self.pick_object()
                time.sleep(1.0)

            # 3. Detect color
            color = self.detect_color()

            # 4. Select route
            route = self.routes.get(color)
            if route is None:
                self.get_logger().error("Unknown color, skipping...")
                continue

            # 5. Go to station
            self.get_logger().info(f"Heading to {color} station...")
            self.go_to_pose(route[1])

            # 6. Drop
            self.drop_object()
            time.sleep(1.0)

            # 7. Return to infeed
            self.get_logger().info("Returning to infeed...")
            self.go_to_pose(route[2])
            
    def detect_color(self): #temp function to test color sorting
        """Placeholder for object classification"""
        self.get_logger().info("Detecting object color...")

        # TEMP: simulate detection
        import random
        color = random.choice(["red", "green", "blue"])

        self.get_logger().info(f"Detected: {color}")
        return color
    
    def go_to_pose(self, pose):
        wp = PoseStamped()
        wp.header.frame_id = 'map'
        wp.header.stamp = self.get_clock().now().to_msg()

        wp.pose.position.x = pose[0]
        wp.pose.position.y = pose[1]
        wp.pose.orientation.z = pose[2]
        wp.pose.orientation.w = pose[3]
        wp.pose.orientation.x = 0.0
        wp.pose.orientation.y = 0.0

        self.navigator.followWaypoints([wp])

        while not self.navigator.isTaskComplete():
            rclpy.spin_once(self, timeout_sec=0.1)

        return self.navigator.getResult() == TaskResult.SUCCEEDED
    
    def pick_object(self):
        self.get_logger().info("Picking object...")

        self.send_gripper_goal(close=False)
        self.send_pick_goal(pos=[0.20, 0.00], z_height=0.15)
        self.send_pick_goal(z_height=0.1)
        self.send_gripper_goal(close=True)
        self.send_pick_goal(pos=[0.15, 0.0], z_height=0.25)
        
    def drop_object(self):
        self.get_logger().info("Dropping object...")

        self.send_pick_goal(pos=[0.35, 0.0], z_height=0.1)
        self.send_gripper_goal(close=False)
        self.send_pick_goal(pos=[0.15, 0.0], z_height=0.25)


def main():
    rclpy.init()
    mission = TB3FinalMission()

    # I. INITIAL POSE (Tell the robot where it is on the map)
    init_pose = PoseStamped()
    init_pose.header.frame_id = 'map'
    init_pose.header.stamp = mission.get_clock().now().to_msg()
    init_pose.pose.position.x, init_pose.pose.position.y, init_pose.pose.orientation.z, init_pose.pose.orientation.w = 0.0, 0.0, 0.0, 1.0
    mission.navigator.setInitialPose(init_pose)
    mission.navigator.waitUntilNav2Active()


    try:
        mission.run()
    finally:
        rclpy.shutdown()

if __name__ == '__main__':
    main()
