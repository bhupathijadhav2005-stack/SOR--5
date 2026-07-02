#!/usr/bin/env python3
"""
custom_ik_mover.py
Session-5 Q2 Part B - Math to the Rescue (Custom Inverse Kinematics)

Bypasses MoveIt 2 entirely. Computes joint angles for a target end-effector
coordinate using the closed-form 4-DOF IK derived in Part A, then publishes
a JointTrajectory message straight to the arm's joint_trajectory_controller.

Kinematic chain (from bme_ros2_simple_arm URDF):
    base_link -> shoulder_pan_joint (z, base yaw)
              -> shoulder_lift_joint (y, pitch)  -- upper_arm_link, L1 = 0.2 m
              -> elbow_joint (y, pitch)          -- forearm_link,   L2 = 0.25 m
              -> wrist_joint (y, pitch)          -- wrist->EE,      L3 = 0.175 m
Shoulder pivot height above base_link origin: h0 = 0.075 m

Author: Badavath Bhupathi (24B4218)
"""

import math

import rclpy
from rclpy.node import Node
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from builtin_interfaces.msg import Duration


class CustomIKMover(Node):

    def __init__(self):
        super().__init__('custom_ik_mover')

        self.publisher_ = self.create_publisher(
            JointTrajectory, '/arm_controller/joint_trajectory', 10)

        self.joint_names = [
            'shoulder_pan_joint',
            'shoulder_lift_joint',
            'elbow_joint',
            'wrist_joint',
        ]

        # Link geometry taken from mogi_arm.xacro (meters)
        self.h0 = 0.075   # base_link origin -> shoulder pivot (fixed on the pan axis)
        self.L1 = 0.20    # upper_arm_link length
        self.L2 = 0.25    # forearm_link length
        self.L3 = 0.175   # wrist_link -> end_effector_link

        self._sent = False
        self.timer = self.create_timer(1.0, self._tick)

    def solve_ik(self, x, y, z, approach_pitch_deg):
        """
        Closed-form IK for the 4-DOF arm.

        approach_pitch_deg: desired absolute pitch (deg, measured from the
        vertical base axis) of the final link, i.e. of the end effector.
        This is the redundant free parameter that resolves the 3-position /
        4-joint ambiguity (see Part A). 90 deg = end effector horizontal.
        """
        # Step 1: base rotation, solved first (per the assignment hint)
        theta1 = math.atan2(y, x)

        # Reduce to the 2D problem inside the vertical plane picked by theta1
        r = math.hypot(x, y)
        z_prime = z - self.h0
        phi = math.radians(approach_pitch_deg)

        # Step 2: strip off the last link to get the wrist point
        r_w = r - self.L3 * math.sin(phi)
        z_w = z_prime - self.L3 * math.cos(phi)

        # Step 3: standard 2-link planar IK (shoulder_lift + elbow) for (r_w, z_w)
        d2 = r_w ** 2 + z_w ** 2
        cos_gamma2 = (d2 - self.L1 ** 2 - self.L2 ** 2) / (2 * self.L1 * self.L2)
        cos_gamma2 = max(-1.0, min(1.0, cos_gamma2))   # guard against float noise
        gamma2 = math.acos(cos_gamma2)                 # elbow-down branch

        gamma1 = math.atan2(r_w, z_w) - math.atan2(
            self.L2 * math.sin(gamma2), self.L1 + self.L2 * math.cos(gamma2))

        # Step 4: wrist angle makes up the remainder of the approach pitch
        gamma3 = phi - gamma1 - gamma2

        return theta1, gamma1, gamma2, gamma3

    def _tick(self):
        if self._sent:
            return

        # ---- Target end-effector coordinate (base_link frame) + approach pitch ----
        target_x, target_y, target_z = 0.35, 0.0, 0.15
        approach_pitch_deg = 90.0   # end effector horizontal, reaching forward

        theta1, gamma1, gamma2, gamma3 = self.solve_ik(
            target_x, target_y, target_z, approach_pitch_deg)

        joint_limits = {
            'shoulder_pan_joint': (-3.14, 3.14),
            'shoulder_lift_joint': (-1.5708, 1.5708),
            'elbow_joint': (-2.3562, 2.3562),
            'wrist_joint': (-2.3562, 2.3562),
        }
        solved = dict(zip(self.joint_names, (theta1, gamma1, gamma2, gamma3)))
        for name, value in solved.items():
            lo, hi = joint_limits[name]
            if not (lo <= value <= hi):
                self.get_logger().warn(
                    f'{name} solution {value:.3f} rad is outside limits [{lo}, {hi}]!')

        self.get_logger().info(
            'IK solution (rad): pan={:.3f}, shoulder_lift={:.3f}, elbow={:.3f}, wrist={:.3f}'
            .format(theta1, gamma1, gamma2, gamma3))

        traj = JointTrajectory()
        traj.joint_names = self.joint_names

        point = JointTrajectoryPoint()
        point.positions = [theta1, gamma1, gamma2, gamma3]
        point.velocities = [0.0, 0.0, 0.0, 0.0]
        point.time_from_start = Duration(sec=3, nanosec=0)
        traj.points.append(point)

        self.publisher_.publish(traj)
        self.get_logger().info('Published trajectory to /arm_controller/joint_trajectory')
        self._sent = True


def main(args=None):
    rclpy.init(args=args)
    node = CustomIKMover()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    main()
