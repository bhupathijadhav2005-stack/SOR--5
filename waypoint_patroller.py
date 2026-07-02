import math

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient

from geometry_msgs.msg import PoseStamped
from nav2_msgs.action import FollowWaypoints


def yaw_to_quaternion(yaw: float):
    """Convert a yaw angle (radians) to a (x, y, z, w) quaternion."""
    return (0.0, 0.0, math.sin(yaw / 2.0), math.cos(yaw / 2.0))


class WaypointPatroller(Node):

    def __init__(self):
        super().__init__('waypoint_patroller')

        self._action_client = ActionClient(self, FollowWaypoints, 'follow_waypoints')
        self._last_waypoint_index = -1

        # ---- Hardcoded patrol route: (x, y, theta[rad]) in the 'map' frame ----
        # Edit these to match free space in your own map.
        raw_waypoints = [
            (1.5, 0.0, 0.0),
            (1.5, 1.5, math.radians(90)),
            (0.0, 1.5, math.radians(180)),
        ]

        self.waypoints = [self._make_pose(x, y, theta) for x, y, theta in raw_waypoints]

    def _make_pose(self, x, y, theta):
        pose = PoseStamped()
        pose.header.frame_id = 'map'
        pose.pose.position.x = x
        pose.pose.position.y = y
        pose.pose.position.z = 0.0
        qx, qy, qz, qw = yaw_to_quaternion(theta)
        pose.pose.orientation.x = qx
        pose.pose.orientation.y = qy
        pose.pose.orientation.z = qz
        pose.pose.orientation.w = qw
        return pose

    def send_patrol_goal(self):
        self.get_logger().info('Waiting for the follow_waypoints action server...')
        self._action_client.wait_for_server()

        stamp = self.get_clock().now().to_msg()
        for wp in self.waypoints:
            wp.header.stamp = stamp

        goal_msg = FollowWaypoints.Goal()
        goal_msg.poses = self.waypoints

        self.get_logger().info('Navigating to Waypoint 1...')
        send_goal_future = self._action_client.send_goal_async(
            goal_msg, feedback_callback=self.feedback_callback)
        send_goal_future.add_done_callback(self.goal_response_callback)

    def goal_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().error('Patrol goal was rejected by the action server.')
            rclpy.shutdown()
            return
        self.get_logger().info('Patrol goal accepted.')
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self.get_result_callback)

    def feedback_callback(self, feedback_msg):
        current = feedback_msg.feedback.current_waypoint
        if current != self._last_waypoint_index:
            if self._last_waypoint_index >= 0:
                self.get_logger().info(f'Waypoint {self._last_waypoint_index + 1} Reached!')
            if current < len(self.waypoints):
                self.get_logger().info(f'Navigating to Waypoint {current + 1}...')
            self._last_waypoint_index = current

    def get_result_callback(self, future):
        result = future.result().result
        missed = list(result.missed_waypoints)
        if missed:
            self.get_logger().warn(f'Patrol finished with missed waypoints: {missed}')
        else:
            self.get_logger().info(f'Waypoint {len(self.waypoints)} Reached!')
            self.get_logger().info('Patrol complete — all waypoints reached successfully.')
        rclpy.shutdown()


def main(args=None):
    rclpy.init(args=args)
    node = WaypointPatroller()
    node.send_patrol_goal()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    main()
