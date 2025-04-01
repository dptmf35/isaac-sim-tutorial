#!/usr/bin/env python3

import os
import time
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from nav2_msgs.action import NavigateToPose
from geometry_msgs.msg import PoseStamped, Twist
from std_msgs.msg import String
from builtin_interfaces.msg import Duration

class CarterBTNode(Node):
    def __init__(self):
        super().__init__('carter_bt_node')
        
        # 로봇 제어를 위한 Publisher
        self.cmd_vel_publisher = self.create_publisher(Twist, 'cmd_vel', 10)
        
        # Nav2 ActionClient 생성
        self.nav_client = ActionClient(self, NavigateToPose, 'navigate_to_pose')
        
        # 상태 관리
        self.is_navigating = False
        self.current_goal = None
        self.waypoints = [
            {'x': 1.0, 'y': 2.0, 'z': 0.0, 'name': 'Point 1'},
            {'x': 4.0, 'y': 1.0, 'z': 0.0, 'name': 'Point 2'},
            {'x': 3.0, 'y': 5.0, 'z': 0.0, 'name': 'Point 3'},
            {'x': 0.0, 'y': 0.0, 'z': 0.0, 'name': 'Home'}
        ]
        
        # 타이머 생성 - 행동 트리 실행
        self.timer = self.create_timer(1.0, self.execute_bt)
        self.waypoint_index = 0
        
        self.get_logger().info('Carter BT Node initialized')
        
    def execute_bt(self):
        """행동 트리의 루트 노드 역할 - 가장 상위 수준의 의사결정"""
        if self.is_navigating:
            # 이미 네비게이션 중이면 아무것도 하지 않음
            return
            
        if self.waypoint_index < len(self.waypoints):
            # 다음 웨이포인트로 이동
            waypoint = self.waypoints[self.waypoint_index]
            self.get_logger().info(f"Navigating to {waypoint['name']}: ({waypoint['x']}, {waypoint['y']})")
            self.navigate_to_pose(waypoint['x'], waypoint['y'])
        else:
            # 모든 웨이포인트 방문 완료
            self.get_logger().info("Patrol complete!")
            self.waypoint_index = 0  # 다시 처음부터
            
    def navigate_to_pose(self, x, y):
        """지정된 위치로 이동하는 Action 실행"""
        # Nav2 Action Server가 준비될 때까지 대기
        self.nav_client.wait_for_server()
        
        # 목표 생성
        goal_pose = PoseStamped()
        goal_pose.header.frame_id = 'map'
        goal_pose.header.stamp = self.get_clock().now().to_msg()
        goal_pose.pose.position.x = float(x)
        goal_pose.pose.position.y = float(y)
        goal_pose.pose.position.z = 0.0
        goal_pose.pose.orientation.w = 1.0
        
        # Action 목표 메시지 생성
        goal_msg = NavigateToPose.Goal()
        goal_msg.pose = goal_pose
        
        # 타임아웃 설정 (옵션)
        timeout = Duration()
        timeout.sec = 300  # 5분
        goal_msg.behavior_tree = ""  # 기본 내비게이션 BT 사용
        
        self.is_navigating = True
        self.current_goal = {'x': x, 'y': y}
        
        # 목표 전송 및 콜백 설정
        send_goal_future = self.nav_client.send_goal_async(
            goal_msg, 
            feedback_callback=self.feedback_callback
        )
        send_goal_future.add_done_callback(self.goal_response_callback)
        
    def goal_response_callback(self, future):
        """목표 요청에 대한 응답 처리"""
        goal_handle = future.result()
        
        if not goal_handle.accepted:
            self.get_logger().error('Goal was rejected!')
            self.is_navigating = False
            return
            
        self.get_logger().info('Goal accepted!')
        
        # 결과 비동기 요청
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self.get_result_callback)
        
    def get_result_callback(self, future):
        """네비게이션 완료 후 결과 처리"""
        result = future.result().result
        status = future.result().status
        
        if status == 4:  # 성공적으로 완료
            self.get_logger().info(f'Navigation succeeded to point ({self.current_goal["x"]}, {self.current_goal["y"]})')
            self.waypoint_index += 1  # 다음 웨이포인트로
        else:
            self.get_logger().error(f'Navigation failed with status: {status}')
            
        # 잠시 대기 후 상태 리셋
        time.sleep(1.0)  # 웨이포인트 도달 후 1초 대기
        self.is_navigating = False
        
    def feedback_callback(self, feedback_msg):
        """네비게이션 진행 중 피드백 처리"""
        feedback = feedback_msg.feedback
        # 필요한 경우 여기서 피드백 정보 처리
        # self.get_logger().info(f'Distance remaining: {feedback.distance_remaining}')
    
    def stop_robot(self):
        """로봇 긴급 정지"""
        msg = Twist()
        msg.linear.x = 0.0
        msg.angular.z = 0.0
        self.cmd_vel_publisher.publish(msg)
        self.get_logger().info('Emergency stop!')

def main(args=None):
    rclpy.init(args=args)
    carter_bt_node = CarterBTNode()
    
    try:
        rclpy.spin(carter_bt_node)
    except KeyboardInterrupt:
        carter_bt_node.stop_robot()
    except Exception as e:
        carter_bt_node.get_logger().error(f'Unexpected error: {e}')
    finally:
        carter_bt_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()