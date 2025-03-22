import json
import random
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from nav2_msgs.action import NavigateToPose
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped
from nav_msgs.msg import OccupancyGrid
from visualization_msgs.msg import Marker
import threading
import math

class GoalPublisher(Node):
    def __init__(self, prim_data):
        super().__init__('goal_publisher')
        self.prim_data = prim_data
        self.action_clients = {}
        self.costmaps = {}
        self.current_poses = {}
        self.lock = threading.Lock()

        # 코스트맵 구독
        self.create_subscription(OccupancyGrid, '/carter1/global_costmap/costmap', 
                                lambda msg: self.costmap_callback('carter1', msg), 10)
        self.create_subscription(OccupancyGrid, '/carter2/global_costmap/costmap', 
                                lambda msg: self.costmap_callback('carter2', msg), 10)
        self.create_subscription(OccupancyGrid, '/carter3/global_costmap/costmap', 
                                lambda msg: self.costmap_callback('carter3', msg), 10)

        # 현재 위치 구독 (amcl_pose)
        self.create_subscription(PoseWithCovarianceStamped, '/carter1/amcl_pose', 
                                lambda msg: self.pose_callback('carter1', msg), 10)
        self.create_subscription(PoseWithCovarianceStamped, '/carter2/amcl_pose', 
                                lambda msg: self.pose_callback('carter2', msg), 10)
        self.create_subscription(PoseWithCovarianceStamped, '/carter3/amcl_pose', 
                                lambda msg: self.pose_callback('carter3', msg), 10)

        # 마커 퍼블리셔
        self.marker_publishers = {}
        for robot in ['carter1', 'carter2', 'carter3']:
            self.marker_publishers[robot] = self.create_publisher(Marker, f'/{robot}/goal_marker', 10)

    def costmap_callback(self, robot_name, msg):
        with self.lock:
            self.costmaps[robot_name] = msg
            self.get_logger().debug(f"{robot_name} 코스트맵 수신")

    def pose_callback(self, robot_name, msg):
        with self.lock:
            self.current_poses[robot_name] = msg.pose.pose
            # self.get_logger().info(f"{robot_name} AMCL 위치 수신: x={msg.pose.pose.position.x}, y={msg.pose.pose.position.y}")

    def is_valid_position(self, x, y, costmap):
        px = int((x - costmap.info.origin.position.x) / costmap.info.resolution)
        py = int((y - costmap.info.origin.position.y) / costmap.info.resolution)
        if 0 <= px < costmap.info.width and 0 <= py < costmap.info.height:
            index = py * costmap.info.width + px
            if index < len(costmap.data):
                cost = costmap.data[index]
                return cost < 100
        return False

    def find_nearest_valid_position(self, base_x, base_y, robot_name):
        costmap = self.costmaps.get(robot_name)
        if not costmap:
            return base_x, base_y
        step = 0.1
        for r in range(1, 4):
            for angle in range(0, 360, 5):
                offset_x = r * step * math.cos(math.radians(angle))
                offset_y = r * step * math.sin(math.radians(angle))
                target_x = base_x + offset_x
                target_y = base_y + offset_y
                if self.is_valid_position(target_x, target_y, costmap):
                    self.get_logger().info(f"{robot_name}의 유효한 위치 발견: x={target_x}, y={target_y}")
                    return target_x, target_y
        return base_x, base_y

    def get_or_create_action_client(self, robot_name):
        with self.lock:
            if robot_name not in self.action_clients:
                self.action_clients[robot_name] = ActionClient(self, NavigateToPose, f'/{robot_name}/navigate_to_pose')
            return self.action_clients[robot_name]

    def find_position(self, location_name):
        for prim in self.prim_data:
            if prim['path'].endswith(location_name):
                return prim['position']
        return None

    def publish_goal_marker(self, robot_name, target_x, target_y):
        marker = Marker()
        marker.header.frame_id = 'map'
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = robot_name
        marker.id = 0
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.pose.position.x = target_x
        marker.pose.position.y = target_y
        marker.pose.position.z = 0.1  # 살짝 띄움
        marker.pose.orientation.w = 1.0
        marker.scale.x = 0.2  # 크기
        marker.scale.y = 0.2
        marker.scale.z = 0.2
        marker.color.r = 1.0  # 빨간색
        marker.color.g = 0.0
        marker.color.b = 0.0
        marker.color.a = 1.0
        marker.lifetime.sec = 0  # 영구 표시 (0은 사라지지 않음)
        self.marker_publishers[robot_name].publish(marker)
        self.get_logger().info(f"{robot_name}의 목표 마커 퍼블리시: x={target_x}, y={target_y}")

    def send_goal(self, robot_name, location_name):
        position = self.find_position(location_name)
        self.get_logger().info(f"위치 '{location_name}'을(를) 찾았습니다: {position}")
        if not position:
            self.get_logger().error(f"위치 '{location_name}'을(를) 찾을 수 없습니다.")
            return

        action_client = self.get_or_create_action_client(robot_name)
        if not action_client.wait_for_server(timeout_sec=10.0):
            self.get_logger().error(f"{robot_name}의 /navigate_to_pose 액션 서버에 연결할 수 없습니다.")
            return

        base_x, base_y = float(position[0]), float(position[1])
        target_x, target_y = self.find_nearest_valid_position(base_x, base_y, robot_name)

        pose = PoseStamped()
        pose.header.frame_id = 'map'
        pose.header.stamp = self.get_clock().now().to_msg()
        pose.pose.position.x = target_x
        pose.pose.position.y = target_y
        pose.pose.position.z = float(position[2]) if len(position) > 2 else 0.0
        pose.pose.orientation.w = 1.0

        goal_msg = NavigateToPose.Goal()
        goal_msg.pose = pose
        self.get_logger().info(f"{robot_name}의 /navigate_to_pose에 목표를 전송합니다: x={target_x}, y={target_y}")
        self.publish_goal_marker(robot_name, target_x, target_y)
        
        # 로깅을 위한 정보 저장
        self.navigation_info = getattr(self, 'navigation_info', {})
        self.navigation_info[robot_name] = {
            'target_x': target_x,
            'target_y': target_y,
            'active': True
        }
        
        # 로깅 타이머 시작 (이미 실행 중이 아니라면)
        if not hasattr(self, 'logging_timer'):
            self.logging_timer = self.create_timer(1.0, self.log_navigation_progress)
        
        future = action_client.send_goal_async(goal_msg)
        future.add_done_callback(lambda f: self.goal_response_callback(f, robot_name))

    def log_navigation_progress(self):
        """1초마다 모든 로봇의 내비게이션 진행 상황을 로그로 출력"""
        if not hasattr(self, 'navigation_info'):
            return
            
        active_robots = False
        for robot_name, info in list(self.navigation_info.items()):
            if not info['active']:
                continue
                
            active_robots = True
            if robot_name not in self.current_poses:
                continue
                
            current_pose = self.current_poses[robot_name]
            current_x = current_pose.position.x
            current_y = current_pose.position.y
            target_x = info['target_x']
            target_y = info['target_y']
            
            dx = target_x - current_x
            dy = target_y - current_y
            distance = math.sqrt(dx**2 + dy**2)
            
            self.get_logger().info(
                f"{robot_name} 현재 위치: x={current_x:.2f}, y={current_y:.2f}, "
                f"목표 위치: x={target_x:.2f}, y={target_y:.2f}, 남은 거리: {distance:.2f}m"
            )
            
            # 목표에 도달했는지 확인
            if distance <= 0.2:
                # self.get_logger().info(f"{robot_name} 목표 지점에 도달했습니다.")
                info['active'] = False
        
        # 모든 로봇이 목표에 도달했으면 타이머 중지
        if not active_robots and hasattr(self, 'logging_timer'):
            self.logging_timer.cancel()
            delattr(self, 'logging_timer')

    def goal_response_callback(self, future, robot_name):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().error(f"{robot_name}의 목표가 거부되었습니다.")
            # 내비게이션 정보에서 제거
            if hasattr(self, 'navigation_info') and robot_name in self.navigation_info:
                self.navigation_info[robot_name]['active'] = False
            return
        self.get_logger().info(f"{robot_name}의 목표가 수락되었습니다.")
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(lambda f: self.get_result_callback(f, robot_name))

    def get_result_callback(self, future, robot_name):
        try:
            result = future.result()
            if result.status == 4:
                self.get_logger().info(f"{robot_name}의 내비게이션 성공!")
            else:
                self.get_logger().warn(f"{robot_name}의 내비게이션이 실패했습니다: 상태 코드 {result.status}")
            
            # 내비게이션 정보에서 비활성화
            if hasattr(self, 'navigation_info') and robot_name in self.navigation_info:
                self.navigation_info[robot_name]['active'] = False
                
        except Exception as e:
            self.get_logger().error(f"{robot_name}의 결과 처리 중 오류 발생: {str(e)}")
            # 오류 발생 시에도 내비게이션 정보에서 비활성화
            if hasattr(self, 'navigation_info') and robot_name in self.navigation_info:
                self.navigation_info[robot_name]['active'] = False

def main():
    with open('hospital_prim_info.json', 'r') as f:
        prim_data = json.load(f)

    rclpy.init()
    node = GoalPublisher(prim_data)

    print("명령 입력 대기 중 (형식: robot_name location_name, 예: carter1 SM_Chart_Rack_01c3)")
    print("종료하려면 'quit' 입력")

    spin_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spin_thread.start()

    try:
        while rclpy.ok():
            user_input = input("> ").strip()
            if user_input.lower() == 'quit':
                break
            try:
                robot_name, location_name = user_input.split(maxsplit=1)
                node.send_goal(robot_name, location_name)
            except ValueError:
                print("잘못된 입력 형식입니다. 예: carter1 SM_Chart_Rack_01c3")
    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()
    print("GoalPublisher 종료")

if __name__ == '__main__':
    main()