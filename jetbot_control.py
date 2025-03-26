# Copyright (c) 2023, Your Name. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#
import argparse
import threading
import time

from isaacsim import SimulationApp

parser = argparse.ArgumentParser()
parser.add_argument("--test", default=False, action="store_true", help="Run in test mode")
parser.add_argument("--headless", default=False, action="store_true", help="Run in headless mode")
args, unknown = parser.parse_known_args()

simulation_app = SimulationApp({"headless": args.headless})

import carb
import numpy as np
from omni.isaac.core import World
from omni.isaac.nucleus import get_assets_root_path
from omni.isaac.wheeled_robots.controllers.differential_controller import DifferentialController
from omni.isaac.wheeled_robots.robots import WheeledRobot

# 조향 모드 상수 정의
MODE_STOP = 0
MODE_FORWARD = 1
MODE_FAST_FORWARD = 2
MODE_BACKWARD = 3
MODE_ROTATE_IN_PLACE = 4
MODE_TURN_RIGHT = 5
MODE_TURN_LEFT = 6
MODE_AUTO_SQUARE = 7  # 새로운 모드: 사각형 자동 주행
MODE_FOLLOW_WAYPOINTS = 8  # 새로운 모드: 웨이포인트 따라가기

# 모드별 설명
MODE_DESCRIPTIONS = {
    MODE_STOP: "정지",
    MODE_FORWARD: "직진",
    MODE_FAST_FORWARD: "빠르게 직진",
    MODE_BACKWARD: "후진",
    MODE_ROTATE_IN_PLACE: "제자리에서 돌기",
    MODE_TURN_RIGHT: "오른쪽으로 가기",
    MODE_TURN_LEFT: "왼쪽으로 가기",
    MODE_AUTO_SQUARE: "사각형 자동 주행",
    MODE_FOLLOW_WAYPOINTS: "웨이포인트 따라가기"
}

# 조향 파라미터 (속도 조절 가능하도록 변수로 설정)
base_speed_factor = 1.0  # 속도 조절 계수
FORWARD_SPEED = 0.05  # 기본 전진 속도
FAST_FORWARD_SPEED = 0.1  # 빠른 전진 속도
BACKWARD_SPEED = -0.05  # 후진 속도
ROTATION_SPEED = np.pi / 12  # 회전 속도
TURN_LINEAR_SPEED = 0.03  # 회전 시 선속도
TURN_ANGULAR_SPEED = np.pi / 24  # 회전 시 각속도

# 자동 주행 관련 변수
auto_drive_step = 0
auto_drive_timer = 0
auto_drive_steps = [
    {"mode": MODE_FORWARD, "duration": 100},  # 직진
    {"mode": MODE_TURN_RIGHT, "duration": 50},  # 오른쪽 회전
    {"mode": MODE_FORWARD, "duration": 100},  # 직진
    {"mode": MODE_TURN_RIGHT, "duration": 50},  # 오른쪽 회전
    {"mode": MODE_FORWARD, "duration": 100},  # 직진
    {"mode": MODE_TURN_RIGHT, "duration": 50},  # 오른쪽 회전
    {"mode": MODE_FORWARD, "duration": 100},  # 직진
    {"mode": MODE_TURN_RIGHT, "duration": 50},  # 오른쪽 회전
    {"mode": MODE_STOP, "duration": 10}  # 정지
]

# 웨이포인트 관련 변수
waypoints = []
current_waypoint_idx = 0
waypoint_threshold = 0.1  # 웨이포인트 도달 판정 거리 (미터)
waypoint_rotation_threshold = 0.1  # 회전 완료 판정 각도 (라디안)
waypoint_state = "rotate"  # 'rotate' 또는 'move'
rotation_timeout = 0  # 회전 타임아웃 카운터 추가
max_rotation_time = 180  # 최대 회전 시간 (프레임 수)

# 월드 생성
my_world = World(stage_units_in_meters=1.0)
assets_root_path = get_assets_root_path()
if assets_root_path is None:
    carb.log_error("Could not find Isaac Sim assets folder")

# JetBot 로봇 추가
jetbot_asset_path = assets_root_path + "/Isaac/Robots/Jetbot/jetbot.usd"
my_jetbot = my_world.scene.add(
    WheeledRobot(
        prim_path="/World/Jetbot",
        name="my_jetbot",
        wheel_dof_names=["left_wheel_joint", "right_wheel_joint"],
        create_robot=True,
        usd_path=jetbot_asset_path,
        position=np.array([0, 0.0, 0.03]),
    )
)

# 바닥 추가
my_world.scene.add_default_ground_plane()

# 디퍼렌셜 컨트롤러 생성
my_controller = DifferentialController(name="simple_control", wheel_radius=0.03, wheel_base=0.1125)

# 월드 초기화
my_world.reset()

# 현재 모드 초기화
current_mode = MODE_STOP
print(f"초기 모드: {MODE_DESCRIPTIONS[current_mode]}")
print("사용 가능한 키:")
print("  0: 정지")
print("  1: 직진")
print("  2: 빠르게 직진")
print("  3: 후진")
print("  4: 제자리에서 돌기")
print("  5: 오른쪽으로 가기")
print("  6: 왼쪽으로 가기")
print("  7: 사각형 자동 주행")
print("  8: 웨이포인트 따라가기")
print("  +: 속도 증가")
print("  -: 속도 감소")
print("  w: 현재 위치를 웨이포인트로 저장")
print("  c: 웨이포인트 목록 초기화")
print("  i: 상태 정보 표시")
print("  q: 종료")

# 사용자 입력을 받는 함수
def get_user_input():
    global current_mode, running, base_speed_factor, waypoints, current_waypoint_idx
    
    while running:
        try:
            user_input = input()
            
            if user_input == "q":
                running = False
                print("프로그램을 종료합니다...")
                break
            
            elif user_input == "+":
                base_speed_factor += 0.1
                print(f"속도 증가: {base_speed_factor:.1f}x")
            
            elif user_input == "-":
                if base_speed_factor > 0.2:
                    base_speed_factor -= 0.1
                    print(f"속도 감소: {base_speed_factor:.1f}x")
                else:
                    print("최소 속도에 도달했습니다.")
            
            elif user_input == "w":
                # 현재 위치를 웨이포인트로 저장
                position = my_jetbot.get_world_pose()[0]
                waypoints.append(position)
                print(f"웨이포인트 #{len(waypoints)} 저장: {position}")
            
            elif user_input == "c":
                # 웨이포인트 목록 초기화
                waypoints = []
                current_waypoint_idx = 0
                print("웨이포인트 목록 초기화")
            
            elif user_input == "i":
                # 상태 정보 표시
                position, orientation = my_jetbot.get_world_pose()
                linear_velocity = my_jetbot.get_linear_velocity()
                angular_velocity = my_jetbot.get_angular_velocity()
                
                print("\n===== JetBot 상태 정보 =====")
                print(f"현재 모드: {MODE_DESCRIPTIONS[current_mode]}")
                print(f"위치: {position}")
                print(f"방향: {orientation}")
                print(f"선속도: {linear_velocity}, 크기: {np.linalg.norm(linear_velocity):.2f} m/s")
                print(f"각속도: {angular_velocity}, 크기: {np.linalg.norm(angular_velocity):.2f} rad/s")
                print(f"속도 계수: {base_speed_factor:.1f}x")
                print(f"저장된 웨이포인트: {len(waypoints)}개")
                print("===========================\n")
            
            else:
                try:
                    mode = int(user_input)
                    if mode in MODE_DESCRIPTIONS:
                        current_mode = mode
                        # 자동 주행 모드로 변경 시 초기화
                        if mode == MODE_AUTO_SQUARE:
                            auto_drive_step = 0
                            auto_drive_timer = 0
                            print(f"모드 변경: {MODE_DESCRIPTIONS[current_mode]} (사각형 경로 주행 시작)")
                        # 웨이포인트 따라가기 모드로 변경 시 초기화
                        elif mode == MODE_FOLLOW_WAYPOINTS:
                            if len(waypoints) > 0:
                                current_waypoint_idx = 0
                                waypoint_state = "rotate"
                                print(f"모드 변경: {MODE_DESCRIPTIONS[current_mode]} (웨이포인트 {len(waypoints)}개 따라가기 시작)")
                            else:
                                print("저장된 웨이포인트가 없습니다. 먼저 'w'를 눌러 웨이포인트를 저장하세요.")
                                current_mode = MODE_STOP
                        else:
                            print(f"모드 변경: {MODE_DESCRIPTIONS[current_mode]}")
                    else:
                        print(f"알 수 없는 모드: {mode}")
                        print("사용 가능한 모드: 0-8")
                except ValueError:
                    print("올바른 입력이 아닙니다. 0-8 사이의 숫자, +, -, w, c, i 또는 q를 입력하세요.")
        
        except Exception as e:
            print(f"입력 처리 중 오류 발생: {e}")

# 웨이포인트 관련 함수 수정
def get_direction_to_waypoint(current_position, waypoint):
    """현재 위치에서 웨이포인트까지의 방향 벡터를 계산"""
    direction = waypoint - current_position
    # 높이(z축) 차이는 무시하고 x-y 평면에서만 방향 계산
    direction[2] = 0
    return direction

def get_forward_vector(orientation):
    """쿼터니언 방향에서 전방 벡터 계산"""
    # 쿼터니언을 회전 행렬로 변환
    qw, qx, qy, qz = orientation
    
    # 회전 행렬의 첫 번째 열이 전방 벡터 (x 축 방향)
    forward_x = 1 - 2 * (qy * qy + qz * qz)
    forward_y = 2 * (qx * qy + qw * qz)
    forward_z = 2 * (qx * qz - qw * qy)
    
    forward = np.array([forward_x, forward_y, forward_z])
    # z 성분은 무시하고 정규화
    forward[2] = 0
    if np.linalg.norm(forward) > 0:
        forward = forward / np.linalg.norm(forward)
    return forward

def get_angle_between_vectors(vec1, vec2):
    """두 벡터 사이의 각도 계산 (라디안)"""
    # 벡터 정규화
    if np.linalg.norm(vec1) > 0 and np.linalg.norm(vec2) > 0:
        vec1_normalized = vec1 / np.linalg.norm(vec1)
        vec2_normalized = vec2 / np.linalg.norm(vec2)
        
        # 내적 계산 (값이 -1과 1 사이에 있도록 클램핑)
        dot_product = np.clip(np.dot(vec1_normalized, vec2_normalized), -1.0, 1.0)
        
        # 각도 계산
        angle = np.arccos(dot_product)
        return angle
    return 0.0

# 입력 스레드 시작
running = True
input_thread = threading.Thread(target=get_user_input)
input_thread.daemon = True
input_thread.start()

# 시뮬레이션 루프
i = 0
reset_needed = False
while simulation_app.is_running() and running:
    # 월드 스텝 실행
    my_world.step(render=True)
    
    if my_world.is_stopped() and not reset_needed:
        reset_needed = True
    
    if my_world.is_playing():
        if reset_needed:
            my_world.reset()
            my_controller.reset()
            reset_needed = False
            current_mode = MODE_STOP
        
        # 속도 계수 적용
        adjusted_forward_speed = FORWARD_SPEED * base_speed_factor
        adjusted_fast_forward_speed = FAST_FORWARD_SPEED * base_speed_factor
        adjusted_backward_speed = BACKWARD_SPEED * base_speed_factor
        adjusted_rotation_speed = ROTATION_SPEED * base_speed_factor
        adjusted_turn_linear_speed = TURN_LINEAR_SPEED * base_speed_factor
        adjusted_turn_angular_speed = TURN_ANGULAR_SPEED * base_speed_factor
        
        # 자동 주행 모드 처리
        if current_mode == MODE_AUTO_SQUARE:
            auto_drive_timer += 1
            current_step = auto_drive_steps[auto_drive_step]
            
            # 현재 단계의 지정된 시간이 지나면 다음 단계로
            if auto_drive_timer >= current_step["duration"]:
                auto_drive_step = (auto_drive_step + 1) % len(auto_drive_steps)
                auto_drive_timer = 0
                print(f"자동 주행: {MODE_DESCRIPTIONS[auto_drive_steps[auto_drive_step]['mode']]} 단계")
            
            # 현재 단계의 모드에 따라 로봇 제어
            step_mode = current_step["mode"]
            
            if step_mode == MODE_STOP:
                my_jetbot.apply_wheel_actions(my_controller.forward(command=[0.0, 0.0]))
            elif step_mode == MODE_FORWARD:
                my_jetbot.apply_wheel_actions(my_controller.forward(command=[adjusted_forward_speed, 0.0]))
            elif step_mode == MODE_TURN_RIGHT:
                my_jetbot.apply_wheel_actions(my_controller.forward(command=[0.0, -adjusted_rotation_speed]))
            
            if i % 60 == 0:
                print(f"자동 주행 중: 단계 {auto_drive_step+1}/{len(auto_drive_steps)}, 타이머: {auto_drive_timer}/{current_step['duration']}")
        
        # 웨이포인트 따라가기 모드 처리
        elif current_mode == MODE_FOLLOW_WAYPOINTS and len(waypoints) > 0:
            current_position, current_orientation = my_jetbot.get_world_pose()
            target_waypoint = waypoints[current_waypoint_idx]
            
            # 현재 웨이포인트까지의 방향 벡터
            direction_to_waypoint = get_direction_to_waypoint(current_position, target_waypoint)
            distance_to_waypoint = np.linalg.norm(direction_to_waypoint)
            
            # 로봇의 전방 벡터
            forward_vector = get_forward_vector(current_orientation)
            
            if waypoint_state == "rotate":
                # 웨이포인트 방향으로 회전
                angle = get_angle_between_vectors(forward_vector, direction_to_waypoint)
                
                # 회전 방향 결정 (외적의 부호로 판단)
                cross_product = np.cross(forward_vector[:2], direction_to_waypoint[:2])
                rotation_direction = 1 if cross_product < 0 else -1
                
                # 회전 타임아웃 증가
                rotation_timeout += 1
                
                if angle > waypoint_rotation_threshold and rotation_timeout < max_rotation_time:
                    # 회전 필요
                    my_jetbot.apply_wheel_actions(my_controller.forward(command=[0.0, rotation_direction * adjusted_rotation_speed]))
                    if i % 60 == 0:
                        print(f"웨이포인트 #{current_waypoint_idx+1} 방향으로 회전 중: 각도 {angle:.2f} rad, 방향: {rotation_direction}")
                else:
                    # 회전 완료 또는 타임아웃, 이동 단계로 전환
                    waypoint_state = "move"
                    rotation_timeout = 0
                    if rotation_timeout >= max_rotation_time:
                        print(f"회전 타임아웃, 웨이포인트 #{current_waypoint_idx+1}로 직접 이동 시작")
                    else:
                        print(f"회전 완료, 웨이포인트 #{current_waypoint_idx+1}로 이동 시작")
            
            elif waypoint_state == "move":
                if distance_to_waypoint > waypoint_threshold:
                    # 웨이포인트로 이동
                    # 약간의 방향 보정을 추가
                    angle = get_angle_between_vectors(forward_vector, direction_to_waypoint)
                    cross_product = np.cross(forward_vector[:2], direction_to_waypoint[:2])
                    rotation_direction = 1 if cross_product < 0 else -1
                    
                    # 각도가 크면 회전 성분 추가
                    angular_correction = 0
                    if angle > 0.3:  # 약 17도 이상 차이나면 보정
                        angular_correction = rotation_direction * adjusted_turn_angular_speed * 0.5
                    
                    my_jetbot.apply_wheel_actions(my_controller.forward(command=[adjusted_forward_speed, angular_correction]))
                    if i % 60 == 0:
                        print(f"웨이포인트 #{current_waypoint_idx+1}로 이동 중: 거리 {distance_to_waypoint:.2f} m, 각도 {angle:.2f} rad")
                else:
                    # 웨이포인트 도달, 다음 웨이포인트로
                    print(f"웨이포인트 #{current_waypoint_idx+1} 도달")
                    current_waypoint_idx = (current_waypoint_idx + 1) % len(waypoints)
                    waypoint_state = "rotate"
                    rotation_timeout = 0
                    print(f"다음 웨이포인트 #{current_waypoint_idx+1}로 이동 준비")
        
        # 일반 모드 처리
        else:
            if current_mode == MODE_STOP:
                # 정지
                my_jetbot.apply_wheel_actions(my_controller.forward(command=[0.0, 0.0]))
            
            elif current_mode == MODE_FORWARD:
                # 직진
                my_jetbot.apply_wheel_actions(my_controller.forward(command=[adjusted_forward_speed, 0.0]))
                if i % 60 == 0:  # 60 프레임마다 속도 출력 (약 1초)
                    print(f"선속도: {my_jetbot.get_linear_velocity()}")
            
            elif current_mode == MODE_FAST_FORWARD:
                # 빠르게 직진
                my_jetbot.apply_wheel_actions(my_controller.forward(command=[adjusted_fast_forward_speed, 0.0]))
                if i % 60 == 0:
                    print(f"선속도: {my_jetbot.get_linear_velocity()}")
            
            elif current_mode == MODE_BACKWARD:
                # 후진
                my_jetbot.apply_wheel_actions(my_controller.forward(command=[adjusted_backward_speed, 0.0]))
                if i % 60 == 0:
                    print(f"선속도: {my_jetbot.get_linear_velocity()}")
            
            elif current_mode == MODE_ROTATE_IN_PLACE:
                # 제자리에서 돌기
                my_jetbot.apply_wheel_actions(my_controller.forward(command=[0.0, adjusted_rotation_speed]))
                if i % 60 == 0:
                    print(f"각속도: {my_jetbot.get_angular_velocity()}")
            
            elif current_mode == MODE_TURN_RIGHT:
                # 오른쪽으로 가기
                my_jetbot.apply_wheel_actions(my_controller.forward(command=[adjusted_turn_linear_speed, -adjusted_turn_angular_speed]))
                if i % 60 == 0:
                    print(f"선속도: {my_jetbot.get_linear_velocity()}, 각속도: {my_jetbot.get_angular_velocity()}")
            
            elif current_mode == MODE_TURN_LEFT:
                # 왼쪽으로 가기
                my_jetbot.apply_wheel_actions(my_controller.forward(command=[adjusted_turn_linear_speed, adjusted_turn_angular_speed]))
                if i % 60 == 0:
                    print(f"선속도: {my_jetbot.get_linear_velocity()}, 각속도: {my_jetbot.get_angular_velocity()}")
        
        i += 1
    
    # 테스트 모드인 경우 한 번만 실행
    if args.test is True:
        break
    
    # 약간의 지연 추가 (CPU 사용량 감소)
    time.sleep(0.01)

# 시뮬레이션 종료
running = False
if input_thread.is_alive():
    input_thread.join(timeout=1.0)
simulation_app.close()