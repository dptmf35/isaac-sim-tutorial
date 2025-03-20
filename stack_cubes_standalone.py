#!/usr/bin/env python3

import os
import sys
import argparse
import numpy as np

# 새로운 방식으로 Isaac Sim 초기화 (isaacsim 모듈 사용)
from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": False})  # 헤드리스 모드 비활성화

# 이제 Isaac Sim 모듈 임포트 가능
import carb
import omni.timeline
import omni.kit
from omni.isaac.core.utils.stage import create_new_stage
from omni.isaac.core.objects import DynamicCuboid
from omni.isaac.core.utils.prims import create_prim
from omni.isaac.franka.tasks import PickPlace
from omni.isaac.franka.controllers import PickPlaceController
from omni.isaac.core.world import World
from omni.isaac.core.objects import GroundPlane
from pxr import Gf, UsdLux


class StackCubesExample:
    def __init__(self):
        # 사용자 선택 변수 초기화
        self.target_cube = None
        
        # 월드 및 시뮬레이션 설정
        self.setup_world()
        
        # 태스크 설정
        self.setup_task()
        
        # 사용자 입력 받기
        self.get_user_choice()
        
        # 시뮬레이션 실행
        self.run_simulation()
    
    def setup_world(self):
        """월드 및 스테이지 설정"""
        # 새 스테이지 생성
        create_new_stage()
        
        # 월드 생성
        self.world = World(physics_dt=1.0/60.0, rendering_dt=1.0/60.0, stage_units_in_meters=1.0)
        
        # 기본 라이트 추가
        stage = self.world.stage
        light_prim = UsdLux.DistantLight.Define(stage, "/World/Light")
        light_prim.CreateIntensityAttr(500.0)
        light_prim.CreateAngleAttr(0.53)
        light_prim.CreateColorAttr(Gf.Vec3f(1.0, 1.0, 0.9))
        
        # 그라운드 플레인 추가
        self.ground_plane = GroundPlane("/World/GroundPlane", z_position=0)
        
        print("월드 설정 완료")
    
    def setup_task(self):
        """태스크 및 로봇 설정"""
        # 태스크 추가
        pick_place_task = PickPlace(name="awesome_task")
        self.world.add_task(pick_place_task)
        
        # 월드 빌드
        self.world.reset()
        
        print("태스크 설정 완료")
    
    def get_user_choice(self):
        """사용자 입력 받기"""
        print("\n어떤 큐브 위에 빨간색 블록을 쌓을까요?")
        print("(1) 노란색 큐브")
        print("(2) 파란색 큐브")
        
        while self.target_cube is None:
            try:
                choice = input("선택하세요 (1 또는 2): ")
                if choice == "1":
                    self.target_cube = "yellow"
                    print("노란색 큐브를 선택했습니다.")
                elif choice == "2":
                    self.target_cube = "blue"
                    print("파란색 큐브를 선택했습니다.")
                else:
                    print("잘못된 선택입니다. 1 또는 2를 입력하세요.")
            except Exception as e:
                print(f"오류 발생: {e}")
                print("다시 시도하세요.")
    
    def setup_post_load(self):
        """환경 로드 후 초기화"""
        # 태스크 파라미터 가져오기
        task_params = self.world.get_task("awesome_task").get_params()
        self.franka = self.world.scene.get_object(task_params["robot_name"]["value"])
        self.red_cube_name = task_params["cube_name"]["value"]
        
        # 기존 큐브 색상 변경 (빨간색으로)
        red_cube = self.world.scene.get_object(self.red_cube_name)
        red_cube.get_applied_visual_material().set_color(color=np.array([1.0, 0.0, 0.0]))
        
        # 기존 큐브의 크기 확인
        red_cube_scale = red_cube.get_local_scale()
        print(f"Red cube scale: {red_cube_scale}")
        
        # 관측값 가져오기
        current_observations = self.world.get_observations()
        # 타겟 위치 가져오기
        target_position = current_observations[self.red_cube_name]["target_position"]
        print(f"Target position: {target_position}")
        
        # 파란색 큐브 추가 - 타겟 위치 근처에 배치
        self.blue_cube_name = "blue_cube"
        blue_cube_prim_path = "/World/" + self.blue_cube_name
        
        # 타겟 위치에서 약간 옆으로 이동한 위치 계산
        blue_cube_position = np.array([
            target_position[0] + 0.1,  # 타겟 위치에서 x축으로 0.1m 이동
            target_position[1] + 0.1,  # 타겟 위치에서 y축으로 0.1m 이동
            target_position[2]         # 타겟 위치와 동일한 높이
        ])
        
        # 파란색 큐브 객체 생성 - 직접 씬에 추가
        self.blue_cube = self.world.scene.add(
            DynamicCuboid(
                prim_path=blue_cube_prim_path,
                name=self.blue_cube_name,
                position=blue_cube_position,
                scale=red_cube_scale,
                color=np.array([0.0, 0.0, 1.0])  # 직접 색상 지정
            )
        )
        
        # 노란색 큐브 추가 - 파란색 큐브 뒤에 배치
        self.yellow_cube_name = "yellow_cube"
        yellow_cube_prim_path = "/World/" + self.yellow_cube_name
        
        # 파란색 큐브 뒤에 위치 계산 (y축으로 약간 뒤에 배치)
        yellow_cube_position = np.array([
            blue_cube_position[0],          # 파란색 큐브와 같은 x 좌표
            blue_cube_position[1] - 0.15,   # 파란색 큐브보다 y축으로 0.15m 뒤에 배치
            blue_cube_position[2]           # 파란색 큐브와 같은 높이
        ])
        
        # 노란색 큐브 객체 생성 - 직접 씬에 추가
        self.yellow_cube = self.world.scene.add(
            DynamicCuboid(
                prim_path=yellow_cube_prim_path,
                name=self.yellow_cube_name,
                position=yellow_cube_position,
                scale=red_cube_scale,  # 빨간색/파란색 큐브와 같은 크기
                color=np.array([1.0, 1.0, 0.0])  # 노란색 (RGB: 1,1,0)
            )
        )
        
        # 컨트롤러 설정
        self.controller = PickPlaceController(
            name="pick_place_controller",
            gripper=self.franka.gripper,
            robot_articulation=self.franka,
        )
        
        print("환경 초기화 완료")
        print(f"선택한 타겟 큐브: {self.target_cube}")
    
    def run_simulation(self):
        """시뮬레이션 실행"""
        # 월드 리셋 및 초기화
        self.world.reset()
        self.setup_post_load()
        
        # 시뮬레이션 실행 (my_application.py와 유사한 방식)
        max_steps = 2000  # 최대 시뮬레이션 단계 수
        step_count = 0
        task_completed = False
        
        print("시뮬레이션 시작")
        
        while step_count < max_steps and not task_completed:
            # 관측값 가져오기
            current_observations = self.world.get_observations()
            
            # 선택한 큐브에 따라 타겟 위치 결정
            if self.target_cube == "yellow":
                target_cube_pose = self.yellow_cube.get_world_pose()
            else:  # "blue"
                target_cube_pose = self.blue_cube.get_world_pose()
            
            target_cube_position = target_cube_pose[0]
            
            # 선택한 큐브 위에 놓을 위치 계산
            target_position = np.array([
                target_cube_position[0],
                target_cube_position[1],
                target_cube_position[2] + 0.05
            ])
            
            # 컨트롤러에 빨간색 큐브 집기 및 선택한 큐브 위에 놓기 명령
            actions = self.controller.forward(
                picking_position=current_observations[self.red_cube_name]["position"],
                placing_position=target_position,
                current_joint_positions=current_observations[self.franka.name]["joint_positions"],
            )
            
            # 로봇에 액션 적용
            self.franka.apply_action(actions)
            
            # 물리 및 렌더링 단계 실행 (중요: render=True로 설정)
            self.world.step(render=True)
            
            # 작업 완료 확인
            if self.controller.is_done():
                print("작업 완료!")
                task_completed = True
                # 작업 완료 후에도 몇 단계 더 실행하여 결과 확인
                for i in range(100):
                    self.world.step(render=True)
            
            step_count += 1
            
            # 디버깅 정보 (100단계마다 출력)
            if step_count % 100 == 0:
                print(f"시뮬레이션 단계: {step_count}")
        
        print("시뮬레이션 종료")


if __name__ == "__main__":
    # 예제 실행
    example = StackCubesExample()
    
    # 정리
    simulation_app.close()