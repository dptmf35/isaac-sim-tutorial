#!/usr/bin/env python3

import os
import sys
import argparse
import numpy as np
import re

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

# LLM 통합을 위한 라이브러리 (OpenAI API 사용)
import openai
import os

# OpenAI API 키 환경 변수에서 가져오기
openai.api_key = os.getenv("OPENAI_API_KEY")
if not openai.api_key:
    print("경고: OPENAI_API_KEY 환경 변수가 설정되지 않았습니다.")
    print("export OPENAI_API_KEY=your-api-key 명령으로 설정하세요.")

class StackCubesLLMExample:
    def __init__(self):
        # 큐브 정보 초기화
        self.cubes = {
            "red": {"name": None, "object": None},
            "blue": {"name": None, "object": None},
            "yellow": {"name": None, "object": None}
        }
        
        # 태스크 정보 초기화
        self.source_cube = None
        self.target_cube = None
        
        # 월드 및 시뮬레이션 설정
        self.setup_world()
        
        # 태스크 설정
        self.setup_task()
        
        # LLM을 통한 사용자 입력 처리
        self.process_user_input_with_llm()
        
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
    
    def parse_task_with_llm(self, user_input):
        """LLM을 사용하여 사용자 입력 파싱 (OpenAI API v1.0.0 이상 버전용)"""
        try:
            # API 키 확인
            if not openai.api_key:
                raise ValueError("OpenAI API 키가 설정되지 않았습니다.")
            
            # OpenAI API 호출 - 최신 버전 (v1.0.0 이상) 사용
            client = openai.OpenAI(api_key=openai.api_key)
            response = client.chat.completions.create(
                model="gpt-4o-mini",  # GPT-4o-mini 모델 사용
                messages=[
                    {"role": "system", "content": """
                    당신은 로봇 제어 시스템의 자연어 처리 모듈입니다. 
                    사용자의 명령을 분석하여 어떤 색상의 큐브를 어떤 색상의 큐브 위에 올려야 하는지 파악해야 합니다.
                    응답은 JSON 형식으로 다음과 같이 제공하세요:
                    {
                        "source_cube": "색상",
                        "target_cube": "색상"
                    }
                    가능한 색상은 "red", "blue", "yellow" 입니다.
                    """}, 
                    {"role": "user", "content": user_input}
                ],
                temperature=0
            )
            
            # 응답에서 JSON 추출
            result = response.choices[0].message.content
            import json
            parsed_result = json.loads(result)
            
            return parsed_result["source_cube"], parsed_result["target_cube"]
        
        except Exception as e:
            print(f"LLM 처리 중 오류 발생: {e}")
            # 오류 발생 시 규칙 기반 파싱으로 대체
            return self.parse_task_with_rule_based(user_input)
    
    def parse_task_with_rule_based(self, user_input):
        """규칙 기반 파싱 (LLM 대체용)"""
        # 소문자로 변환
        input_lower = user_input.lower()
        
        # 색상 패턴 찾기
        colors = ["red", "빨간", "빨강", "blue", "파란", "파랑", "yellow", "노란", "노랑"]
        color_mapping = {
            "빨간": "red", "빨강": "red", 
            "파란": "blue", "파랑": "blue", 
            "노란": "yellow", "노랑": "yellow"
        }
        
        found_colors = []
        for color in colors:
            if color in input_lower:
                # 한글 색상을 영어로 변환
                if color in color_mapping:
                    found_colors.append(color_mapping[color])
                else:
                    found_colors.append(color)
        
        # 중복 제거
        found_colors = list(dict.fromkeys(found_colors))
        
        if len(found_colors) >= 2:
            # 첫 번째 색상이 source, 두 번째 색상이 target
            return found_colors[0], found_colors[1]
        elif len(found_colors) == 1:
            # 하나의 색상만 발견된 경우, 기본값 사용
            if found_colors[0] == "red":
                return "red", "blue"
            else:
                return "red", found_colors[0]
        else:
            # 색상을 찾지 못한 경우 기본값 반환
            return "red", "blue"
    
    def process_user_input_with_llm(self):
        """LLM을 통한 사용자 입력 처리"""
        print("\nPick & Place 태스크를 입력하세요 (e.g. '빨간색 큐브를 파란색 큐브 위에 올려줘')")
        user_input = input("입력 태스크: ")
        
        # LLM 또는 규칙 기반 파싱 사용
        try:
            # OpenAI API 키가 설정되어 있으면 LLM 사용
            if openai.api_key:
                self.source_cube, self.target_cube = self.parse_task_with_llm(user_input)
            else:
                # API 키가 없으면 규칙 기반 파싱 사용
                self.source_cube, self.target_cube = self.parse_task_with_rule_based(user_input)
        except:
            # 오류 발생 시 규칙 기반 파싱 사용
            self.source_cube, self.target_cube = self.parse_task_with_rule_based(user_input)
        
        print(f"태스크 파싱 결과: {self.source_cube} 큐브를 {self.target_cube} 큐브 위에 올립니다.")
    
    def setup_post_load(self):
        """환경 로드 후 초기화"""
        # 태스크 파라미터 가져오기
        task_params = self.world.get_task("awesome_task").get_params()
        self.franka = self.world.scene.get_object(task_params["robot_name"]["value"])
        
        # 기존 큐브 정보 저장 (빨간색 큐브)
        self.cubes["red"]["name"] = task_params["cube_name"]["value"]
        red_cube = self.world.scene.get_object(self.cubes["red"]["name"])
        self.cubes["red"]["object"] = red_cube
        
        # 빨간색 큐브 색상 설정
        red_cube.get_applied_visual_material().set_color(color=np.array([1.0, 0.0, 0.0]))
        
        # 기존 큐브의 크기 확인
        red_cube_scale = red_cube.get_local_scale()
        
        # 관측값 가져오기
        current_observations = self.world.get_observations()
        # 타겟 위치 가져오기
        target_position = current_observations[self.cubes["red"]["name"]]["target_position"]
        
        # 파란색 큐브 추가 - 타겟 위치 근처에 배치
        self.cubes["blue"]["name"] = "blue_cube"
        blue_cube_prim_path = "/World/" + self.cubes["blue"]["name"]
        
        # 타겟 위치에서 약간 옆으로 이동한 위치 계산
        blue_cube_position = np.array([
            target_position[0] + 0.1,  # 타겟 위치에서 x축으로 0.1m 이동
            target_position[1] + 0.1,  # 타겟 위치에서 y축으로 0.1m 이동
            target_position[2]         # 타겟 위치와 동일한 높이
        ])
        
        # 파란색 큐브 객체 생성 - 직접 씬에 추가
        self.cubes["blue"]["object"] = self.world.scene.add(
            DynamicCuboid(
                prim_path=blue_cube_prim_path,
                name=self.cubes["blue"]["name"],
                position=blue_cube_position,
                scale=red_cube_scale,
                color=np.array([0.0, 0.0, 1.0])  # 직접 색상 지정
            )
        )
        
        # 노란색 큐브 추가 - 파란색 큐브 뒤에 배치
        self.cubes["yellow"]["name"] = "yellow_cube"
        yellow_cube_prim_path = "/World/" + self.cubes["yellow"]["name"]
        
        # 파란색 큐브 뒤에 위치 계산 (y축으로 약간 뒤에 배치)
        yellow_cube_position = np.array([
            blue_cube_position[0],          # 파란색 큐브와 같은 x 좌표
            blue_cube_position[1] - 0.15,   # 파란색 큐브보다 y축으로 0.15m 뒤에 배치
            blue_cube_position[2]           # 파란색 큐브와 같은 높이
        ])
        
        # 노란색 큐브 객체 생성 - 직접 씬에 추가
        self.cubes["yellow"]["object"] = self.world.scene.add(
            DynamicCuboid(
                prim_path=yellow_cube_prim_path,
                name=self.cubes["yellow"]["name"],
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
        print(f"태스크: {self.source_cube} 큐브를 {self.target_cube} 큐브 위에 올립니다.")
    
    def run_simulation(self):
        """시뮬레이션 실행"""
        # 월드 리셋 및 초기화
        self.world.reset()
        self.setup_post_load()
        
        # 시뮬레이션 실행
        max_steps = 2000  # 최대 시뮬레이션 단계 수
        step_count = 0
        task_completed = False
        
        print("시뮬레이션 시작")
        
        # 초기 몇 단계 실행하여 물리 시뮬레이션 안정화
        for i in range(10):
            self.world.step(render=True)
        
        while step_count < max_steps and not task_completed:
            # 관측값 가져오기
            current_observations = self.world.get_observations()
            
            # 소스 큐브와 타겟 큐브 위치 가져오기
            source_cube_obj = self.cubes[self.source_cube]["object"]
            target_cube_obj = self.cubes[self.target_cube]["object"]
            
            # 직접 큐브 위치 가져오기 (관측값 대신 객체에서 직접 가져옴)
            source_cube_pose = source_cube_obj.get_world_pose()
            source_cube_position = source_cube_pose[0]
            
            target_cube_pose = target_cube_obj.get_world_pose()
            target_cube_position = target_cube_pose[0]
            
            # 타겟 큐브 위에 놓을 위치 계산
            placing_position = np.array([
                target_cube_position[0],
                target_cube_position[1],
                target_cube_position[2] + 0.05
            ])
            
            # 현재 관절 위치 가져오기
            current_joint_positions = current_observations[self.franka.name]["joint_positions"]
            
            # 컨트롤러에 소스 큐브 집기 및 타겟 큐브 위에 놓기 명령
            actions = self.controller.forward(
                picking_position=source_cube_position,
                placing_position=placing_position,
                current_joint_positions=current_joint_positions,
            )
            
            # 로봇에 액션 적용
            self.franka.apply_action(actions)
            
            # 물리 및 렌더링 단계 실행
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
    # 명령줄 인자 파싱
    parser = argparse.ArgumentParser(description="Stack Cubes LLM Example")
    parser.add_argument("--api-key", type=str, help="OpenAI API 키 (환경 변수 대신 사용)")
    args = parser.parse_args()
    
    # 명령줄 인자로 API 키가 제공된 경우 우선 사용
    if args.api_key:
        openai.api_key = args.api_key
        print("명령줄 인자에서 제공된 API 키를 사용합니다.")
    elif openai.api_key:
        print("환경 변수 OPENAI_API_KEY에서 API 키를 불러왔습니다.")
    else:
        print("API 키가 설정되지 않았습니다. 규칙 기반 파싱을 사용합니다.")
    
    # 예제 실행
    example = StackCubesLLMExample()
    
    # 정리
    simulation_app.close()