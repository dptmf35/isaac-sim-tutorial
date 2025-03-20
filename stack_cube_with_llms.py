#!/usr/bin/env python3

import os
import sys
import argparse
import numpy as np

from isaacsim import SimulationApp
# init isaac sim
simulation_app = SimulationApp({"headless": False})

import carb
import omni.timeline
import omni.kit
from omni.isaac.core.utils.stage import create_new_stage
from omni.isaac.core.objects import DynamicCuboid, GroundPlane
from omni.isaac.franka.tasks import PickPlace
from omni.isaac.franka.controllers import PickPlaceController
from omni.isaac.core.world import World
from pxr import Gf, UsdLux

# OpenAI API 설정
import openai
openai.api_key = os.getenv("OPENAI_API_KEY")

class StackCubesLLMExample:
    def __init__(self):
        # 큐브 정보 초기화
        self.cubes = {
            "red": {"name": None, "object": None},
            "blue": {"name": None, "object": None},
            "yellow": {"name": None, "object": None}
        }
        self.source_cube = None
        self.target_cube = None

        # 시뮬레이션 설정
        self.setup_world()
        self.setup_task()
        self.process_user_input_with_llm()
        self.run_simulation()

    def setup_world(self):
        """월드 및 스테이지 설정"""
        create_new_stage()
        self.world = World(physics_dt=1.0/60.0, rendering_dt=1.0/60.0, stage_units_in_meters=1.0)

        # 기본 조명 추가
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
        pick_place_task = PickPlace(name="awesome_task")
        self.world.add_task(pick_place_task)
        self.world.reset()
        print("태스크 설정 완료")

    def parse_task_with_llm(self, user_input):
        """LLM을 사용해 사용자 입력 파싱"""
        try:
            if not openai.api_key:
                raise ValueError("OpenAI API 키가 설정되지 않았습니다.")
            
            client = openai.OpenAI(api_key=openai.api_key)
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": """
                    당신은 로봇 제어 시스템의 자연어 처리 모듈입니다. 
                    사용자의 명령을 분석하여 어떤 색상의 큐브를 어떤 색상의 큐브 위에 올려야 하는지 파악하세요.
                    응답은 JSON 형식으로:
                    {
                        "source_cube": "색상",
                        "target_cube": "색상"
                    }
                    가능한 색상: "red", "blue", "yellow"
                    """},
                    {"role": "user", "content": user_input}
                ],
                temperature=0
            )
            
            result = response.choices[0].message.content
            import json
            parsed_result = json.loads(result)
            return parsed_result["source_cube"], parsed_result["target_cube"]
        
        except Exception as e:
            print(f"LLM 처리 중 오류 발생: {e}")
            return None

    def process_user_input_with_llm(self):
        """사용자 입력 처리"""
        valid_colors = ["red", "blue", "yellow"]
        while True:
            print("\nPick & Place 태스크를 입력하세요 (e.g. '빨간색 큐브를 파란색 큐브 위에 올려줘')")
            user_input = input("입력 태스크: ")
            result = self.parse_task_with_llm(user_input)
            if result is not None:
                self.source_cube, self.target_cube = result
                if self.source_cube in valid_colors and self.target_cube in valid_colors:
                    break
                else:
                    print(f"오류: 유효하지 않은 색상입니다. 가능한 색상: {', '.join(valid_colors)}")
            else:
                print("입력을 처리하는 중 오류가 발생했습니다. 다시 시도해주세요.")
        
        print(f"태스크 파싱 결과: {self.source_cube} 큐브를 {self.target_cube} 큐브 위에 올립니다.")

    def setup_post_load(self):
        """환경 로드 후 초기화"""
        task_params = self.world.get_task("awesome_task").get_params()
        self.franka = self.world.scene.get_object(task_params["robot_name"]["value"])
        
        # 빨간색 큐브 설정
        self.cubes["red"]["name"] = task_params["cube_name"]["value"]
        red_cube = self.world.scene.get_object(self.cubes["red"]["name"])
        self.cubes["red"]["object"] = red_cube
        red_cube.get_applied_visual_material().set_color(color=np.array([1.0, 0.0, 0.0]))
        red_cube_scale = red_cube.get_local_scale()

        # 관측값 및 타겟 위치
        current_observations = self.world.get_observations()
        target_position = current_observations[self.cubes["red"]["name"]]["target_position"]

        # 파란색 큐브 추가
        self.cubes["blue"]["name"] = "blue_cube"
        blue_cube_position = np.array([target_position[0] + 0.1, target_position[1] + 0.1, target_position[2]])
        self.cubes["blue"]["object"] = self.world.scene.add(
            DynamicCuboid(
                prim_path="/World/blue_cube",
                name="blue_cube",
                position=blue_cube_position,
                scale=red_cube_scale,
                color=np.array([0.0, 0.0, 1.0])
            )
        )

        # 노란색 큐브 추가
        self.cubes["yellow"]["name"] = "yellow_cube"
        yellow_cube_position = np.array([blue_cube_position[0], blue_cube_position[1] - 0.15, blue_cube_position[2]])
        self.cubes["yellow"]["object"] = self.world.scene.add(
            DynamicCuboid(
                prim_path="/World/yellow_cube",
                name="yellow_cube",
                position=yellow_cube_position,
                scale=red_cube_scale,
                color=np.array([1.0, 1.0, 0.0])
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
        self.world.reset()
        self.setup_post_load()

        max_steps = 2000
        step_count = 0
        task_completed = False
        print("시뮬레이션 시작")

        for _ in range(10):
            self.world.step(render=True)

        while step_count < max_steps and not task_completed:
            current_observations = self.world.get_observations()
            source_cube_obj = self.cubes[self.source_cube]["object"]
            target_cube_obj = self.cubes[self.target_cube]["object"]

            source_cube_position = source_cube_obj.get_world_pose()[0]
            target_cube_position = target_cube_obj.get_world_pose()[0]
            placing_position = np.array([target_cube_position[0], target_cube_position[1], target_cube_position[2] + 0.05])
            current_joint_positions = current_observations[self.franka.name]["joint_positions"]

            actions = self.controller.forward(
                picking_position=source_cube_position,
                placing_position=placing_position,
                current_joint_positions=current_joint_positions,
            )
            self.franka.apply_action(actions)
            self.world.step(render=True)

            if self.controller.is_done():
                print("작업 완료!")
                task_completed = True
                for _ in range(100):
                    self.world.step(render=True)

            step_count += 1
            if step_count % 100 == 0:
                print(f"시뮬레이션 단계: {step_count}")

        print("시뮬레이션 종료")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stack Cubes LLM Example")
    parser.add_argument("--api-key", type=str, help="OpenAI API 키 (환경 변수 대신 사용)")
    args = parser.parse_args()

    if args.api_key:
        openai.api_key = args.api_key
        print("명령줄 인자에서 제공된 API 키를 사용합니다.")
    elif openai.api_key:
        print("환경 변수 OPENAI_API_KEY에서 API 키를 불러왔습니다.")
    else:
        print("오류: OpenAI API 키가 설정되지 않았습니다.")
        print("환경 변수 OPENAI_API_KEY를 설정하거나 --api-key 옵션을 사용하세요.")
        sys.exit(1)

    example = StackCubesLLMExample()
    simulation_app.close()