from omni.isaac.examples.base_sample import BaseSample
from omni.isaac.franka.tasks import PickPlace
from omni.isaac.franka.controllers import PickPlaceController
from omni.isaac.core.objects import DynamicCuboid
from omni.isaac.core.utils.prims import create_prim
import numpy as np
from pxr import UsdShade, Sdf


class HelloWorld(BaseSample):
    def __init__(self) -> None:
        super().__init__()
        return

    def setup_scene(self):
        world = self.get_world()
        # 기본 태스크 추가 (Isaac Sim 4.2 버전에서는 파라미터를 생성자에 직접 전달)
        pick_place_task = PickPlace(name="awesome_task")
        
        # 태스크를 월드에 추가
        world.add_task(pick_place_task)
        return

    async def setup_post_load(self):
        self._world = self.get_world()
        # 태스크 파라미터 가져오기
        task_params = self._world.get_task("awesome_task").get_params()
        self._franka = self._world.scene.get_object(task_params["robot_name"]["value"])
        self._red_cube_name = task_params["cube_name"]["value"]
        
        # 기존 큐브 색상 변경 (빨간색으로)
        red_cube = self._world.scene.get_object(self._red_cube_name)
        red_cube.get_applied_visual_material().set_color(color=np.array([1.0, 0.0, 0.0]))
        
        # 기존 큐브의 크기 확인
        red_cube_scale = red_cube.get_local_scale()
        print(f"Red cube scale: {red_cube_scale}")  # 디버깅용
        
        # 관측값 가져오기
        current_observations = self._world.get_observations()
        # 타겟 위치 가져오기
        target_position = current_observations[self._red_cube_name]["target_position"]
        print(f"Target position: {target_position}")  # 디버깅용
        
        # 파란색 큐브 추가 - 타겟 위치 근처에 배치
        self._blue_cube_name = "blue_cube"
        blue_cube_prim_path = "/World/" + self._blue_cube_name
        
        # 타겟 위치에서 약간 옆으로 이동한 위치 계산
        blue_cube_position = np.array([
            target_position[0] + 0.1,  # 타겟 위치에서 x축으로 0.1m 이동
            target_position[1] + 0.1,  # 타겟 위치에서 y축으로 0.1m 이동
            target_position[2]         # 타겟 위치와 동일한 높이
        ])
        
        # 프림 생성 - 기존 큐브와 동일한 크기로 설정
        create_prim(
            prim_path=blue_cube_prim_path,
            prim_type="Cube",
            attributes={"size": 1},  # 기본 크기를 작게 설정
        )
        
        # 파란색 큐브 객체 생성 - 기존 큐브와 동일한 스케일 사용
        self._blue_cube = DynamicCuboid(
            prim_path=blue_cube_prim_path,
            name=self._blue_cube_name,
            position=blue_cube_position,  # 타겟 위치 근처에 배치
            scale=red_cube_scale,  # 기존 큐브와 동일한 스케일 사용
        )
        
        # 씬에 파란색 큐브 추가
        self._world.scene.add(self._blue_cube)
        
        # 파란색 큐브 색상 설정
        visual_material = self._blue_cube.get_applied_visual_material()
        if visual_material:
            visual_material.set_color(color=np.array([0.0, 0.0, 1.0]))
        
        # 컨트롤러 설정
        self._controller = PickPlaceController(
            name="pick_place_controller",
            gripper=self._franka.gripper,
            robot_articulation=self._franka,
        )
        
        # 물리 시뮬레이션 콜백 등록
        self._world.add_physics_callback("sim_step", callback_fn=self.physics_step)
        await self._world.play_async()
        return

    async def setup_post_reset(self):
        self._controller.reset()
        await self._world.play_async()
        return

    def physics_step(self, step_size):
        # 관측값 가져오기
        current_observations = self._world.get_observations()
        
        # 파란색 큐브 위치 가져오기
        blue_cube_pose = self._blue_cube.get_world_pose()
        blue_cube_position = blue_cube_pose[0]
        print(f"blue_cube_position: {blue_cube_position}")
        
        # 기본 작업: 빨간색 큐브 집어서 목표 위치에 놓기
        actions = self._controller.forward(
            picking_position=current_observations[self._red_cube_name]["position"],
            placing_position=current_observations[self._red_cube_name]["target_position"],
            current_joint_positions=current_observations[self._franka.name]["joint_positions"],
        )
        


        # 로봇에 액션 적용
        self._franka.apply_action(actions)
        
        # 작업 완료 확인
        if self._controller.is_done():
            self._world.pause()
        return