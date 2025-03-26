# Copyright (c) 2020-2023, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#
import argparse
import sys
import os
import json

# 사용자 정의 USD 파일 경로
WAREHOUSE_USD_PATH = "/home/yeseul/Documents/my_carter_navigation.usd"

import carb
from isaacsim import SimulationApp

CONFIG = {"renderer": "RayTracedLighting", "headless": False}

# Example ROS2 bridge sample demonstrating the manual loading of Warehouse Navigation scenario
simulation_app = SimulationApp(CONFIG)
import omni
from omni.isaac.core import SimulationContext
from omni.isaac.core.utils.extensions import enable_extension

# enable ROS2 bridge extension
enable_extension("omni.isaac.ros2_bridge")

simulation_app.update()

# 사용자 정의 USD 파일 로드
usd_path = WAREHOUSE_USD_PATH
if not os.path.exists(usd_path):
    carb.log_error(f"Could not find USD file at path: {usd_path}")
    simulation_app.close()
    sys.exit()

omni.usd.get_context().open_stage(usd_path, None)

# Wait two frames so that stage starts loading
simulation_app.update()
simulation_app.update()

print("Loading stage...")
from omni.isaac.core.utils.stage import is_stage_loading

while is_stage_loading():
    simulation_app.update()
print("Loading Complete")

simulation_context = SimulationContext(stage_units_in_meters=1.0)

simulation_app.update()

stage = omni.usd.get_context().get_stage()

from pxr import Gf

# 모든 prim 정보 추출
def extract_all_prims(root_path="/World", exclude_paths=["/World/Nova_Carter_ROS"]):
    root_prim = stage.GetPrimAtPath(root_path)
    if not root_prim:
        print(f"'{root_path}' 경로에 prim이 없습니다.")
        return []
    
    # prim 정보를 저장할 리스트
    prim_data = []
    
    # 모든 prim을 재귀적으로 순회하는 함수
    def traverse_prims(prim, depth=0):
        prim_path = str(prim.GetPath())
        
        # 제외할 경로인지 확인
        for exclude_path in exclude_paths:
            if prim_path.startswith(exclude_path):
                print(f"제외된 경로: {prim_path}")
                return
        
        # 현재 prim 정보 저장
        prim_info = {
            "path": prim_path,
            "type": str(prim.GetTypeName()),
            "position": None
        }
        
        # 위치 속성 가져오기
        translate_attr = prim.GetAttribute("xformOp:translate")
        if translate_attr and translate_attr.IsValid():
            translation = translate_attr.Get()
            if isinstance(translation, Gf.Vec3d) or isinstance(translation, Gf.Vec3f):
                prim_info["position"] = [float(translation[0]), float(translation[1]), float(translation[2])]
        
        # 유효한 위치 정보가 있는 경우만 추가
        if prim_info["position"] is not None:
            prim_data.append(prim_info)
        
        # 자식 prim 순회
        for child in prim.GetChildren():
            traverse_prims(child, depth + 1)
    
    # 루트 prim부터 순회 시작
    traverse_prims(root_prim)
    return prim_data

# 모든 prim 정보 추출 (Carter 로봇 제외)
all_prims = extract_all_prims(exclude_paths=["/World/Nova_Carter_ROS"])

# 추출된 프림 수 출력
print(f"Carter 로봇을 제외한 총 {len(all_prims)}개의 prim 정보를 추출했습니다.")

# JSON 파일로 저장
output_file = "/home/yeseul/Downloads/isaac-sim-standalone@4.2.0-rc.18+release.16044.3b2ed111.gl.linux-x86_64.release/standalone_examples/api/omni.isaac.ros2_bridge/warehouse_prim_info.json"
with open(output_file, "w") as json_file:
    json.dump(all_prims, json_file, indent=4)

print(f"총 {len(all_prims)}개의 prim 정보가 '{output_file}' 파일에 저장되었습니다.")

# 일부 prim 정보 출력 (처음 10개)
print("\n처음 10개 prim 정보 샘플:")
for i, prim in enumerate(all_prims[:10]):
    print(f"{i+1}. {prim['path']} - 위치: {prim['position']}")

simulation_context.play()

simulation_app.update()

while simulation_app.is_running():
    # runs with a realtime clock
    simulation_app.update()

simulation_context.stop()
simulation_app.close()