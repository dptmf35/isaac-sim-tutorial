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

parser = argparse.ArgumentParser()
parser.add_argument(
    "--environment",
    type=str,
    choices=["hospital", "office"],
    default="hospital",
    help="Choice of navigation environment.",
)
args, _ = parser.parse_known_args()

HOSPITAL_USD_PATH = "/Isaac/Samples/ROS2/Scenario/multiple_robot_carter_hospital_navigation.usd"
OFFICE_USD_PATH = "/Isaac/Samples/ROS2/Scenario/multiple_robot_carter_office_navigation.usd"

if args.environment == "hospital":
    ENV_USD_PATH = HOSPITAL_USD_PATH
elif args.environment == "office":
    ENV_USD_PATH = OFFICE_USD_PATH

import carb
from isaacsim import SimulationApp

CONFIG = {"renderer": "RayTracedLighting", "headless": False}

# Example ROS2 bridge sample demonstrating the manual loading of Multiple Robot Navigation scenario
simulation_app = SimulationApp(CONFIG)
import omni
from omni.isaac.core import SimulationContext
from omni.isaac.core.utils.extensions import enable_extension
from omni.isaac.nucleus import get_assets_root_path

# enable ROS2 bridge extension
enable_extension("omni.isaac.ros2_bridge")

simulation_app.update()

# Locate assets root folder to load sample
assets_root_path = get_assets_root_path()
if assets_root_path is None:
    carb.log_error("Could not find Isaac Sim assets folder")
    simulation_app.close()
    sys.exit()

usd_path = assets_root_path + ENV_USD_PATH
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

import json
from pxr import Gf
# /World/hospital 경로의 prim 가져오기
hospital_prim = stage.GetPrimAtPath("/World/hospital")
if not hospital_prim:
    print("'/World/hospital' 경로에 prim이 없습니다.")
else:
    # prim 정보를 저장할 리스트
    prim_data = []

    # /World/hospital 아래의 모든 prim 순회
    for prim in hospital_prim.GetAllChildren():
        prim_info = {
            "path": str(prim.GetPath()),
            "type": prim.GetTypeName(),
            "position": None
        }

        # 위치 속성 가져오기
        translate_attr = prim.GetAttribute("xformOp:translate")
        if translate_attr:
            translation = translate_attr.Get()
            if isinstance(translation, Gf.Vec3d):
                prim_info["position"] = [translation[0], translation[1], translation[2]]

        prim_data.append(prim_info)

    # JSON 파일로 저장
    with open("hospital_prim_info.json", "w") as json_file:
        json.dump(prim_data, json_file, indent=4)

    print("Prim 정보가 'hospital_prim_info.json' 파일에 저장되었습니다.")




simulation_context.play()

simulation_app.update()

while simulation_app.is_running():

    # runs with a realtime clock
    simulation_app.update()

simulation_context.stop()
simulation_app.close()
