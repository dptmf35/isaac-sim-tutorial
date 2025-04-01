#!/usr/bin/env python3

import argparse
import time
import sys
import os
from threading import Thread

def start_ros_node():
    """별도 스레드에서 ROS2 노드 실행"""
    import rclpy
    from carter_bt_nav.carter_bt_node import CarterBTNode
    
    rclpy.init()
    node = CarterBTNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

def main():
    parser = argparse.ArgumentParser(description="Carter BT Navigation in Isaac Sim")
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--headless", action="store_true")
    args, unknown = parser.parse_known_args()

    # Isaac Sim 임포트
    from isaacsim import SimulationApp
    simulation_app = SimulationApp({"renderer": "RayTracedLighting", "headless": args.headless})
    
    import carb
    import omni
    import omni.graph.core as og
    from omni.isaac.core import SimulationContext
    from omni.isaac.core.utils.extensions import enable_extension
    from omni.isaac.nucleus import get_assets_root_path
    
    # ROS2 브릿지 확장 활성화
    enable_extension("omni.isaac.ros2_bridge")
    
    # Nav2 확장 활성화 (필요한 경우)
    # enable_extension("omni.isaac.ros2_bridge.nav2")
    
    simulation_app.update()
    
    # 에셋 루트 폴더 찾기
    assets_root_path = get_assets_root_path()
    if assets_root_path is None:
        carb.log_error("에셋 폴더를 찾을 수 없습니다")
        simulation_app.close()
        exit()
    
    # Carter USD 로드
    usd_path = assets_root_path + "/Isaac/Samples/ROS2/Scenario/carter_warehouse_navigation.usd"
    omni.usd.get_context().open_stage(usd_path, None)
    
    # 스테이지 로딩 대기
    simulation_app.update()
    simulation_app.update()
    
    print("스테이지 로드 중...")
    from omni.isaac.core.utils.stage import is_stage_loading
    while is_stage_loading():
        simulation_app.update()
    print("로드 완료")
    
    # 시뮬레이션 컨텍스트 설정
    simulation_context = SimulationContext(stage_units_in_meters=1.0)
    
    # ROS2 노드 시작 (별도 스레드에서)
    ros_thread = Thread(target=start_ros_node)
    ros_thread.daemon = True
    ros_thread.start()
    
    # 시뮬레이션 시작
    simulation_context.play()
    simulation_context.step()
    
    # 워밍업
    print("시뮬레이션 워밍업...")
    for frame in range(60):
        simulation_context.step()
    
    # 메인 시뮬레이션 루프
    print("시뮬레이션 실행 중...")
    frame = 0
    while simulation_app.is_running():
        simulation_context.step(render=True)
        
        if args.test and frame > 1000:  # 테스트 모드에서는 1000프레임 후 종료
            break
            
        frame += 1
    
    # 종료 처리
    simulation_context.stop()
    simulation_app.close()

if __name__ == "__main__":
    main()