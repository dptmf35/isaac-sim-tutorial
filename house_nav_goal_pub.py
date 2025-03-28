#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
import json
import os
import openai
from rclpy.action import ActionClient
from nav2_msgs.action import NavigateToPose
from action_msgs.msg import GoalStatus
import threading
import time

class HouseNavGoalPublisher(Node):
    def __init__(self):
        super().__init__('house_nav_goal_publisher')
        
        # Map configuration
        self.map_resolution = 0.05  # Map resolution (meters/pixel)
        self.map_origin = [-6.975, -6.475, 0.0]  # Map origin (meters)
        self.map_height = 250  # Map height in pixels (needed for y-axis flipping)
        
        self.get_logger().info(f'Map resolution: {self.map_resolution}, origin: {self.map_origin}, height: {self.map_height}')
        
        # Set up OpenAI API key
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        if self.openai_api_key:
            self.get_logger().info("OpenAI API key loaded from environment variable.")
            openai.api_key = self.openai_api_key
        else:
            self.get_logger().error("OpenAI API key is not set. This application requires a valid API key.")
            self.get_logger().error("Please set the OPENAI_API_KEY environment variable and restart.")
        
        # Create ROS2 action client
        self.action_client = ActionClient(self, NavigateToPose, 'navigate_to_pose')
        
        # Load room information
        self.room_info = self.load_room_info()
        
        # Room name to number mapping
        self.room_mapping = {
            'storage': 'room1',
            'study': 'room2',
            'bedroom': 'room3',
            'bathroom': 'room4',
            'bedroom3': 'room5',
            'bedroom2': 'room6',
            'kitchen': 'room7',
            'livingroom': 'room8'
        }
        
        # Room number to name mapping
        self.room_number_mapping = {
            'room1': 'storage',
            'room2': 'study',
            'room3': 'bedroom',
            'room4': 'bathroom',
            'room5': 'bedroom3',
            'room6': 'bedroom2',
            'room7': 'kitchen',
            'room8': 'livingroom'
        }
        
        # Load object information
        self.prim_info = self.load_prim_info()
        
        # Task tracking
        self.current_task = None
        self.task_completed = threading.Event()
        self.task_result = None
        
        self.get_logger().info('House Navigation Goal Publisher has been initialized')
        self.get_logger().info('Available rooms:')
        for room_name, room_number in self.room_mapping.items():
            self.get_logger().info(f'- {room_name} ({room_number})')

    def load_room_info(self):
        """Load room waypoint information from JSON file"""
        json_path = '/home/yeseul/Documents/IsaacSim-ros_workspaces/humble_ws/src/navigation/carter_navigation/params/house_waypoints.json'
        
        self.get_logger().info(f'Loading room info from: {json_path}')
        try:
            with open(json_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            self.get_logger().error(f'Failed to load room information: {e}')
            self.get_logger().warn("Using hardcoded room information")
            return {
                "room1": {"x": 40, "y": 119},
                "room2": {"x": 49, "y": 52},
                "room3": {"x": 52, "y": 193},
                "room4": {"x": 238, "y": 109},
                "room5": {"x": 221, "y": 187},
                "room6": {"x": 225, "y": 52},
                "room7": {"x": 140, "y": 49},
                "room8": {"x": 134, "y": 138}
            }

    def load_prim_info(self):
        """Load object (prim) information from JSON file"""
        json_path = '/home/yeseul/Downloads/isaac-sim-standalone@4.2.0-rc.18+release.16044.3b2ed111.gl.linux-x86_64.release/standalone_examples/samsung_house_env/house_prim_info.json'
        
        self.get_logger().info(f'Loading prim info from: {json_path}')
        try:
            with open(json_path, 'r') as f:
                prim_data = json.load(f)
                
                # Log number of Xform type objects
                xform_count = sum(1 for prim in prim_data if prim.get('type') == 'Xform')
                self.get_logger().info(f'Loaded {len(prim_data)} prims, {xform_count} of them are Xform type')
                
                return prim_data
        except Exception as e:
            self.get_logger().error(f'Failed to load prim information: {e}')
            return []

    def parse_natural_language_command(self, user_input):
        """Parse natural language command using LLM"""
        # Check for API key
        if not self.openai_api_key:
            self.get_logger().error("OpenAI API key is not set. Cannot process commands.")
            return None
            
        try:
            client = openai.OpenAI(api_key=self.openai_api_key)
            
            # First determine command type (location or object)
            pre_analysis_prompt = """
            You are a natural language processing module for a robot navigation system.
            Your task is to determine if the user's command is asking to go to a room location or to a specific object.
            
            Available rooms:
            - storage (room1)
            - study (room2)
            - bedroom (room3)
            - bathroom (room4)
            - bedroom3 (room5)
            - bedroom2 (room6)
            - kitchen (room7)
            - livingroom (room8)
            
            Return ONLY "location" if the user wants to go to a room, or "object" if the user wants to go to a specific object.
            Do not include any explanation, just reply with the single word "location" or "object".
            
            Examples:
            - "주방으로 가" -> "location"
            - "서재로 이동" -> "location"
            - "냉장고로 가" -> "object"
            - "TV 찾아줘" -> "object"
            - "침실3으로 가줘" -> "location"
            - "침실3의 침대로 이동해줘" -> "object"
            """
            
            pre_analysis = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": pre_analysis_prompt},
                    {"role": "user", "content": user_input}
                ],
                temperature=0
            )
            
            command_type = pre_analysis.choices[0].message.content.strip().lower()
            print(f"Command type determined: {command_type}")
            
            # Handle location-based commands
            if command_type == "location":
                location_prompt = """
                You are a natural language processing module for a robot navigation system.
                Parse the user's command to determine which room they want to go to.
                
                Available rooms:
                - storage (room1)
                - study (room2)
                - bedroom (room3)
                - bathroom (room4)
                - bedroom3 (room5)
                - bedroom2 (room6)
                - kitchen (room7)
                - livingroom (room8)
                
                Return ONLY a valid JSON object with this structure:
                {"type": "location", "target": "ROOM_NAME"}
                where ROOM_NAME is one of: storage, study, bedroom, bathroom, bedroom3, bedroom2, kitchen, livingroom
                
                Examples:
                - User: "go to the kitchen" -> {"type": "location", "target": "kitchen"}
                - User: "서재로 이동" -> {"type": "location", "target": "study"}
                - User: "침실3으로 가줘" -> {"type": "location", "target": "bedroom3"}
                """
                
                response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": location_prompt},
                        {"role": "user", "content": user_input}
                    ],
                    temperature=0
                )
            
            # Handle object-based commands
            else:
                # Filter only Xform type objects
                xform_prims = []
                if self.prim_info:
                    for prim in self.prim_info:
                        if prim.get('type') == 'Xform':
                            xform_prims.append(prim)
                
                # Use all Xform object paths
                prim_paths = [prim['path'] for prim in xform_prims]
                
                object_prompt = f"""
                You are a natural language processing module for a robot navigation system.
                Parse the user's command to determine which object they want to go to.
                
                Available objects in the environment (all paths):
                {json.dumps(prim_paths, indent=2, ensure_ascii=False)}
                
                Return ONLY a valid JSON object with this structure:
                {{"type": "object", "target": "OBJECT_PATH"}}
                where OBJECT_PATH is the EXACT full path of the object from the list above.
                
                IMPORTANT:
                - Return the EXACT full path from the list, not a partial or modified path.
                - If the user mentions "TV", "television", etc., find the exact TV object path from the list.
                - If the user mentions a room with an object (like "bedroom3's bed"), find the bed object within bedroom3.
                - Don't make up paths that aren't in the list.
                
                Examples:
                - User: "TV 찾아줘" -> {{"type": "object", "target": "[EXACT TV PATH FROM LIST]"}}
                - User: "침실3의 침대로 이동해줘" -> {{"type": "object", "target": "[EXACT BED PATH IN BEDROOM3 FROM LIST]"}}
                """
                
                response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": object_prompt},
                        {"role": "user", "content": user_input}
                    ],
                    temperature=0
                )
            
            # Extract JSON from the response
            result = response.choices[0].message.content.strip()
            print(f"Raw LLM response: {result}")
            
            # Find start and end of JSON
            try:
                json_start = result.find('{')
                json_end = result.rfind('}')
                
                if json_start >= 0 and json_end > json_start:
                    json_str = result[json_start:json_end+1]
                    parsed_result = json.loads(json_str)
                    print(f"LLM parsed result: {parsed_result}")
                    return parsed_result
                else:
                    self.get_logger().error("Could not extract JSON from LLM response.")
                    return None
            except json.JSONDecodeError as e:
                self.get_logger().error(f"JSON parsing error: {e}")
                return None
            
        except Exception as e:
            self.get_logger().error(f"Error during LLM processing: {str(e)}")
            return None

    def get_room_coordinates(self, room_name):
        """Get coordinates for a room by name"""
        room_number = self.room_mapping.get(room_name.lower())
        if not room_number:
            self.get_logger().error(f'Room not found: {room_name}')
            return None
            
        room_data = self.room_info.get(room_number)
        if not room_data:
            self.get_logger().error(f'Coordinates not found for room: {room_name}')
            return None
            
        return room_data['x'], room_data['y']

    def get_object_coordinates(self, object_path):
        """Find coordinates based on object path - using exact path matching"""
        # Exact path matching
        for prim in self.prim_info:
            if prim['path'] == object_path:
                self.get_logger().info(f"Found exact object path: '{object_path}'")
                
                # Check if position field exists and has minimum length of 2
                if 'position' in prim and isinstance(prim['position'], list) and len(prim['position']) >= 2:
                    # Check if coordinates are within map range (-15~15)
                    x, y = prim['position'][0], prim['position'][1]
                    if abs(x) > 15 or abs(y) > 15:
                        self.get_logger().warn(f"Object coordinates out of map bounds: ({x}, {y}), limiting to [-15, 15] range")
                        x = max(min(x, 15), -15)
                        y = max(min(y, 15), -15)
                        position = [x, y]
                        if len(prim['position']) > 2:
                            position.append(prim['position'][2])
                    else:
                        position = prim['position']
                        
                    return position, prim.get('orientation', [1.0, 0.0, 0.0, 0.0])
                else:
                    self.get_logger().error(f"Invalid position data for object '{object_path}'")
                    return [0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]  # Default values
        
        # Partial path matching (fallback)
        object_path_lower = object_path.lower()
        best_match = None
        for prim in self.prim_info:
            if object_path_lower in prim['path'].lower():
                best_match = prim
                break
        
        if best_match:
            self.get_logger().info(f"Partial path match: '{object_path}' → '{best_match['path']}'")
            
            # Check if position field exists and has minimum length of 2
            if 'position' in best_match and isinstance(best_match['position'], list) and len(best_match['position']) >= 2:
                # Check if coordinates are within map range (-15~15)
                x, y = best_match['position'][0], best_match['position'][1]
                if abs(x) > 15 or abs(y) > 15:
                    self.get_logger().warn(f"Object coordinates out of map bounds: ({x}, {y}), limiting to [-15, 15] range")
                    x = max(min(x, 15), -15)
                    y = max(min(y, 15), -15)
                    position = [x, y]
                    if len(best_match['position']) > 2:
                        position.append(best_match['position'][2])
                else:
                    position = best_match['position']
                    
                return position, best_match.get('orientation', [1.0, 0.0, 0.0, 0.0])
            else:
                self.get_logger().error(f"Invalid position data for object '{best_match['path']}'")
                return [0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]  # Default values
        
        self.get_logger().warn(f"Could not find object path '{object_path}'")
        return [0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]  # Default values

    async def send_goal(self, position, orientation=None, is_object_position=False):
        """Asynchronously send navigation goal to Nav2"""
        if not self.action_client.wait_for_server(timeout_sec=10.0):
            self.get_logger().error('Action server not available')
            return False

        # Initialize task state
        self.task_completed.clear()
        self.task_result = None

        goal_msg = NavigateToPose.Goal()
        goal_msg.pose.header.frame_id = 'map'
        goal_msg.pose.header.stamp = self.get_clock().now().to_msg()
        
        try:
            # Set position
            if is_object_position:
                # Object-based coordinates are already in map frame, no transformation needed
                # Validate position data
                if position is None or not isinstance(position, list) or len(position) < 2:
                    self.get_logger().error(f'Invalid coordinate data: {position}')
                    self.task_completed.set()
                    self.task_result = False
                    return False
                    
                # Ensure coordinates are within map bounds
                x = float(position[0])
                y = float(position[1])
                z = float(position[2]) if len(position) > 2 else 0.0
                
                # Limit coordinates to map range (-15~15)
                x = max(min(x, 15), -15)
                y = max(min(y, 15), -15)
                
                goal_msg.pose.pose.position.x = x
                goal_msg.pose.pose.position.y = y
                goal_msg.pose.pose.position.z = z
                self.get_logger().info(f'Object coordinates (map frame): ({x:.2f}, {y:.2f})')
            else:
                # Room coordinates are pixel-based and need transformation to map frame
                # Y-axis is flipped
                if position is None or not isinstance(position, list) or len(position) < 2:
                    self.get_logger().error(f'Invalid coordinate data: {position}')
                    self.task_completed.set()
                    self.task_result = False
                    return False
                
                pixel_x = float(position[0])
                pixel_y = self.map_height - float(position[1])  # Flip y-axis
                
                # Convert pixel coordinates to meters and apply origin
                x = pixel_x * self.map_resolution + self.map_origin[0]
                y = pixel_y * self.map_resolution + self.map_origin[1]
                z = float(position[2]) if len(position) > 2 else self.map_origin[2]
                
                # Limit coordinates to map range (-15~15)
                x = max(min(x, 15), -15)
                y = max(min(y, 15), -15)
                
                goal_msg.pose.pose.position.x = x
                goal_msg.pose.pose.position.y = y
                goal_msg.pose.pose.position.z = z
                
                self.get_logger().info(f'Original pixel coordinates: ({position[0]}, {position[1]})')
                self.get_logger().info(f'Flipped pixel coordinates: ({pixel_x}, {pixel_y}) (y-axis flipped)')
                self.get_logger().info(f'Transformed map coordinates: ({x:.2f}, {y:.2f})')
            
            # Set orientation
            if orientation and isinstance(orientation, list) and len(orientation) >= 4:
                goal_msg.pose.pose.orientation.w = orientation[0]
                goal_msg.pose.pose.orientation.x = orientation[1]
                goal_msg.pose.pose.orientation.y = orientation[2]
                goal_msg.pose.pose.orientation.z = orientation[3]
            else:
                goal_msg.pose.pose.orientation.w = 1.0
                goal_msg.pose.pose.orientation.x = 0.0
                goal_msg.pose.pose.orientation.y = 0.0
                goal_msg.pose.pose.orientation.z = 0.0

            self.get_logger().info(f'Publishing goal to map coordinates: ({goal_msg.pose.pose.position.x:.2f}, {goal_msg.pose.pose.position.y:.2f})')
            
            # Send goal
            send_goal_future = self.action_client.send_goal_async(goal_msg)
            # Wait for goal acceptance
            await send_goal_future
            
            if not send_goal_future.done():
                self.get_logger().error('Goal sending did not complete')
                self.task_completed.set()
                self.task_result = False
                return False
                
            goal_handle = send_goal_future.result()
            if not goal_handle.accepted:
                self.get_logger().error('Goal rejected')
                self.task_completed.set()
                self.task_result = False
                return False
                
            self.get_logger().info('Goal accepted')
            
            # Store goal handle
            self.current_task = goal_handle
            
            # Set up callback for result
            get_result_future = goal_handle.get_result_async()
            get_result_future.add_done_callback(self._handle_task_result)
            
            return True
        except Exception as e:
            self.get_logger().error(f'Error sending goal: {str(e)}')
            import traceback
            traceback.print_exc()
            self.task_completed.set()
            self.task_result = False
            return False

    def _handle_task_result(self, future):
        """Handle navigation task result"""
        try:
            result = future.result().result
            status = future.result().status
            
            if status == GoalStatus.STATUS_SUCCEEDED:
                self.get_logger().info('Navigation task completed successfully')
                self.task_result = True
            else:
                status_name = self._get_status_name(status)
                self.get_logger().warn(f'Navigation task failed with status: {status_name} ({status})')
                self.task_result = False
                
        except Exception as e:
            self.get_logger().error(f'Error handling task result: {str(e)}')
            import traceback
            traceback.print_exc()
            self.task_result = False
            
        # Mark task as completed
        self.task_completed.set()
        
    def _get_status_name(self, status):
        """Convert status code to string representation"""
        status_map = {
            GoalStatus.STATUS_UNKNOWN: 'UNKNOWN',
            GoalStatus.STATUS_ACCEPTED: 'ACCEPTED',
            GoalStatus.STATUS_EXECUTING: 'EXECUTING',
            GoalStatus.STATUS_CANCELING: 'CANCELING',
            GoalStatus.STATUS_SUCCEEDED: 'SUCCEEDED',
            GoalStatus.STATUS_CANCELED: 'CANCELED',
            GoalStatus.STATUS_ABORTED: 'ABORTED'
        }
        return status_map.get(status, f'UNKNOWN ({status})')
        
    def wait_for_task_completion(self, timeout=None):
        """Wait for task completion"""
        if self.task_completed.wait(timeout):
            return self.task_result
        return None  # Timeout

    def send_goal_sync(self, position, orientation=None, is_object_position=False):
        """Synchronous version of goal publishing method"""
        import asyncio
        
        # Get or create event loop
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
        try:
            # Run async function synchronously
            return loop.run_until_complete(self.send_goal(position, orientation, is_object_position))
        except Exception as e:
            self.get_logger().error(f'Error in send_goal_sync: {e}')
            return False

def main():
    # Initialize ROS
    rclpy.init()
    node = HouseNavGoalPublisher()
    
    # Check API key
    if not node.openai_api_key:
        print("\n" + "!"*80)
        print("OpenAI API key is not set!")
        print("Please set the OPENAI_API_KEY environment variable and restart.")
        print("!"*80 + "\n")
        node.destroy_node()
        rclpy.shutdown()
        return
    
    print("\n" + "="*80)
    print("LLM-based Automatic Navigation System")
    print("="*80)
    print("Available command formats:")
    print("1. Location-based: 'go to the kitchen', 'move to bedroom', 'navigate to study', etc.")
    print("2. Object-based: 'go to the refrigerator', 'move to the chair', etc.")
    print("Type 'quit' or 'exit' to terminate")
    print("="*80 + "\n")
    
    # Task monitoring thread function
    def monitor_task():
        while rclpy.ok():
            if node.task_completed.is_set():
                # Display new command input message
                print("\n" + "-"*50)
                if node.task_result:
                    print("✅ Navigation task completed successfully!")
                else:
                    print("❌ Navigation task failed.")
                print("Please enter a new command.")
                print("-"*50)
                
                # Reset event
                node.task_completed.clear()
            
            # Short sleep
            time.sleep(0.5)
    
    # Start monitoring thread
    monitoring_thread = threading.Thread(target=monitor_task, daemon=True)
    monitoring_thread.start()
    
    # Start ROS spin thread
    spin_thread = threading.Thread(target=lambda: rclpy.spin(node), daemon=True)
    spin_thread.start()
    
    try:
        while rclpy.ok():
            user_input = input("\nInput command >>> ").strip()
            
            if user_input.lower() in ['quit', 'exit']:
                break
                
            if not user_input:
                continue
                
            print("-"*50)
            print(f"Input command: '{user_input}'")
            
            # Analyze command using LLM
            result = node.parse_natural_language_command(user_input)
            if not result:
                print("⚠️ Command analysis failed. Please try a different command.")
                continue
                
            # Check if there's a task in progress
            if node.current_task and not node.task_completed.is_set():
                print("⚠️ Previous navigation task is still in progress. Send new command? (y/n)")
                confirm = input(">>> ").strip().lower()
                if confirm != 'y':
                    print("Command canceled.")
                    continue
                    
                # Cancel previous task
                print("Canceling previous task and processing new command...")
                try:
                    if node.current_task:
                        cancel_future = node.current_task.cancel_goal_async()
                        # Give time for cancellation to process
                        time.sleep(1)
                except Exception as e:
                    print(f"Error while canceling task: {e}")
                
            # Navigate to target location
            if result['type'] == 'location':
                coords = node.get_room_coordinates(result['target'])
                if coords:
                    position = [coords[0], coords[1], 0.0]
                    # Convert pixel coordinates to physical coordinates (with y-axis flip)
                    pixel_x = float(coords[0])
                    pixel_y = node.map_height - float(coords[1])  # y-axis flip
                    
                    physical_x = pixel_x * node.map_resolution + node.map_origin[0]
                    physical_y = pixel_y * node.map_resolution + node.map_origin[1]
                    
                    print(f"Original pixel coordinates: ({coords[0]}, {coords[1]})")
                    print(f"Flipped pixel coordinates: ({pixel_x}, {pixel_y}) (y-axis flipped)")
                    print(f"Transformed map coordinates: ({physical_x:.2f}m, {physical_y:.2f}m)")
                    print(f"Moving to '{result['target']}' room")
                    
                    # Use synchronous version (is_object_position=False for room coordinates)
                    success = node.send_goal_sync(position, is_object_position=False)
                    if success:
                        print(f"🎯 Navigating to '{result['target']}'")
                    else:
                        print(f"⚠️ Failed to send goal.")
                else:
                    print(f"⚠️ Could not find location: '{result['target']}'")
            else:  # object-based
                position, orientation = node.get_object_coordinates(result['target'])
                if position:
                    # Object coordinates are already in map frame, no transformation needed
                    print(f"Sending goal to object: '{result['target']}' at position ({position[0]:.2f}, {position[1]:.2f})")
                    # Use synchronous version (is_object_position=True for object coordinates)
                    success = node.send_goal_sync(position, orientation, is_object_position=True)
                    if success:
                        print(f"🎯 Navigating to '{result['target']}'")
                    else:
                        print(f"⚠️ Failed to send goal.")
                else:
                    print(f"⚠️ Could not find object: '{result['target']}'")
            
            # Spin after sending command
            for _ in range(5):  # Spin 5 times to give chance for message processing
                rclpy.spin_once(node, timeout_sec=0.1)
                
            print("-"*50)
            
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"💥 Unexpected error: {str(e)}")
        # Print detailed error information
        import traceback
        traceback.print_exc()
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
