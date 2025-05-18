import carla
import pygame
import numpy as np
import random
import time
import os
import torch

# Assuming your Dev folder is in PYTHONPATH or you adjust imports accordingly
from DQN_control.model import DQN
# Constants from environment.py that might be needed for observation processing
from environment import IMAGE_WIDTH, IMAGE_HEIGHT, N_ACTIONS # Make sure N_ACTIONS matches your model output

# Pygame settings
SCREEN_WIDTH = 1280
SCREEN_HEIGHT = 720
FONT_SIZE = 16

# CARLA settings
CLIENT_HOST = 'localhost'
CLIENT_PORT = 2000
CLIENT_TIMEOUT = 10.0
FIXED_DELTA_SECONDS = 0.05 # Must match synchronous mode setting

# Agent settings
# IMPORTANT: Update this path to your trained model
MODEL_PATH = "dqn_carla_model_final.pth" # Example path, adjust as needed
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def process_agent_observation(image_data, target_width, target_height):
    """
    Processes raw CARLA image data into the format expected by the DQN agent.
    This should be identical to the _process_image function in your environment.py
    """
    img = np.frombuffer(image_data.raw_data, dtype=np.dtype("uint8"))
    img = np.reshape(img, (image_data.height, image_data.width, 4))
    img = img[:, :, :3]  # Remove alpha channel

    # Grayscale
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Resize
    img_resized = cv2.resize(img_gray, (target_width, target_height))
    # Add channel dimension (H, W, 1)
    img_processed = np.expand_dims(img_resized, axis=-1)
    # Normalize
    img_processed = img_processed / 255.0
    return img_processed.astype(np.float32)

class HUD(object):
    def __init__(self, width, height):
        self.dim = (width, height)
        font = pygame.font.Font(pygame.font.get_default_font(), FONT_SIZE)
        font_name = 'courier' if os.name == 'nt' else 'mono'
        fonts = [x for x in pygame.font.get_fonts() if font_name in x]
        default_font = 'ubuntumono'
        mono = default_font if default_font in fonts else fonts[0]
        mono = pygame.font.match_font(mono)
        self._font_mono = pygame.font.Font(mono, FONT_SIZE)
        self._notifications = FadingText(font, (width, 40), (0, height - 40))
        self.help = HelpText(pygame.font.Font(mono, 16), width, height)
        self.server_fps = 0
        self.frame = 0
        self.simulation_time = 0
        self._show_info = True
        self._info_text = []
        self._server_clock = pygame.time.Clock()

    def on_world_tick(self, timestamp):
        self._server_clock.tick()
        self.server_fps = self._server_clock.get_fps()
        self.frame = timestamp.frame
        self.simulation_time = timestamp.elapsed_seconds

    def tick(self, world, clock):
        self._notifications.tick(world, clock)
        if not self._show_info:
            return
        t = world.vehicle.get_transform()
        v = world.vehicle.get_velocity()
        c = world.vehicle.get_control()

        heading = 'N' if abs(t.rotation.yaw) < 89.5 else ''
        heading += 'S' if abs(t.rotation.yaw) > 90.5 else ''
        heading += 'E' if 179.5 > t.rotation.yaw > 0.5 else ''
        heading += 'W' if -0.5 > t.rotation.yaw > -179.5 else ''
        
        colhist = world.collision_sensor.get_collision_history()
        collision = [colhist[x + self.frame - 200] for x in range(0, 200)]
        max_col = max(1.0, max(collision))
        collision = [x / max_col for x in collision]
        
        vehicles = world.world.get_actors().filter('vehicle.*')
        
        self._info_text = [
            'Server:  % 16.0f FPS' % self.server_fps,
            'Client:  % 16.0f FPS' % clock.get_fps(),
            '',
            'Vehicle: % 20s' % get_actor_display_name(world.vehicle, truncate=20),
            'Map:     % 20s' % world.world.get_map().name.split('/')[-1],
            'Sim time: % 12.2f' % self.simulation_time,
            '',
            'Speed:   % 15.0f km/h' % (3.6 * np.linalg.norm([v.x, v.y, v.z])),
            u'Heading:% 16.0f\N{DEGREE SIGN} % 2s' % (t.rotation.yaw, heading),
            'Location:% 20s' % ('(% 5.1f, % 5.1f)' % (t.location.x, t.location.y)),
            'Height:  % 18.0f m' % t.location.z,
            '',
            ('Throttle:', c.throttle, 0.0, 1.0),
            ('Steer:', c.steer, -1.0, 1.0),
            ('Brake:', c.brake, 0.0, 1.0),
            '',
            'Collision:',
            collision,
            '',
            'Number of vehicles: % 8d' % len(vehicles)
        ]
        if len(vehicles) > 1:
            self._info_text += ['Nearby vehicles:']
            distance = lambda l: np.sqrt((l.x - t.location.x)**2 + (l.y - t.location.y)**2 + (l.z - t.location.z)**2)
            vehicles = [(distance(x.get_location()), x) for x in vehicles if x.id != world.vehicle.id]
            for d, vehicle in sorted(vehicles, key=lambda vehicles: vehicles[0]):
                if d > 200.0:
                    break
                vehicle_type = get_actor_display_name(vehicle, truncate=22)
                self._info_text.append('% 4dm %s' % (d, vehicle_type))
        self._notifications.render(self.dim) # Render notifications on top of everything

    def toggle_info(self):
        self._show_info = not self._show_info

    def notification(self, text, seconds=2.0):
        self._notifications.set_text(text, seconds=seconds)

    def error(self, text):
        self._notifications.set_text('Error: %s' % text, (255, 0, 0))

    def render(self, display):
        if self._show_info:
            info_surface = pygame.Surface((220, self.dim[1]))
            info_surface.set_alpha(100)
            display.blit(info_surface, (0, 0))
            v_offset = 4
            bar_h_offset = 100
            bar_width = 106
            for item in self._info_text:
                if v_offset + FONT_SIZE > self.dim[1]:
                    break
                if isinstance(item, list):
                    if len(item) > 1:
                        points = [(x + 8, v_offset + 8 + (1.0 - y) * 30) for x, y in enumerate(item)]
                        pygame.draw.lines(display, (255, 136, 0), False, points, 2)
                    item = None
                    v_offset += 18
                elif isinstance(item, tuple):
                    if isinstance(item[1], bool):
                        rect = pygame.Rect((bar_h_offset, v_offset + 8), (6, 6))
                        pygame.draw.rect(display, (255, 255, 255), rect, 0 if item[1] else 1)
                    else:
                        rect_border = pygame.Rect((bar_h_offset, v_offset + 8), (bar_width, 6))
                        pygame.draw.rect(display, (255, 255, 255), rect_border, 1)
                        f = (item[1] - item[2]) / (item[3] - item[2])
                        if item[2] < 0.0:
                            rect = pygame.Rect((bar_h_offset + f * (bar_width - 6), v_offset + 8), (6, 6))
                        else:
                            rect = pygame.Rect((bar_h_offset, v_offset + 8), (f * bar_width, 6))
                        pygame.draw.rect(display, (255, 255, 255), rect)
                    item = item[0]
                if item:  # At this point item is just a str.
                    surface = self._font_mono.render(item, True, (255, 255, 255))
                    display.blit(surface, (8, v_offset))
                v_offset += FONT_SIZE
        self._notifications.render(display)
        self.help.render(display)

class FadingText(object):
    def __init__(self, font, dim, pos):
        self.font = font
        self.dim = dim
        self.pos = pos
        self.seconds_left = 0
        self.surface = pygame.Surface(self.dim)

    def set_text(self, text, color=(255, 255, 255), seconds=2.0):
        text_texture = self.font.render(text, True, color)
        self.surface = pygame.Surface(self.dim)
        self.seconds_left = seconds
        self.surface.fill((0, 0, 0, 0))
        self.surface.blit(text_texture, (10, 11))

    def tick(self, _, clock):
        delta_seconds = 1e-3 * clock.get_time()
        self.seconds_left = max(0.0, self.seconds_left - delta_seconds)
        self.surface.set_alpha(500.0 * self.seconds_left)

    def render(self, display):
        display.blit(self.surface, self.pos)

class HelpText(object):
    def __init__(self, font, width, height):
        lines = __doc__.split('\n')
        self.font = font
        self.dim = (680, len(lines) * 22 + 12)
        self.pos = (0.5 * width - 0.5 * self.dim[0], 0.5 * height - 0.5 * self.dim[1])
        self.seconds_left = 0
        self.surface = pygame.Surface(self.dim)
        self.surface.fill((0, 0, 0, 0))
        for n, line in enumerate(lines):
            text_texture = self.font.render(line, True, (255, 255, 255))
            self.surface.blit(text_texture, (22, n * 22))
            self._render = False
        self.surface.set_alpha(220)

    def toggle(self):
        self._render = not self._render

    def render(self, display):
        if self._render:
            display.blit(self.surface, self.pos)

def get_actor_display_name(actor, truncate=250):
    name = ' '.join(actor.type_id.replace('_', '.').title().split('.')[1:])
    return (name[:truncate - 1] + u'\u2026') if len(name) > truncate else name


class CameraManager(object):
    def __init__(self, parent_actor, hud, gamma_correction):
        self.sensor = None
        self.surface = None
        self._parent = parent_actor
        self.hud = hud
        self.recording = False
        bound_y = 0.5 + self._parent.bounding_box.extent.y
        attachment = carla.AttachmentType.Rigid
        self._camera_transforms = [
            (carla.Transform(carla.Location(x=-5.5, z=2.5), carla.Rotation(pitch=8.0)), attachment), # Chase cam
            (carla.Transform(carla.Location(x=1.6, z=1.7)), attachment), # Default FPV
        ]
        self.transform_index = 0 # Start with chase cam
        self.sensors = [
            ['sensor.camera.rgb', cc.Raw, 'Camera RGB', {}],
        ]
        world = self._parent.get_world()
        bp_library = world.get_blueprint_library()
        for item in self.sensors:
            bp = bp_library.find(item[0])
            if item[0].startswith('sensor.camera'):
                bp.set_attribute('image_size_x', str(hud.dim[0]))
                bp.set_attribute('image_size_y', str(hud.dim[1]))
                if bp.has_attribute('gamma'):
                    bp.set_attribute('gamma', str(gamma_correction))
                for attr_name, attr_value in item[3].items():
                    bp.set_attribute(attr_name, attr_value)
            item.append(bp)
        self.index = None

    def toggle_camera(self):
        self.transform_index = (self.transform_index + 1) % len(self._camera_transforms)
        self.set_sensor(self.index, notify=False, force_respawn=True)

    def set_sensor(self, index, notify=True, force_respawn=False):
        index = index % len(self.sensors)
        needs_respawn = True if self.index is None else \
            (force_respawn or (self.sensors[index][2] != self.sensors[self.index][2]))
        if needs_respawn:
            if self.sensor is not None:
                self.sensor.destroy()
                self.surface = None
            self.sensor = self._parent.get_world().spawn_actor(
                self.sensors[index][-1],
                self._camera_transforms[self.transform_index][0],
                attach_to=self._parent,
                attachment_type=self._camera_transforms[self.transform_index][1])
            # We need to pass the lambda a weak reference to self to avoid
            # circular reference.
            weak_self = weakref.ref(self)
            self.sensor.listen(lambda image: CameraManager._parse_image(weak_self, image))
        if notify:
            self.hud.notification(self.sensors[index][2])
        self.index = index

    def next_sensor(self):
        self.set_sensor(self.index + 1)

    def render(self, display):
        if self.surface is not None:
            display.blit(self.surface, (0, 0))

    @staticmethod
    def _parse_image(weak_self, image):
        self = weak_self()
        if not self:
            return
        if self.sensors[self.index][0].startswith('sensor.lidar'):
            points = np.frombuffer(image.raw_data, dtype=np.dtype('f4'))
            points = np.reshape(points, (int(points.shape[0] / 4), 4))
            lidar_data = np.array(points[:, :2])
            lidar_data *= min(self.hud.dim) / (2.0 * lidar_range)
            lidar_data += (0.5 * self.hud.dim[0], 0.5 * self.hud.dim[1])
            lidar_data = np.fabs(lidar_data)  # pylint: disable=E1111
            lidar_data = lidar_data.astype(np.int32)
            lidar_data = np.reshape(lidar_data, (-1, 2))
            lidar_img_size = (self.hud.dim[0], self.hud.dim[1], 3)
            lidar_img = np.zeros(lidar_img_size, dtype=np.uint8)
            lidar_img[tuple(lidar_data.T)] = (255, 255, 255)
            self.surface = pygame.surfarray.make_surface(lidar_img)
        elif self.sensors[self.index][0].startswith('sensor.camera.dvs'):
            # Example of converting DVS events to image
            dvs_events = np.frombuffer(image.raw_data, dtype=np.dtype([
                ('x', np.uint16), ('y', np.uint16), ('t', np.int64), ('pol', np.bool)]))
            dvs_img = np.zeros((image.height, image.width, 3), dtype=np.uint8)
            # Blue is positive, red is negative
            dvs_img[dvs_events[:]['y'], dvs_events[:]['x'], dvs_events[:]['pol'] * 2] = 255
            self.surface = pygame.surfarray.make_surface(dvs_img.swapaxes(0, 1))
        else:
            image.convert(self.sensors[self.index][1])
            array = np.frombuffer(image.raw_data, dtype=np.uint8)
            array = np.reshape(array, (image.height, image.width, 4))
            array = array[:, :, :3]
            array = array[:, :, ::-1]
            self.surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))

# Need to import weakref for CameraManager
import weakref
import carla.ColorConverter as cc # For CameraManager
import cv2 # For image processing

class World(object):
    def __init__(self, carla_world, hud, actor_filter, spawn_point_index=0):
        self.world = carla_world
        self.hud = hud
        self.player_spawn_point = spawn_point_index
        try:
            self.map = self.world.get_map()
        except RuntimeError as e:
            print('RuntimeError: {}'.format(e))
            print('  The server could not send the map related information, exiting...')
            sys.exit()
        self.vehicle = None
        self.agent_camera_sensor = None # For DQN agent
        self.collision_sensor = None
        self.lane_invasion_sensor = None
        self.camera_manager = None # For Pygame display
        self._actor_filter = actor_filter
        self.restart()
        self.world.on_tick(hud.on_world_tick)
        self.recording_enabled = False
        self.recording_start = 0

    def restart(self):
        # Get a random blueprint.
        blueprint = random.choice(self.world.get_blueprint_library().filter(self._actor_filter))
        blueprint.set_attribute('role_name', 'hero') # Agent vehicle
        if blueprint.has_attribute('color'):
            color = random.choice(blueprint.get_attribute('color').recommended_values)
            blueprint.set_attribute('color', color)

        # Spawn the player.
        if self.vehicle is not None:
            spawn_point = self.vehicle.get_transform()
            spawn_point.location.z += 2.0
            spawn_point.rotation.roll = 0.0
            spawn_point.rotation.pitch = 0.0
            self.destroy() # Destroy previous actors
            self.vehicle = self.world.try_spawn_actor(blueprint, spawn_point)
        
        while self.vehicle is None:
            if not self.map.get_spawn_points():
                print('There are no spawn points available in your map/town.')
                print('Please add some Vehicle Spawn Point to your UE4 scene.')
                sys.exit()
            spawn_points = self.map.get_spawn_points()
            spawn_point = spawn_points[self.player_spawn_point] if self.player_spawn_point < len(spawn_points) else random.choice(spawn_points)
            self.vehicle = self.world.try_spawn_actor(blueprint, spawn_point)

        # Set up the sensors.
        self.collision_sensor = CollisionSensor(self.vehicle, self.hud)
        self.lane_invasion_sensor = LaneInvasionSensor(self.vehicle, self.hud)
        self.camera_manager = CameraManager(self.vehicle, self.hud, 2.2) # Gamma 2.2 for display
        self.camera_manager.transform_index = 0 # Start with chase cam
        self.camera_manager.set_sensor(0, notify=False)

        # Setup agent's observation camera
        agent_cam_bp = self.world.get_blueprint_library().find('sensor.camera.rgb')
        agent_cam_bp.set_attribute('image_size_x', str(IMAGE_WIDTH * 2)) # Higher res for agent if needed, then resize
        agent_cam_bp.set_attribute('image_size_y', str(IMAGE_HEIGHT * 2))
        agent_cam_bp.set_attribute('fov', '90')
        # Position this camera like a front-facing dashboard cam or as per training
        agent_cam_transform = carla.Transform(carla.Location(x=1.5, z=1.7)) 
        self.agent_camera_sensor = self.world.spawn_actor(agent_cam_bp, agent_cam_transform, attach_to=self.vehicle)
        
        self.actor_list = [self.vehicle, self.collision_sensor.sensor, self.lane_invasion_sensor.sensor, self.camera_manager.sensor, self.agent_camera_sensor]


    def tick(self, clock):
        self.hud.tick(self, clock)

    def render(self, display):
        self.camera_manager.render(display)
        self.hud.render(display)

    def destroy_sensors(self):
        self.camera_manager.sensor.destroy()
        self.camera_manager.sensor = None
        self.camera_manager.index = None

    def destroy(self):
        actors_to_destroy = []
        if self.collision_sensor is not None and self.collision_sensor.sensor is not None:
             actors_to_destroy.append(self.collision_sensor.sensor)
        if self.lane_invasion_sensor is not None and self.lane_invasion_sensor.sensor is not None:
             actors_to_destroy.append(self.lane_invasion_sensor.sensor)
        if self.camera_manager is not None and self.camera_manager.sensor is not None:
             actors_to_destroy.append(self.camera_manager.sensor)
        if self.agent_camera_sensor is not None:
            actors_to_destroy.append(self.agent_camera_sensor)
        
        # Filter out None objects before attempting to destroy
        actors_to_destroy = [actor for actor in actors_to_destroy if actor is not None and actor.is_alive]
        if actors_to_destroy:
            self.client.apply_batch([carla.command.DestroyActor(x) for x in actors_to_destroy])

        if self.vehicle is not None and self.vehicle.is_alive:
            self.client.apply_batch([carla.command.DestroyActor(self.vehicle)])
        
        self.vehicle = None
        self.collision_sensor = None
        self.lane_invasion_sensor = None
        self.camera_manager = None
        self.agent_camera_sensor = None
        self.actor_list = []


class CollisionSensor(object):
    def __init__(self, parent_actor, hud):
        self.sensor = None
        self.history = []
        self._parent = parent_actor
        self.hud = hud
        world = self._parent.get_world()
        bp = world.get_blueprint_library().find('sensor.other.collision')
        self.sensor = world.spawn_actor(bp, carla.Transform(), attach_to=self._parent)
        # We need to pass the lambda a weak reference to self to avoid circular reference.
        weak_self = weakref.ref(self)
        self.sensor.listen(lambda event: CollisionSensor._on_collision(weak_self, event))

    def get_collision_history(self):
        history = collections.defaultdict(int)
        for frame, intensity in self.history:
            history[frame] += intensity
        return history

    @staticmethod
    def _on_collision(weak_self, event):
        self = weak_self()
        if not self:
            return
        actor_type = get_actor_display_name(event.other_actor)
        self.hud.notification('Collision with %r' % actor_type)
        impulse = event.normal_impulse
        intensity = np.linalg.norm([impulse.x, impulse.y, impulse.z])
        self.history.append((event.frame, intensity))
        if len(self.history) > 4000:
            self.history.pop(0)

import collections # For CollisionSensor history

class LaneInvasionSensor(object):
    def __init__(self, parent_actor, hud):
        self.sensor = None
        self._parent = parent_actor
        self.hud = hud
        world = self._parent.get_world()
        bp = world.get_blueprint_library().find('sensor.other.lane_invasion')
        self.sensor = world.spawn_actor(bp, carla.Transform(), attach_to=self._parent)
        # We need to pass the lambda a weak reference to self to avoid circular reference.
        weak_self = weakref.ref(self)
        self.sensor.listen(lambda event: LaneInvasionSensor._on_invasion(weak_self, event))

    @staticmethod
    def _on_invasion(weak_self, event):
        self = weak_self()
        if not self:
            return
        lane_types = set(x.type for x in event.crossed_lane_markings)
        text = ['%r' % str(x).split('.')[-1] for x in lane_types]
        self.hud.notification('Crossed line %s' % ' and '.join(text))


def get_action_from_dqn(policy_net, state_np):
    """Gets an action from the DQN agent."""
    # Process state_np to tensor (add batch dim, permute if necessary)
    # (H, W, C) -> (1, C, H, W) for PyTorch CNN
    state_tensor = torch.FloatTensor(state_np).unsqueeze(0).to(DEVICE)
    state_tensor = state_tensor.permute(0, 3, 1, 2) 

    with torch.no_grad():
        q_values = policy_net(state_tensor)
        action = q_values.max(1)[1].item()  # Get action with max Q-value
    return action

def apply_control_from_action(vehicle, action_idx):
    """Applies control to vehicle based on action index from DQN."""
    # This should match the action mapping in your environment.py
    control = carla.VehicleControl()
    if action_idx == 0: # Example: Straight/Accelerate
        control.throttle = 0.7
        control.steer = 0.0
    elif action_idx == 1: # Example: Turn Left
        control.throttle = 0.3
        control.steer = -0.5
    elif action_idx == 2: # Example: Turn Right
        control.throttle = 0.3
        control.steer = 0.5
    # Add more actions if N_ACTIONS > 3
    # elif action_idx == 3: # Example: Brake
    #    control.brake = 1.0
    vehicle.apply_control(control)


def game_loop(args):
    pygame.init()
    pygame.font.init()
    world = None
    agent_observation_queue = queue.Queue() # For agent's camera

    try:
        client = carla.Client(args.host, args.port)
        client.set_timeout(CLIENT_TIMEOUT)

        display = pygame.display.set_mode(
            (args.width, args.height),
            pygame.HWSURFACE | pygame.DOUBLEBUF)
        pygame.display.set_caption("CARLA Agent Visualization")

        hud = HUD(args.width, args.height)
        sim_world = client.load_world('Town01_Opt') # Or make this configurable
        
        # Set synchronous mode
        settings = sim_world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = FIXED_DELTA_SECONDS
        sim_world.apply_settings(settings)

        world = World(sim_world, hud, args.filter)
        
        # Setup agent's camera callback
        def agent_camera_callback(image):
            agent_observation_queue.put(image)
        world.agent_camera_sensor.listen(agent_camera_callback)

        # Load DQN Agent
        # Determine state shape from processed observation
        # Example: processed_state_shape = (IMAGE_HEIGHT, IMAGE_WIDTH, 1)
        # The DQN model expects input channels as the second dimension for Conv2D
        # So, if state_shape is (H, W, C), model input is (C, H, W)
        # The model definition in DQN_control/model.py should reflect this.
        # For now, assume state_shape is (channels, height, width) for model input
        # This needs to be consistent with your DQN model's first conv layer
        # If your _process_image returns (H,W,1), then input_channels = 1
        # The DQN model's __init__ should take (input_channels, height, width) or similar
        # For this example, let's assume the DQN model is flexible or state_shape is (C,H,W)
        # This part needs careful alignment with your model.py
        
        # Let's assume the model expects (C, H, W) and process_agent_observation gives (H, W, C)
        # The get_action_from_dqn function handles the permute.
        # The state_shape for DQN init should be what the *network* expects as input *features*.
        # If your network's first nn.Conv2d has in_channels=1, then state_shape for DQN could be (1, IMAGE_HEIGHT, IMAGE_WIDTH)
        # For simplicity, I'll use the (H,W,C) shape from environment.py and assume model.py handles it or can be adapted.
        agent_state_shape_env = (IMAGE_HEIGHT, IMAGE_WIDTH, 1) # Shape from process_agent_observation
        
        policy_net = DQN(agent_state_shape_env, N_ACTIONS).to(DEVICE) # Use shape from env
        if os.path.exists(MODEL_PATH):
            policy_net.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
            print(f"Loaded trained model from {MODEL_PATH}")
        else:
            print(f"Warning: Model file not found at {MODEL_PATH}. Agent will act randomly (or based on initial weights).")
        policy_net.eval()


        clock = pygame.time.Clock()
        current_agent_observation_np = None

        while True:
            clock.tick_busy_loop(60) # Limit client FPS
            
            # Advance the simulation
            sim_world.tick() # Crucial for synchronous mode

            if world.tick(clock): # HUD tick
                return # Exit if HUD signals exit (e.g. help text)

            # Get agent observation
            try:
                raw_agent_image = agent_observation_queue.get(timeout=0.1) # Non-blocking with timeout
                current_agent_observation_np = process_agent_observation(raw_agent_image, IMAGE_WIDTH, IMAGE_HEIGHT)
            except queue.Empty:
                # Keep previous observation if new one not ready, or handle as error
                if current_agent_observation_np is None: # First frame
                    print("Waiting for initial agent observation...")
                    world.render(display) # Render Pygame display
                    pygame.display.flip()
                    continue # Skip agent action if no observation yet

            # Agent takes action
            if current_agent_observation_np is not None:
                action_idx = get_action_from_dqn(policy_net, current_agent_observation_np)
                apply_control_from_action(world.vehicle, action_idx)

            world.render(display) # Render Pygame display
            pygame.display.flip()

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return
                elif event.type == pygame.KEYUP:
                    if event.key == pygame.K_ESCAPE:
                        return
                    if event.key == pygame.K_TAB:
                        world.camera_manager.toggle_camera()
                    # Add other controls if needed (e.g., toggle HUD)
                    if event.key == pygame.K_h:
                        hud.help.toggle()


    finally:
        if world is not None:
            # Ensure agent camera is destroyed if it was created
            if world.agent_camera_sensor is not None and world.agent_camera_sensor.is_alive:
                 world.agent_camera_sensor.destroy()
            world.destroy()
        
        # Stop CARLA simulation settings
        if 'sim_world' in locals() and sim_world is not None:
            settings = sim_world.get_settings()
            settings.synchronous_mode = False
            settings.fixed_delta_seconds = None
            sim_world.apply_settings(settings)

        pygame.quit()
import queue # For agent_observation_queue

def main():
    import argparse
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument(
        'v', '--verbose',
        action='store_true',
        dest='debug',
        help='print debug information')
    argparser.add_argument(
        '--host',
        metavar='H',
        default=CLIENT_HOST,
        help='IP of the host server (default: 127.0.0.1)')
    argparser.add_argument(
        '-p', '--port',
        metavar='P',
        default=CLIENT_PORT,
        type=int,
        help='TCP port to listen to (default: 2000)')
    argparser.add_argument(
        '--res',
        metavar='WIDTHxHEIGHT',
        default=f'{SCREEN_WIDTH}x{SCREEN_HEIGHT}',
        help=f'window resolution (default: {SCREEN_WIDTH}x{SCREEN_HEIGHT})')
    argparser.add_argument(
        '--filter',
        metavar='PATTERN',
        default='vehicle.tesla.model3', # Or your preferred agent vehicle
        help='actor filter (default: "vehicle.tesla.model3")')
    
    args = argparser.parse_args()
    args.width, args.height = [int(x) for x in args.res.split('x')]

    try:
        game_loop(args)
    except KeyboardInterrupt:
        print('\nCancelled by user. Bye!')
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()
