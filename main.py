import carla
import pygame
import numpy as np
import random
import time
import os
import torch
import torch.optim as optim
from collections import deque # İstatistik takibi için
import weakref # CameraManager ve sensörler için
import collections # CollisionSensor için
import cv2 # Görüntü işleme için (HUD veya ek sensörler için gerekebilir)
import gc # <<< YENİ: gc import'u dosyanın başına taşındı

# Kendi modüllerimizi import edelim
from environment import CarlaEnv, IMAGE_WIDTH as AGENT_IMAGE_WIDTH, IMAGE_HEIGHT as AGENT_IMAGE_HEIGHT, N_ACTIONS 
from DQN_control.model import DQN  
from DQN_control.replay_buffer import ReplayBuffer 

# --- Pygame Ayarları ---
SCREEN_WIDTH = 1280
SCREEN_HEIGHT = 720
FONT_SIZE = 16
VISUALIZE_TRAINING = False # <<< YENİ: Görselleştirmeyi aç/kapa

# --- Hiperparametreler (Eğitim için) ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# Eğitim Parametreleri
NUM_EPISODES = 1000
MAX_STEPS_PER_EPISODE = 500 
LEARNING_RATE = 0.0001
GAMMA = 0.99
BATCH_SIZE = 64

# Tekrar Hafızası
REPLAY_BUFFER_SIZE = 20000
MIN_REPLAY_SIZE_TO_TRAIN = 1000

# Epsilon Greedy Stratejisi
EPSILON_START = 1.0
EPSILON_END = 0.01
EPSILON_DECAY_RATE = 0.995

# Ağ Güncelleme
TARGET_UPDATE_FREQ = 10

# Kayıt Ayarları
MODEL_SAVE_PATH = "dqn_carla_model.pth" 
SAVE_FREQ = 25

# CARLA Yeniden Başlatma ve Zaman Aşımı
CARLA_RESTART_INTERVAL = 25
MAX_EPISODE_DURATION_SECONDS = 300

#region Pygame Yardımcı Sınıfları
def get_actor_display_name(actor, truncate=250):
    name = ' '.join(actor.type_id.replace('_', '.').title().split('.')[1:])
    return (name[:truncate - 1] + u'\u2026') if len(name) > truncate else name

class HUD(object):
    def __init__(self, width, height):
        self.dim = (width, height)
        pygame.font.init()
        font_name = 'courier' if os.name == 'nt' else 'mono'
        fonts = [x for x in pygame.font.get_fonts() if font_name in x]
        default_font = 'ubuntumono'
        mono = default_font if default_font in fonts else (fonts[0] if fonts else pygame.font.get_default_font())
        mono = pygame.font.match_font(mono)
        self._font_mono = pygame.font.Font(mono, FONT_SIZE if mono else 12)
        self._notifications = FadingText(self._font_mono, (width, 40), (0, height - 40))
        self.help = HelpText(pygame.font.Font(mono, 16 if mono else 12), width, height)
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

    def tick(self, world_vehicle, clock, episode_data):
        self._notifications.tick(world_vehicle, clock)
        if not self._show_info:
            return
        
        if world_vehicle is None or not world_vehicle.is_alive:
            self._info_text = ["Vehicle not available"] 
            return

        t = world_vehicle.get_transform()
        v = world_vehicle.get_velocity()
        c = world_vehicle.get_control()
        
        vehicles = world_vehicle.get_world().get_actors().filter('vehicle.*')
        
        self._info_text = [
            'CARLA Server: %5.0f FPS' % self.server_fps,
            'Pygame Client: %4.0f FPS' % clock.get_fps(),
            '',
            'Episode: %s' % episode_data.get('episode', 'N/A'),
            'Step: %s' % episode_data.get('step', 'N/A'),
            'Total Steps: %s' % episode_data.get('total_steps', 'N/A'),
            'Reward: %.2f' % episode_data.get('reward', 0),
            'Avg Reward (100): %.2f' % episode_data.get('avg_reward', 0),
            'Epsilon: %.4f' % episode_data.get('epsilon', 0),
            'Loss: %s' % episode_data.get('loss', 'N/A'),
            '',
            'Vehicle: %s' % get_actor_display_name(world_vehicle, truncate=20),
            'Map: %s' % world_vehicle.get_world().get_map().name.split('/')[-1],
            'Sim Time: % 10.2f s' % self.simulation_time,
            'Speed: %10.0f km/h' % (3.6 * np.linalg.norm([v.x, v.y, v.z])),
            u'Heading: %8.0f\N{DEGREE SIGN}' % (t.rotation.yaw),
            'Location: (%5.1f, %5.1f)' % (t.location.x, t.location.y),
            'Height: %8.1f m' % t.location.z,
            '',
            ('Throttle:', c.throttle, 0.0, 1.0),
            ('Steer:', c.steer, -1.0, 1.0),
            ('Brake:', c.brake, 0.0, 1.0),
            '',
            'Vehicles in Sim: %d' % len(vehicles)
        ]

    def toggle_info(self):
        self._show_info = not self._show_info

    def notification(self, text, seconds=2.0):
        self._notifications.set_text(text, seconds=seconds)

    def error(self, text):
        self._notifications.set_text('Error: %s' % text, (255, 0, 0))

    def render(self, display):
        if self._show_info:
            info_surface = pygame.Surface((240, self.dim[1]))
            info_surface.set_alpha(100)
            display.blit(info_surface, (0, 0))
            v_offset = 4
            bar_h_offset = 100
            bar_width = 120
            for item in self._info_text:
                if v_offset + FONT_SIZE > self.dim[1]:
                    break
                if isinstance(item, tuple):
                    if isinstance(item[1], bool):
                        rect = pygame.Rect((bar_h_offset, v_offset + 8), (6, 6))
                        pygame.draw.rect(display, (255, 255, 255), rect, 0 if item[1] else 1)
                    else:
                        rect_border = pygame.Rect((bar_h_offset, v_offset + 8), (bar_width, 6))
                        pygame.draw.rect(display, (255, 255, 255), rect_border, 1)
                        f = (item[1] - item[2]) / (item[3] - item[2])
                        if item[2] < 0.0:
                            rect_val = f * (bar_width - 6)
                            rect = pygame.Rect((bar_h_offset + (bar_width - 6)/2 + rect_val/2 -3 , v_offset + 8), (6,6))
                        else:
                            rect = pygame.Rect((bar_h_offset, v_offset + 8), (f * bar_width, 6))
                        pygame.draw.rect(display, (255, 255, 255), rect)
                    item = item[0]
                if item: 
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
        self.surface = pygame.Surface(self.dim, pygame.SRCALPHA)

    def set_text(self, text, color=(255, 255, 255), seconds=2.0):
        text_texture = self.font.render(text, True, color)
        self.surface = pygame.Surface(self.dim, pygame.SRCALPHA)
        self.seconds_left = seconds
        self.surface.blit(text_texture, (10, 11))

    def tick(self, _, clock):
        delta_seconds = 1e-3 * clock.get_time()
        self.seconds_left = max(0.0, self.seconds_left - delta_seconds)
        self.surface.set_alpha(int(min(1,self.seconds_left / 2.0) * 255))

    def render(self, display):
        display.blit(self.surface, self.pos)

class HelpText(object):
    """Pygame Help Text. Keys: H (toggle), ESC (quit), TAB (camera), I (info panel)"""
    def __init__(self, font, width, height):
        lines = self.__doc__.split('\n')
        self.font = font
        self.dim = (width - 40, len(lines) * 22 + 12)
        self.pos = (20, 0.5 * height - 0.5 * self.dim[1])
        self.surface = pygame.Surface(self.dim, pygame.SRCALPHA)
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

class CameraManager(object):
    def __init__(self, parent_actor, hud_surface_size):
        self.sensor = None
        self.surface = None
        self._parent = parent_actor
        self.hud_surface_size = hud_surface_size
        attachment = carla.AttachmentType.SpringArm
        self._camera_transforms = [
            (carla.Transform(carla.Location(x=-7.5, z=3.0), carla.Rotation(pitch=-20.0)), attachment),
            (carla.Transform(carla.Location(x=1.6, z=1.7)), carla.AttachmentType.Rigid),
        ]
        self.transform_index = 0
        world = self._parent.get_world()
        bp_library = world.get_blueprint_library()
        self.bp = bp_library.find('sensor.camera.rgb')
        self.bp.set_attribute('image_size_x', str(self.hud_surface_size[0]))
        self.bp.set_attribute('image_size_y', str(self.hud_surface_size[1]))
        if self.bp.has_attribute('gamma'):
            self.bp.set_attribute('gamma', str(2.2))
        self.sensor_active = False
        self._setup_sensor()

    def _setup_sensor(self):
        if self.sensor is not None and self.sensor.is_alive:
            self.sensor.stop()
            self.sensor.destroy()
        self.sensor = self._parent.get_world().spawn_actor(
            self.bp,
            self._camera_transforms[self.transform_index][0],
            attach_to=self._parent,
            attachment_type=self._camera_transforms[self.transform_index][1])
        weak_self = weakref.ref(self)
        self.sensor.listen(lambda image: CameraManager._parse_image(weak_self, image))
        self.sensor_active = True

    def toggle_camera(self):
        self.transform_index = (self.transform_index + 1) % len(self._camera_transforms)
        if self._parent and self._parent.is_alive:
             self._setup_sensor()
        else:
            print("CameraManager: Parent actor not valid, cannot toggle camera.")
            self.sensor_active = False

    def render(self, display):
        if self.surface is not None:
            display.blit(self.surface, (0, 0))

    @staticmethod
    def _parse_image(weak_self, image):
        self = weak_self()
        if not self or not self.sensor_active or not self.sensor.is_alive:
            return
        try:
            image.convert(carla.ColorConverter.Raw)
            array = np.frombuffer(image.raw_data, dtype=np.uint8)
            array = np.reshape(array, (image.height, image.width, 4))
            array = array[:, :, :3]
            array = array[:, :, ::-1]
            self.surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))
        except Exception as e:
            print(f"Error parsing camera image: {e}")
            self.surface = None

    def destroy(self):
        self.sensor_active = False
        if self.sensor is not None and self.sensor.is_alive:
            self.sensor.stop()
            self.sensor.destroy()
        self.sensor = None
        self.surface = None
#endregion

# --- Ana Eğitim Fonksiyonu ---
def main():
    if VISUALIZE_TRAINING:
        pygame.init()
        pygame.font.init()
    display = None
    env = None
    policy_net = None
    target_net = None
    camera_manager = None
    hud = None
    
    episode_rewards_list = []

    try:
        if VISUALIZE_TRAINING:
            display = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.HWSURFACE | pygame.DOUBLEBUF)
            pygame.display.set_caption("CARLA DQN Training Visualization")
        clock = pygame.time.Clock()

        env = CarlaEnv(town='Town01_Opt') 
        agent_state_shape = (AGENT_IMAGE_HEIGHT, AGENT_IMAGE_WIDTH, 1)
        action_size = env.action_space_size

        policy_net = DQN(agent_state_shape, action_size).to(DEVICE)
        target_net = DQN(agent_state_shape, action_size).to(DEVICE)
        
        target_net.load_state_dict(policy_net.state_dict())
        target_net.eval()

        optimizer = optim.Adam(policy_net.parameters(), lr=LEARNING_RATE)
        replay_buffer = ReplayBuffer(REPLAY_BUFFER_SIZE, BATCH_SIZE, DEVICE, state_shape=agent_state_shape)

        if VISUALIZE_TRAINING:
            hud = HUD(SCREEN_WIDTH, SCREEN_HEIGHT)
            if env.world: env.world.on_tick(hud.on_world_tick)
        else:
            hud = None

        episode_rewards_deque = deque(maxlen=100)
        epsilon = EPSILON_START
        total_steps_counter = 0
        current_loss_value = "N/A"

        print("Starting training with Pygame visualization...")
        for episode_num in range(1, NUM_EPISODES + 1):
            episode_start_sim_time = env.world.get_snapshot().timestamp.elapsed_seconds if env.world else time.time()
            
            if episode_num > 1 and episode_num % CARLA_RESTART_INTERVAL == 0:
                print(f"\n--- Restarting CARLA environment (Episode {episode_num}) ---")
                if camera_manager: camera_manager.destroy(); camera_manager = None
                if env: del env; time.sleep(3); env = None; gc.collect()
                try:
                    env = CarlaEnv(town='Town01_Opt')
                    if env.world: env.world.on_tick(hud.on_world_tick)
                    print("CARLA environment restarted successfully.")
                except Exception as e_restart:
                    print(f"Critical error restarting CARLA: {e_restart}. Exiting.")
                    raise e_restart
            
            try:
                state = env.reset() 
                episode_reward_sum = 0

                if VISUALIZE_TRAINING and env.vehicle and (camera_manager is None or not camera_manager.sensor_active):
                    if camera_manager: camera_manager.destroy()
                    camera_manager = CameraManager(env.vehicle, (SCREEN_WIDTH, SCREEN_HEIGHT))
                elif VISUALIZE_TRAINING and not env.vehicle:
                    print("Error: env.vehicle is None after reset. Skipping episode visual setup.")
                elif not VISUALIZE_TRAINING:
                    camera_manager = None

                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(DEVICE)
                if state_tensor.ndim == 2:
                    state_tensor = state_tensor.unsqueeze(0)
                elif state_tensor.ndim == 3 and state_tensor.shape[0] == AGENT_IMAGE_HEIGHT:
                    state_tensor = state_tensor.permute(2, 0, 1)
                if state_tensor.ndim != 3 or state_tensor.shape[1] != AGENT_IMAGE_HEIGHT or state_tensor.shape[2] != AGENT_IMAGE_WIDTH:
                    if state_tensor.numel() == AGENT_IMAGE_HEIGHT * AGENT_IMAGE_WIDTH:
                        state_tensor = state_tensor.reshape(1, AGENT_IMAGE_HEIGHT, AGENT_IMAGE_WIDTH)
                state_tensor = state_tensor.unsqueeze(0)

                for step_in_episode in range(1, MAX_STEPS_PER_EPISODE + 1):
                    total_steps_counter += 1
                    
                    if VISUALIZE_TRAINING:
                        for event in pygame.event.get():
                            if event.type == pygame.QUIT: raise KeyboardInterrupt
                            if event.type == pygame.KEYUP:
                                if event.key == pygame.K_ESCAPE: raise KeyboardInterrupt
                                if event.key == pygame.K_TAB and camera_manager: camera_manager.toggle_camera()
                                if event.key == pygame.K_h and hud: hud.help.toggle()
                                if event.key == pygame.K_i and hud: hud.toggle_info()
                    else:
                        pass

                    current_sim_time = env.world.get_snapshot().timestamp.elapsed_seconds if env.world else time.time()
                    if (current_sim_time - episode_start_sim_time) > MAX_EPISODE_DURATION_SECONDS:
                        print(f"Episode {episode_num} timed out.")
                        break
                    
                    if random.random() > epsilon:
                        with torch.no_grad():
                            action = policy_net(state_tensor).max(1)[1].item()
                    else:
                        action = random.randrange(action_size)
                    
                    next_state, reward, done, info = env.step(action)
                    episode_reward_sum += reward
                    replay_buffer.add(state, action, reward, next_state, done)
                    state = next_state
                    state_tensor = torch.FloatTensor(state).unsqueeze(0).to(DEVICE)
                    if state_tensor.ndim == 2:
                        state_tensor = state_tensor.unsqueeze(0)
                    elif state_tensor.ndim == 3 and state_tensor.shape[0] == AGENT_IMAGE_HEIGHT:
                        state_tensor = state_tensor.permute(2, 0, 1)
                    if state_tensor.ndim != 3 or state_tensor.shape[1] != AGENT_IMAGE_HEIGHT or state_tensor.shape[2] != AGENT_IMAGE_WIDTH:
                        if state_tensor.numel() == AGENT_IMAGE_HEIGHT * AGENT_IMAGE_WIDTH:
                            state_tensor = state_tensor.reshape(1, AGENT_IMAGE_HEIGHT, AGENT_IMAGE_WIDTH)
                    state_tensor = state_tensor.unsqueeze(0)

                    if len(replay_buffer) > MIN_REPLAY_SIZE_TO_TRAIN:
                        s_batch, a_batch, r_batch, ns_batch, d_batch = replay_buffer.sample()
                        if isinstance(s_batch, np.ndarray):
                            s_batch = torch.from_numpy(s_batch).float()
                            ns_batch = torch.from_numpy(ns_batch).float()
                            a_batch = torch.from_numpy(a_batch).long()
                            r_batch = torch.from_numpy(r_batch).float()
                            d_batch = torch.from_numpy(d_batch).float()
                        if s_batch.ndim == 2:
                            s_batch = s_batch.reshape(BATCH_SIZE, AGENT_IMAGE_HEIGHT, AGENT_IMAGE_WIDTH, 1)
                        elif s_batch.ndim == 3:
                            s_batch = s_batch.unsqueeze(3)
                        s_batch_tensor = s_batch.permute(0, 3, 1, 2).to(DEVICE)
                        if ns_batch.ndim == 2:
                            ns_batch = ns_batch.reshape(BATCH_SIZE, AGENT_IMAGE_HEIGHT, AGENT_IMAGE_WIDTH, 1)
                        elif ns_batch.ndim == 3:
                            ns_batch = ns_batch.unsqueeze(3)
                        ns_batch_tensor = ns_batch.permute(0, 3, 1, 2).to(DEVICE)
                        a_batch_tensor = a_batch.to(DEVICE).unsqueeze(1)
                        r_batch_tensor = r_batch.to(DEVICE).unsqueeze(1)
                        d_batch_tensor = d_batch.to(DEVICE).unsqueeze(1)
                        current_q = policy_net(s_batch_tensor).gather(1, a_batch_tensor)
                        next_q_target = target_net(ns_batch_tensor).max(1)[0].detach()
                        target_q = r_batch_tensor + (GAMMA * next_q_target * (1 - d_batch_tensor))
                        loss = torch.nn.functional.mse_loss(current_q, target_q.unsqueeze(1))
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                        current_loss_value = loss.item()
                    
                    if VISUALIZE_TRAINING and display and camera_manager and hud and env.vehicle and env.vehicle.is_alive:
                        camera_manager.render(display)
                        episode_data_for_hud = {
                            'episode': f"{episode_num}/{NUM_EPISODES}",
                            'step': f"{step_in_episode}/{MAX_STEPS_PER_EPISODE}",
                            'total_steps': total_steps_counter,
                            'reward': episode_reward_sum,
                            'avg_reward': np.mean(episode_rewards_deque) if episode_rewards_deque else 0.0,
                            'epsilon': epsilon,
                            'loss': f"{current_loss_value:.4f}" if isinstance(current_loss_value, float) else current_loss_value
                        }
                        hud.tick(env.vehicle, clock, episode_data_for_hud)
                        hud.render(display)
                        pygame.display.flip()
                    
                    clock.tick(0)

                    if done:
                        break
                
                episode_rewards_deque.append(episode_reward_sum)
                episode_rewards_list.append(episode_reward_sum)
                epsilon = max(EPSILON_END, epsilon * EPSILON_DECAY_RATE)
                
                if episode_num % TARGET_UPDATE_FREQ == 0:
                    print(f"Updating target network (Episode {episode_num}).")
                    target_net.load_state_dict(policy_net.state_dict())
                
                if episode_num % SAVE_FREQ == 0:
                    iter_model_path = MODEL_SAVE_PATH.replace(".pth", f"_ep{episode_num}.pth")
                    print(f"Saving model to {iter_model_path} (Episode {episode_num}).")
                    torch.save(policy_net.state_dict(), iter_model_path)
                
                avg_r = np.mean(episode_rewards_deque) if episode_rewards_deque else 0.0
                print(f"Ep: {episode_num} | Steps: {step_in_episode} | Reward: {episode_reward_sum:.2f} | AvgR: {avg_r:.2f} | Eps: {epsilon:.4f} | Loss: {current_loss_value if isinstance(current_loss_value, float) else 'N/A'}")
            
            except Exception as e_episode:
                print(f"Error in episode {episode_num}: {e_episode}")
                import traceback; traceback.print_exc()
                if env and env.world:
                    settings = env.world.get_settings()
                    settings.synchronous_mode = False; env.world.apply_settings(settings)
                time.sleep(1)
                continue 

        print("Training finished.")
        final_model_path = MODEL_SAVE_PATH.replace(".pth", "_final.pth")
        print(f"Saving final model to {final_model_path}...")
        torch.save(policy_net.state_dict(), final_model_path)

    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")
    except Exception as e_main:
        print(f"\nCritical error in main: {e_main}")
        import traceback; traceback.print_exc()
    finally:
        print("Cleaning up...")
        if VISUALIZE_TRAINING and camera_manager: camera_manager.destroy()
        if env is not None: 
            if env.world and env.original_settings:
                print("Restoring original CARLA world settings...")
                current_settings = env.world.get_settings()
                current_settings.synchronous_mode = env.original_settings.synchronous_mode
                current_settings.fixed_delta_seconds = env.original_settings.fixed_delta_seconds
                env.world.apply_settings(current_settings)
            del env
        if VISUALIZE_TRAINING and 'pygame' in locals() and pygame.get_init(): 
            pygame.quit()
        if 'gc' in globals() or 'gc' in locals(): 
             gc.collect()
        print("Cleanup complete. Exiting.")

if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(f"Unhandled exception in __main__: {e}")
        import traceback
        traceback.print_exc()