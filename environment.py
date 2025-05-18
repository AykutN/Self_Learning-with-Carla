import carla
import numpy as np
import random
import time
import cv2 # Görüntü işleme için (veya PIL)
import queue

# Kendi görüntü işleme fonksiyonunuzu import edin
# from DQN_control.process_image import process_image 

# Sabitler
IMAGE_WIDTH = 84  # İşlenmiş görüntü genişliği (örnek)
IMAGE_HEIGHT = 84 # İşlenmiş görüntü yüksekliği (örnek)
N_ACTIONS = 3     # Örnek: [Düz Git/Hızlan, Sola Dön, Sağa Dön]

try:
    client = carla.Client('localhost', 2000)
    client.set_timeout(5.0)
    available_maps = client.get_available_maps()
    print("Available maps:")
    for map_name in available_maps:
        print(f"- {map_name}")
except Exception as e:
    print(f"Could not connect or get maps: {e}")

class CarlaEnv:
    def __init__(self, host='localhost', port=2000, town='Town01_Opt'):
        print(f"Connecting to CARLA on {host}:{port}...")
        self.client = carla.Client(host, port)
        self.client.set_timeout(300.0) # Timeout süresini artır (örneğin 30 saniye)
        print(f"Loading world: {town}...")
        self.world = self.client.load_world(town)
        print("World loaded.")
        self.blueprint_library = self.world.get_blueprint_library()
        
        # Simülasyon Ayarları
        self.settings = self.world.get_settings()
        self.original_settings = self.world.get_settings()
        self.settings.synchronous_mode = True # Senkron modu etkinleştir
        self.settings.fixed_delta_seconds = 0.05 # Önemli: RL adımıyla uyumlu olmalı
        # self.settings.no_rendering_mode = True # Render'ı kapat (opsiyonel)
        self.world.apply_settings(self.settings)
        
        self.tm = self.client.get_trafficmanager(8000)
        self.tm.set_synchronous_mode(True) # Trafik Yöneticisi de senkron modda kalmalı

        self.actor_list = []
        self.vehicle = None
        self.rgb_cam = None
        self.col_sensor = None
        self.lane_sensor = None

        self.image_queue = None # Son görüntüyü saklamak için
        self.collision_data = None
        self.lane_invasion_data = None

        self.sensor_queue = queue.Queue()  # Sensör verisi için thread-safe queue

        # RL Arayüzü için
        self.action_space_size = N_ACTIONS
        # Gözlem uzayı: İşlenmiş, gri tonlamalı görüntü (örnek)
        self.observation_space_shape = (IMAGE_HEIGHT, IMAGE_WIDTH, 1) 

        print("CARLA Environment Initialized.")

    def _process_image(self, carla_image):
        """CARLA kamera verisini işler."""
        # Ham veriyi numpy dizisine çevir
        img = np.frombuffer(carla_image.raw_data, dtype=np.dtype("uint8"))
        img = np.reshape(img, (carla_image.height, carla_image.width, 4))
        img = img[:, :, :3] # Alfa kanalını at

        # ---- BURAYA KENDİ İŞLEME KODUNUZU EKLEYİN ----
        # If img could be None from carla_image processing (though unlikely with frombuffer),
        # you might add a check:
        if img is None:
            print("Error: Processed carla_image is None before further processing.")
            # Handle this case, perhaps by returning a default black image or raising an error
            return np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH, 1), dtype=np.float32) 

        # The following lines depend on 'img' being a valid image array
        scale_percent = 25 # This might be too small if IMAGE_WIDTH/HEIGHT are already small like 84
        
        # Örnek: Gri tonlama ve yeniden boyutlandırma
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_resized = cv2.resize(img_gray, (IMAGE_WIDTH, IMAGE_HEIGHT))
        # Kanal boyutu ekle (PyTorch Conv2D için: Channels First)
        # img_processed = np.expand_dims(img_resized, axis=0) # (1, H, W)
        # Veya (TensorFlow/Keras Conv2D için: Channels Last)
        img_processed = np.expand_dims(img_resized, axis=-1) # (H, W, 1) 
        # Normalizasyon (isteğe bağlı, 0-1 arasına getirme)
        img_processed = img_processed / 255.0 

        return img_processed.astype(np.float32) # DQN için float32

    def _camera_callback(self, image):
        """Kamera verilerini işler ve kuyruğa ekler."""
        try:
            if self.sensor_queue:
                self.sensor_queue.put(image, block=False)
        except queue.Full:
            pass
        except Exception as e:
            pass

    def _collision_callback(self, event):
        """Çarpışma sensörü verisini alır."""
        self.collision_data = event

    def _lane_invasion_callback(self, event):
        """Şerit ihlali sensörü verisini alır."""
        self.lane_invasion_data = event

    def _spawn_traffic(self, num_vehicles=20, num_walkers=10):
        """Simülasyona trafik (araçlar ve yayalar) ekler."""
        print("Spawning traffic...")
        blueprints = self.world.get_blueprint_library().filter('vehicle.*')
        spawn_points = self.world.get_map().get_spawn_points()
        random.shuffle(spawn_points)

        # Araçları spawn et
        for i in range(min(num_vehicles, len(spawn_points))):
            vehicle_bp = random.choice(blueprints)
            transform = spawn_points[i]
            vehicle = self.world.try_spawn_actor(vehicle_bp, transform)
            if vehicle:
                self.actor_list.append(vehicle)
                print(f"Spawned vehicle {vehicle.id} at {transform.location}")

        # Yayaları spawn et (opsiyonel, daha fazla detay eklenebilir)
        walker_blueprints = self.world.get_blueprint_library().filter('walker.pedestrian.*')
        for i in range(num_walkers):
            walker_bp = random.choice(walker_blueprints)
            spawn_point = carla.Transform()
            spawn_point.location = self.world.get_random_location_from_navigation()
            if spawn_point.location:
                walker = self.world.try_spawn_actor(walker_bp, spawn_point)
                if walker:
                    self.actor_list.append(walker)
                    print(f"Spawned walker {walker.id} at {spawn_point.location}")

    def reset(self):
        """Ortamı sıfırlar ve ilk gözlemi döndürür."""
        print("Resetting environment...")
        self._destroy_actors()
        print("Actors destroyed.")
        self.image_queue = None
        self.collision_data = None
        self.lane_invasion_data = None

        self.sensor_queue.queue.clear()  # Queue'u temizle

        # Trafiği başlat (opsiyonel, eğitim hızını etkileyebilir)
        # self._spawn_traffic(num_vehicles=5, num_walkers=0) # Daha az trafik

        # Aracı spawn et
        vehicle_bp = self.blueprint_library.filter('model3')[0] # Belirli bir araç modeli
        transform = random.choice(self.world.get_map().get_spawn_points())
        self.vehicle = self.world.try_spawn_actor(vehicle_bp, transform)
        while self.vehicle is None:
            print("Warning: Failed to spawn vehicle, retrying...")
            time.sleep(0.5)
            transform = random.choice(self.world.get_map().get_spawn_points())
            self.vehicle = self.world.try_spawn_actor(vehicle_bp, transform)
        self.actor_list.append(self.vehicle)
        print(f"Vehicle spawned at: {transform.location}")

        # Kamera sensörünü ekle (arka kamera görünümü için ayarlandı)
        print("Spawning rear camera sensor...")
        cam_bp = self.blueprint_library.find('sensor.camera.rgb')
        cam_bp.set_attribute('image_size_x', '84') # DQN ile uyumlu boyut
        cam_bp.set_attribute('image_size_y', '84') # DQN ile uyumlu boyut
        cam_bp.set_attribute('fov', '90')
        # cam_bp.set_attribute('sensor_tick', '0.1') # Sensör tick hızını ayarla (opsiyonel)
        cam_transform = carla.Transform(carla.Location(x=1.5, z=2.4)) # Kamera konumu
        self.rgb_cam = self.world.spawn_actor(cam_bp, cam_transform, attach_to=self.vehicle)
        self.rgb_cam.listen(self._camera_callback)
        print("Rear camera sensor spawned and listening.")
        self.actor_list.append(self.rgb_cam)

        # Çarpışma sensörünü ekle
        print("Spawning collision sensor...")
        col_bp = self.blueprint_library.find('sensor.other.collision')
        col_transform = carla.Transform() # Araç merkezine göre
        self.col_sensor = self.world.spawn_actor(col_bp, col_transform, attach_to=self.vehicle)
        self.col_sensor.listen(self._collision_callback)
        self.actor_list.append(self.col_sensor)

        # Şerit ihlali sensörünü ekle
        print("Spawning lane invasion sensor...")
        lane_bp = self.blueprint_library.find('sensor.other.lane_invasion')
        lane_transform = carla.Transform() # Araç merkezine göre
        self.lane_sensor = self.world.spawn_actor(lane_bp, lane_transform, attach_to=self.vehicle)
        self.lane_sensor.listen(self._lane_invasion_callback)
        self.actor_list.append(self.lane_sensor)

        # İlk görüntünün gelmesini bekle
        print("Waiting for first sensor data...")
        if self.world.get_settings().synchronous_mode:
            self.world.tick()  # İlk tick
            print("Initial tick done.")
        
        start_time = time.time()
        tick_count = 0
        while True:
            try:
                print(f"Trying to get image, elapsed: {time.time() - start_time:.2f}s, ticks: {tick_count}")
                image = self.sensor_queue.get_nowait()  # Veri varsa hemen al
                initial_state = self._process_image(image)
                print("First image received.")
                break
            except queue.Empty:
                if time.time() - start_time > 10:  # 10 saniye timeout (artırıldı)
                    print("Timeout waiting for sensor data!")
                    raise RuntimeError("Failed to get initial sensor data from CARLA.")
                if self.world.get_settings().synchronous_mode:
                    tick_count += 1
                    print(f"No image yet. Ticking world, count: {tick_count}")
                    self.world.tick()  # Bir tick daha at
                time.sleep(0.05)  # Kısa bir bekleme, CPU'yu yormamak için

        # Başlangıçta aracın kaymasını engellemek için hafif fren
        self.vehicle.apply_control(carla.VehicleControl(brake=0.1))
        self.world.tick() # Kontrolü uygula
        print("Reset completed successfully.")

        return initial_state

    def step(self, action_index):
        """Bir eylem uygular ve sonucu döndürür."""
        self.sensor_queue.queue.clear()
        self.collision_data = None 
        self.lane_invasion_data = None 

        control = self._get_control_from_action(action_index)
        self.vehicle.apply_control(control)

        if self.world.get_settings().synchronous_mode:
            self.world.tick()

        try:
            image = self.sensor_queue.get(timeout=1.0) # Timeout süresi kısaltıldı
            new_state = self._process_image(image)
        except queue.Empty:
            print("[Env Step] Warning: Timed out waiting for image in step. Applying brake.")
            self.vehicle.apply_control(carla.VehicleControl(brake=1.0))
            new_state = np.zeros(self.observation_space_shape, dtype=np.float32)
            reward = -10 # Hata için ceza
            done = True # Bölümü bitir
            return new_state, reward, done, {"error": "Image timeout"}

        reward, done = self._calculate_reward_and_done()

        return new_state, reward, done, {} # {} ek bilgi için

    def _get_control_from_action(self, action_index):
        """Eylem indeksini carla.VehicleControl'e çevirir."""
        steer = 0.0
        throttle = 0.5 # Varsayılan olarak hafif gaz verelim
        brake = 0.0

        if action_index == 0: # Düz Git / Hızlan
            steer = 0.0
            throttle = 0.7
        elif action_index == 1: # Sola Dön
            steer = -0.5 
            throttle = 0.3 # Dönerken yavaşla
        elif action_index == 2: # Sağa Dön
            steer = 0.5
            throttle = 0.3 # Dönerken yavaşla
        # Daha fazla eylem eklenebilir (örn. fren)

        return carla.VehicleControl(throttle=throttle, steer=steer, brake=brake)

    def _calculate_reward_and_done(self):
        """Ödülü hesaplar ve bölümün bitip bitmediğini kontrol eder."""
        reward = 0.0
        done = False

        # Hedef hız (örnek: 30 km/s)
        target_speed_kmh = 30
        v = self.vehicle.get_velocity()
        speed_kmh = 3.6 * np.sqrt(v.x**2 + v.y**2 + v.z**2)
        
        # Hız ödülü/cezası
        speed_diff = abs(speed_kmh - target_speed_kmh)
        reward += max(0, 1.0 - speed_diff / target_speed_kmh) * 0.1 # Hıza yaklaştıkça küçük ödül

        # Şerit ihlali cezası
        if self.lane_invasion_data is not None:
            reward -= 2.0 # Şeritten çıkma cezası
            # done = True # İsterseniz şeritten çıkınca bölüm bitebilir

        # Çarpışma cezası ve bölüm sonu
        if self.collision_data is not None:
            print(f"Collision detected with: {self.collision_data.other_actor.type_id}")
            reward -= 100.0 # Büyük çarpışma cezası
            done = True # Çarpışınca bölüm biter

        # Çok yavaş gitme cezası (opsiyonel)
        # if speed_kmh < 5 and throttle > 0.1: # Gaz verirken çok yavaşsa
        #    reward -= 0.5

        # Hayatta kalma ödülü (opsiyonel)
        if not done:
             reward += 0.01 

        return reward, done

    def _destroy_actors(self):
        """Tüm oluşturulan aktörleri temizler."""
        print(f"Attempting to destroy actors...")
        # self.actor_list'in var olup olmadığını ve bir liste olduğunu kontrol et
        if hasattr(self, 'actor_list') and isinstance(self.actor_list, list) and self.actor_list:
            print(f"Destroying {len(self.actor_list)} actors...")
            # Önce sensörlerin dinlemesini durdur
            for actor in self.actor_list:
                 # Aktörün hala geçerli olup olmadığını ve dinleyip dinlemediğini kontrol et
                 if actor is not None and actor.is_alive and hasattr(actor, 'is_listening') and actor.is_listening:
                     try:
                         actor.stop()
                     except RuntimeError as e:
                         # Aktör zaten yok edilmişse veya başka bir sorun varsa hata verebilir
                         print(f"Warning: Could not stop actor {actor.id}: {e}")
            
            # Sonra tüm aktörleri yok et (client var mı kontrol et)
            if hasattr(self, 'client') and self.client:
                try:
                    # Yok edilecek komutları oluştururken None veya ölü aktörleri filtrele
                    destroy_commands = [carla.command.DestroyActor(actor) for actor in self.actor_list if actor is not None and actor.is_alive]
                    if destroy_commands:
                        self.client.apply_batch_sync(destroy_commands, True) # Hata olursa exception fırlat
                    else:
                        print("No living actors found in the list to destroy.")
                except RuntimeError as e:
                    # Toplu yok etme sırasında hata oluşabilir (örn. sunucu bağlantısı koptuysa)
                    print(f"Error during actor destruction batch: {e}")
            else:
                print("Warning: CARLA client not available for destroying actors.")

            self.actor_list = [] # Listeyi temizle
        else:
            # Eğer actor_list yoksa veya boşsa bilgi ver
            print("No actor_list found or it's empty, skipping destruction.")

        # Aktör referanslarını her durumda None yap
        self.vehicle = None
        self.rgb_cam = None
        self.col_sensor = None
        self.lane_sensor = None

    def __del__(self):
        """Ortam silindiğinde temizlik yapar."""
        print("Cleaning up CARLA environment...")
        if hasattr(self, 'original_settings'):
             self.world.apply_settings(self.original_settings)
        self._destroy_actors()
        print("Cleanup complete.")

# --- Test Kodu ---
if __name__ == '__main__':
    env = None
    try:
        # CARLA sunucusunun çalıştığından emin olun!
        env = CarlaEnv()
        for episode in range(2): # 2 bölüm test et
            print(f"\n--- Episode {episode + 1} ---")
            obs = env.reset()
            print(f"Initial observation shape: {obs.shape}, dtype: {obs.dtype}")
            done = False
            step = 0
            total_reward = 0
            while not done and step < 100: # En fazla 100 adım
                action = random.randint(0, env.action_space_size - 1) # Rastgele eylem seç
                new_obs, reward, done, info = env.step(action)
                total_reward += reward
                print(f"Step: {step+1}, Action: {action}, Reward: {reward:.2f}, Done: {done}, Obs Shape: {new_obs.shape}")
                obs = new_obs
                step += 1
                # time.sleep(0.1) # İsterseniz adımlar arası bekleyebilirsiniz
            print(f"Episode finished after {step} steps. Total reward: {total_reward:.2f}")

    except Exception as e:
        print(f"\nAn error occurred: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if env is not None:
            del env # Temizlik fonksiyonunun çağrıldığından emin olun

