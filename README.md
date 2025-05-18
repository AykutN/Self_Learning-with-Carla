# Self Learning with CARLA

Bu proje, CARLA simülatörü ile derin pekiştirmeli öğrenme (Deep Reinforcement Learning, DQN) kullanarak otonom araç sürüşü gerçekleştirmek için geliştirilmiştir.

![CARLA Simulator](https://carla.readthedocs.io/en/latest/_static/img/carla.jpg)

## Proje Özeti

Amaç, otonom araçların farklı trafik ve yol koşullarında, kamera görüntülerinden (gri tonlamalı, yeniden boyutlandırılmış) faydalanarak karar verme ve sürüş yeteneklerini geliştirmesidir. Ajan, görüntüleri işler ve uygun sürüş aksiyonunu seçer.

## Ana Özellikler

- **DQN Mimarisi**: CNN tabanlı derin Q-öğrenme ajanı (`DQN/model.py`)
- **Replay Buffer**: Deneyimleri saklamak için yeniden oynatma hafızası (`DQN/replay_buffer.py`)
- **Görüntü İşleme**: Girdi kameradan alınan görüntüler işlenir (`DQN/process_image.py`, ayrıca `environment.py`)
- **CARLA Entegrasyonu**: Simülasyon ortamı ve sensör yönetimi (`environment.py`)
- **Pygame ile Görselleştirme**: Eğitilen ajanın görsel olarak test edilmesi (`agent_visualization_pygame.py`)
- **Model Eğitimi**: Ana eğitim döngüsü (`main.py`)

---

## Kurulum

1. **CARLA Simülatörünü Kurun**
   - [CARLA İndir](https://carla.org/) ve simülatörü `localhost:2000` portunda başlatın.
   - CARLA Python API’sinin sisteminizde kurulu olması gerekir.
2. **Python Bağımlılıklarını Kurun**
   ```bash
   pip install torch torchvision opencv-python pygame numpy carla

---

## Kod Yapısı
.
├── DQN/
│   ├── model.py
│   ├── replay_buffer.py
│   └── process_image.py
├── environment.py
├── main.py
├── agent_visualization_pygame.py
└── README.md

---
## Model eğitimi ve test

   ```bash
   python main.py
   ```

   ```bash
   python main.py
   ```


## Notlar
main.py ve agent_visualization_pygame.py dosyalarının başında yer alan model ve ortam parametrelerini ihtiyacınıza göre düzenleyiniz.
Çalıştırmadan önce CARLA simülatörünü başlatmayı unutmayın!
Eğitim sırasında veya testte simülasyonun yavaşlaması sistem kaynaklarınızla ilgilidir.



   
