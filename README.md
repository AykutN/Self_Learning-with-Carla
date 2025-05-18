# Self Learning with CARLA

Bu proje, CARLA simülatörü kullanarak pekiştirmeli öğrenme (reinforcement learning) ile otonom araç sürüş becerilerinin geliştirilmesini amaçlamaktadır.

![CARLA Simulator](https://carla.readthedocs.io/en/latest/_static/img/carla.jpg)

## Proje Hakkında

Bu çalışma, otonom araçların gerçek dünya koşullarında karşılaşabileceği çeşitli senaryolarda karar verme yeteneklerini geliştirmek için makine öğrenmesi tekniklerini kullanmaktadır. CARLA simülatörü üzerinde gerçekleştirilen deneyler, araçların kendi kendine öğrenme kabiliyetini artırmayı hedeflemektedir.

## Özellikler

- CARLA simülatörü ile entegre çalışma
- Derin pekiştirmeli öğrenme algoritmaları implementasyonu
- Çeşitli sürüş senaryoları için eğitim ortamları
- Performans metriklerinin toplanması ve analizi
- Model eğitimi ve test sonuçlarının görselleştirilmesi

## Kurulum

### Gereksinimler

- Python 3.7+
- CARLA 0.9.10+
- PyTorch 1.7.0+
- TensorFlow 2.3.0+ (isteğe bağlı)
- Numpy, Matplotlib, Pandas

### Kurulum Adımları

1. Repoyu klonlayın:
```bash
git clone https://github.com/AykutN/Self_Learning-with-Carla.git
cd Self_Learning-with-Carla
```

2. Gerekli Python paketlerini yükleyin:
```bash
pip install -r requirements.txt
```

3. CARLA simülatörünü [resmi web sitesinden](https://carla.org/download/) indirin ve kurun.

## Kullanım

### Eğitim Başlatma

```bash
python train.py --scenario basic --episodes 1000 --model ddpg
```

### Test Etme

```bash
python test.py --model-path models/my_model.pth --scenario highway
```

### Sonuçları Görselleştirme

```bash
python visualize.py --log-dir logs/experiment1
```

## Proje Yapısı

```
Self_Learning-with-Carla/
├── agents/                  # Ajan implementasyonları
├── environments/            # CARLA ortam adaptörleri
├── models/                  # Eğitilmiş modeller
├── scripts/                 # Yardımcı scriptler
├── utils/                   # Yardımcı fonksiyonlar
├── config.py                # Konfigürasyon ayarları
├── train.py                 # Eğitim script'i
├── test.py                  # Test script'i
└── visualize.py             # Görselleştirme araçları
```

## Sonuçlar

Bu bölümde, farklı eğitim senaryoları ve modelleriyle elde edilen performans sonuçlarını paylaşabilirsiniz. Grafikler, tablolar veya örnek videolar ekleyebilirsiniz.

## Kaynaklar

- [CARLA Simulator](https://carla.org/)
- [Deep Reinforcement Learning for Autonomous Driving](https://arxiv.org/abs/1810.06581)
- [RL in Autonomous Driving: A Survey](https://arxiv.org/abs/2002.00444)

## Lisans

Bu proje [MIT Lisansı](LICENSE) altında lisanslanmıştır.

## İletişim

Aykut N - [GitHub](https://github.com/AykutN)

Proje Linki: [https://github.com/AykutN/Self_Learning-with-Carla](https://github.com/AykutN/Self_Learning-with-Carla)
```

## Katkıda Bulunma

1. Bu repoyu fork edin
2. Feature branch'inizi oluşturun (`git checkout -b feature/amazing-feature`)
3. Değişikliklerinizi commit edin (`git commit -m 'Add some amazing feature'`)
4. Branch'inize push edin (`git push origin feature/amazing-feature`)
5. Pull Request açın
