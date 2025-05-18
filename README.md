# Self Learning with CARLA

Bu proje, CARLA simülatörü kullanarak pekiştirmeli öğrenme (reinforcement learning) ile otonom araç sürüş becerilerinin geliştirilmesini amaçlamaktadır.

![CARLA Simulator](https://carla.readthedocs.io/en/latest/_static/img/carla.jpg)

## Proje Hakkında

Bu çalışma, otonom araçların gerçek dünya koşullarında karşılaşabileceği çeşitli senaryolarda karar verme yeteneklerini geliştirmek için makine öğrenmesi tekniklerini kullanmaktadır. CARLA simülatörü üzerinde gerçekleştirilen deneyler, araçların kendi kendine öğrenme kabiliyetini artırmayı hedeflemektedir.

# CARLA ile Otonom Sürüş için DQN Tabanlı RL Ajanı

Bu proje, CARLA simülasyon ortamında Derin Q-Öğrenme (DQN) kullanarak otonom araç kontrolü sağlayan bir yapay zeka ajanı içerir. Ajan, kamera girdilerini işleyerek araç kontrol kararları alır ve Pygame ile gerçek zamanlı görselleştirme sunar.

---

## 🛠 Teknolojiler ve Bağımlılıklar
- **Python 3.8+**
- **CARLA Simülasyon Sunucusu** ([İndirme Linki](https://carla.org/))
- **PyTorch** (Derin Öğrenme Modeli)
- **OpenCV** (Görüntü İşleme)
- **Pygame** (Görselleştirme)
- **NumPy, Queue, Collections**

### Kurulum
1. **CARLA'yı İndirin** ve `localhost:2000` portunda çalıştırın.
2. Gereken Python kütüphanelerini yükleyin:
   ```bash
   pip install torch torchvision opencv-python pygame numpy carla

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
