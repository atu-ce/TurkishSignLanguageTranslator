# Gerçek Zamanlı El Hareketi Tanıma ile Türk İşaret Dili Çevirmeni

Gömülü bir sistem üzerinde çalışan, kamera aracılığıyla elde edilen görüntülerden el hareketlerini analiz ederek **Türk İşaret Dili (TİD) alfabesindeki harfleri gerçek zamanlı olarak tanıyan** bir bilgisayarla görme sistemidir.

---

## İçindekiler

- [Amaç](#amaç)
- [Sistem Mimarisi](#sistem-mimarisi)
- [Kullanılan Teknolojiler](#kullanılan-teknolojiler)
- [Hedef Donanım Platformları](#hedef-donanım-platformları)
- [Kurulum](#kurulum)
- [Kullanım](#kullanım)
- [Proje Yapısı](#proje-yapısı)
- [Yol Haritası](#yol-haritası)
- [Katkıda Bulunma](#katkıda-bulunma)

---

## Amaç

- İşaret dilini bilmeyen bireyler ile işitme engelli bireyler arasında iletişimi kolaylaştırmak.
- Düşük donanım kaynaklarına sahip sistemlerde gerçek zamanlı görüntü işleme gerçekleştirmek.
- Bilgisayarla görme ve derin öğrenme yöntemlerini gömülü sistemlere entegre etmek.

---

## Sistem Mimarisi

Sistem, aşağıdaki ardışık işlem hattı (pipeline) üzerinden çalışır:

```
┌──────────────┐   ┌──────────────┐   ┌───────────────────┐   ┌────────────────┐   ┌─────────────────┐
│ Görüntü Alımı │ → │  Önişleme    │ → │ El Tespiti &      │ → │ Özellik        │ → │ Sınıflandırma   │
│  (Kamera)     │   │ (Resize/NR)  │   │ Landmark (21 nk.) │   │ Çıkarımı       │   │ (CNN / LSTM)    │
└──────────────┘   └──────────────┘   └───────────────────┘   └────────────────┘   └─────────────────┘
                                                                                             │
                                                                                             ▼
                                                                                      Tahmin Edilen Harf
```

### 1. Görüntü Alımı (Image Acquisition)

USB kamera veya gömülü kamera modülü kullanılarak gerçek zamanlı video akışı elde edilir.

### 2. Önişleme (Preprocessing)

- RGB → BGR dönüşümü
- Görüntü yeniden boyutlandırma
- Gürültü azaltma (noise reduction)

### 3. El Tespiti ve Landmark Çıkarımı

**MediaPipe** kütüphanesi kullanılarak her karede elin **21 adet landmark (anahtar nokta)** koordinatı çıkarılır.

### 4. Özellik Çıkarımı (Feature Extraction)

- Landmark koordinatları normalize edilir.
- Noktalar arası **mesafe** ve **açı** bilgileri hesaplanır.

### 5. Sınıflandırma (Classification)

Elde edilen özellik vektörü derin öğrenme modeline verilir:

- **CNN** → Statik harfler (A, B, C, …) için.
- **LSTM** → Zamansal hareket gerektiren harfler (J, Ğ vb.) için.

---

## Kullanılan Teknolojiler

| Katman         | Teknoloji                   |
| -------------- | --------------------------- |
| Dil            | Python 3.9+                 |
| Görüntü İşleme | OpenCV                      |
| El Tespiti     | MediaPipe                   |
| Derin Öğrenme  | TensorFlow / Keras, PyTorch |
| Sayısal İşlem  | NumPy                       |
| Model Mimarisi | CNN, LSTM                   |

---

## Hedef Donanım Platformları

- Raspberry Pi (4 / 5)
- NVIDIA Jetson Nano

---

## Kurulum

```bash
git clone https://github.com/<kullanici>/TurkishSignLanguageTranslator.git
cd TurkishSignLanguageTranslator

python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

---

## Kullanım

Gerçek zamanlı çevirmeni başlatmak için:

```bash
python src/main.py --camera 0
```

Kendi veri kümenizi toplamak için:

```bash
python src/collect_data.py --label A
```

Modeli yeniden eğitmek için:

```bash
python src/train.py --model cnn
```

---

## Proje Yapısı

```
TurkishSignLanguageTranslator/
├── data/                 # Ham ve işlenmiş veri
├── models/               # Eğitilmiş model dosyaları
├── notebooks/            # Deneme ve analiz not defterleri
├── src/
│   ├── preprocessing/    # Önişleme modülleri
│   ├── landmarks/        # MediaPipe landmark çıkarımı
│   ├── features/         # Özellik mühendisliği
│   ├── models/           # CNN & LSTM model tanımları
│   ├── train.py          # Eğitim betiği
│   ├── collect_data.py   # Veri toplama betiği
│   └── main.py           # Gerçek zamanlı çıkarım
├── requirements.txt
└── README.md
```

---

## Yol Haritası

- [ ] Veri toplama arayüzü
- [ ] MediaPipe landmark çıkarım modülü
- [ ] Statik harfler için CNN modeli
- [ ] Dinamik harfler için LSTM modeli
- [ ] Raspberry Pi üzerinde optimize çıkarım
- [ ] Jetson Nano TensorRT entegrasyonu
- [ ] Harf dizisini kelimeye dönüştüren son işleme katmanı

---

## Katkıda Bulunma

Katkılar memnuniyetle karşılanır. Lütfen bir issue açın veya pull request gönderin.
