# Gerçek Zamanlı El Hareketi Tanıma ile Türk İşaret Dili Çevirmeni

Gömülü bir sistem üzerinde çalışan, kamera aracılığıyla elde edilen görüntülerden el hareketlerini analiz ederek **Türk İşaret Dili (TİD) alfabesindeki harfleri gerçek zamanlı olarak tanıyan** bir bilgisayarla görme sistemidir.

---

## İçindekiler

- [Amaç](#amaç)
- [Yaklaşım](#yaklaşım)
- [Sistem Mimarisi](#sistem-mimarisi)
- [Kullanılan Teknolojiler](#kullanılan-teknolojiler)
- [Hedef Donanım Platformları](#hedef-donanım-platformları)
- [Kurulum](#kurulum)
- [Kullanım](#kullanım)
- [Veri Seti](#veri-seti)
- [Proje Yapısı](#proje-yapısı)
- [Yol Haritası](#yol-haritası)
- [Katkıda Bulunma](#katkıda-bulunma)

---

## Amaç

- İşaret dilini bilmeyen bireyler ile işitme engelli bireyler arasında iletişimi kolaylaştırmak.
- Düşük donanım kaynaklarına sahip sistemlerde gerçek zamanlı görüntü işleme gerçekleştirmek.
- Bilgisayarla görme ve derin öğrenme yöntemlerini gömülü sistemlere entegre etmek.

---

## Yaklaşım

Sistem **landmark tabanlı** bir pipeline kullanır: kameradan alınan görüntü doğrudan bir sınıflandırıcıya verilmez; önce MediaPipe ile el üzerindeki 21 anahtar nokta çıkarılır, sonra bu koordinatlar özel olarak eğitilmiş bir modelle sınıflandırılır.

**Neden landmark tabanlı?** Projenin gömülü sistem hedefi (Raspberry Pi, Jetson Nano) nedeniyle:

| Kriter | Ham Görüntü CNN | Landmark (seçilen) |
|---|---|---|
| Raspberry Pi hızı | 1-5 FPS | 15-30 FPS |
| Model boyutu | 20-100 MB | 0.5-2 MB |
| Veri ihtiyacı | Binlerce/harf | Yüzlerce/harf |
| Işık/arka plan dayanıklılığı | Düşük | Yüksek |
| Doğruluk (2024 araştırmalar) | %96-99 | %97-99 |

Sınıflandırıcı iki seviyelidir:
- **Statik harfler** için yoğun katmanlı (MLP/CNN) model
- **Dinamik harfler** (Ç, Ğ, J gibi hareket içeren) için LSTM

---

## Sistem Mimarisi

```
┌───────────────┐   ┌──────────────┐   ┌──────────────────────┐   ┌────────────────┐   ┌─────────────────┐
│ Görüntü Alımı │ → │  Önişleme    │ → │ El Tespiti &         │ → │ Özellik        │ → │ Sınıflandırma   │
│  (Kamera)     │   │ (RGB/flip)   │   │ Landmark (21 nk.)    │   │ Mühendisliği   │   │ (MLP / LSTM)    │
│               │   │              │   │ MediaPipe Tasks API  │   │ (normalize)    │   │                 │
└───────────────┘   └──────────────┘   └──────────────────────┘   └────────────────┘   └─────────────────┘
                                                                                                │
                                                                                                ▼
                                                                                         Tahmin Edilen Harf
```

### 1. Görüntü Alımı

USB kamera veya gömülü kamera modülü ile gerçek zamanlı video akışı alınır.

### 2. Önişleme

- BGR → RGB dönüşümü (MediaPipe RGB bekler)
- Ayna etkisi için yatay çevirme
- Kare boyutu ayarlama

### 3. El Tespiti ve Landmark Çıkarımı

**MediaPipe Tasks API** (`HandLandmarker`) ile el üzerindeki 21 landmark `(x, y, z)` normalize koordinat olarak çıkarılır. Önceden eğitilmiş `hand_landmarker.task` modeli (Google tarafından) kullanılır.

### 4. Özellik Mühendisliği

- **Normalizasyon:** Bilek orijin olarak alınır, orta parmak MCP ölçek referansı
- **Türetilmiş özellikler:** Parmak eklem açıları, nokta-nokta mesafeleri
- Kameradan uzaklık ve el pozisyonundan bağımsız temsil

### 5. Sınıflandırma

- **Statik harfler:** Tek kare landmark vektörü → MLP/CNN
- **Dinamik harfler:** Landmark sekansı `(T, 21, 3)` → LSTM

---

## Kullanılan Teknolojiler

| Katman | Teknoloji |
| --- | --- |
| Dil | Python 3.12 |
| Görüntü İşleme | OpenCV 4.13 |
| El Tespiti | MediaPipe 0.10.33 (Tasks API) |
| Derin Öğrenme | TensorFlow 2.21 / Keras 3.14 |
| Sayısal İşlem | NumPy 2.4 |
| Model Mimarisi | MLP (statik), LSTM (dinamik) |

> **Not:** MediaPipe 0.10.33'te legacy `mp.solutions.hands` kaldırılmıştır. Kod, yeni **Tasks API** (`mediapipe.tasks.python.vision.HandLandmarker`) kullanır.

---

## Hedef Donanım Platformları

- Raspberry Pi (4 / 5)
- NVIDIA Jetson Nano

---

## Kurulum

### 1. Depoyu klonla

```bash
git clone https://github.com/<kullanici>/TurkishSignLanguageTranslator.git
cd TurkishSignLanguageTranslator
```

### 2. Sanal ortam oluştur (Python 3.12 gereklidir)

```bash
py -3.12 -m venv venv
source venv/Scripts/activate   # Windows (Git Bash)
# veya
venv\Scripts\activate          # Windows (CMD / PowerShell)
# veya
source venv/bin/activate       # Linux / macOS
```

> **Neden Python 3.12?** MediaPipe 0.10.33 ve TensorFlow 2.21 Windows üzerinde Python 3.13 için önceden derlenmiş wheel sağlamaz.

### 3. Bağımlılıkları yükle

```bash
pip install -r requirements.txt
```

### 4. Önceden eğitilmiş MediaPipe modelini indir

```bash
python scripts/download_models.py
```

Bu komut Google'ın `hand_landmarker.task` dosyasını (~7.6 MB) `models/` klasörüne indirir.

---

## Kullanım

### Canlı landmark demosu (doğrulama)

Kamera açılır, tespit edilen 21 nokta ve parmak bağlantıları çizilir:

```bash
python src/demo_landmarks.py
python src/demo_landmarks.py --camera 1 --hands 2
```

Çıkış: `q` tuşu.

### Gerçek zamanlı çevirmen (yapım aşamasında)

```bash
python src/main.py --camera 0
```

### Model eğitimi (yapım aşamasında)

```bash
python src/train.py --model mlp
```

---

## Veri Seti

Proje **landmark tabanlı** olduğu için eğitim verisi `(N, 21, 3)` şeklinde NPY dosyalarıdır, görüntü değil.

İki veri kaynağı kullanılabilir:

### 1. Hazır TİD görüntü veri setinden dönüştürme (önerilen)

- Kaynak: [Kaggle — Turkish Sign Language (Fingerspelling)](https://www.kaggle.com/datasets/feronial/turkish-sign-languagefinger-spelling)
- `scripts/convert_dataset.py` her görüntüyü MediaPipe'tan geçirir ve landmark NPY olarak kaydeder
- **Avantaj:** 30+ kişiden toplanmış çeşitlilik, hızlı başlangıç

### 2. Kendi verinizi toplama

- `python src/collect_data.py --label A`
- Kameranızla harf başına 200+ örnek kaydeder
- **Avantaj:** Sizin kullanım koşullarınıza özel, ince ayar için faydalı

İdeal iş akışı: Kaggle verisiyle temel modeli eğit, sonra kendi verinizle ince ayar yap.

---

## Proje Yapısı

```
TurkishSignLanguageTranslator/
├── data/                        # Landmark dataset (NPY, gitignore'da)
├── models/                      # .task ve eğitilmiş .h5 modelleri (gitignore'da)
├── notebooks/                   # Deneme ve analiz not defterleri
├── scripts/
│   ├── download_models.py       # MediaPipe .task indirici
│   └── convert_dataset.py       # Görüntü → landmark dönüştürücü (yapım aşamasında)
├── src/
│   ├── preprocessing/           # Görüntü önişleme yardımcıları
│   ├── landmarks/
│   │   └── hand_detector.py     # HandLandmarkExtractor (Tasks API)
│   ├── features/                # Özellik mühendisliği
│   ├── models/                  # MLP & LSTM model tanımları
│   ├── demo_landmarks.py        # Canlı kamera + landmark görselleştirme
│   ├── collect_data.py          # Veri toplama (yapım aşamasında)
│   ├── train.py                 # Eğitim (yapım aşamasında)
│   └── main.py                  # Gerçek zamanlı çıkarım (yapım aşamasında)
├── .env.example
├── requirements.txt
└── README.md
```

---

## Yol Haritası

- [x] Proje iskeleti ve bağımlılık yönetimi
- [x] MediaPipe Tasks API ile landmark çıkarıcı
- [x] Canlı kamera + landmark görselleştirme demosu
- [ ] Kaggle TID dataset → landmark dönüştürme script'i
- [ ] Özellik mühendisliği (normalize, açı, mesafe)
- [ ] Statik harfler için MLP/CNN modeli
- [ ] Dinamik harfler için LSTM modeli
- [ ] Harf-sekansı → kelime mantığı (sabitlik filtresi)
- [ ] Raspberry Pi için TFLite dönüşümü
- [ ] Jetson Nano için TensorRT entegrasyonu

---

## Katkıda Bulunma

Katkılar memnuniyetle karşılanır. Lütfen bir issue açın veya pull request gönderin.
