# Student Guide

Bu rehber, **Encoder-Only-Transformer** projesinde çalışacak öğrenciler için hazırlanmıştır.
Amaç yalnızca kod yazmak değil; mevcut sistemi anlayarak, kontrollü ve profesyonel bir şekilde katkı sunmaktır.

---

## 1. Bu projedeki rolünüz

Bu proje, sıfırdan yazılmış bir **Encoder-only Transformer** öğrenme repository'sidir.
Sizden beklenen şey doğrudan rastgele feature eklemek değildir.

Önce:
- repository'yi ayağa kaldırmanız,
- mevcut mimariyi anlamanız,
- testleri çalıştırmanız,
- sistemin nasıl çalıştığını teknik olarak anlatabilmeniz

beklenir.

Bundan sonra size atanmış feature alanlarında geliştirme yapmanız istenir.

---

## 2. Sizden beklenen çalışma modeli

Bu projede şu sırayla ilerlemeniz beklenir:

### Aşama 1 — Repository'yi ayağa kaldır
- Repo'yu clone et
- `uv sync` çalıştır
- testleri çalıştır
- package yapısını incele

### Aşama 2 — Sistemi anla
Aşağıdaki akışı teknik olarak açıklayabilir hale gel:

`input_ids -> embeddings -> encoder -> pooling -> classification head -> trainer -> metrics`

### Aşama 3 — Kısa walkthrough yap
Eğitmene veya takım liderine şu başlıkları anlat:
- Proje ne yapıyor?
- Hangi klasör hangi sorumluluğu taşıyor?
- Attention katmanı nasıl çalışıyor?
- Pooling neden gerekli?
- Trainer ve metrics nasıl bağlanıyor?

### Aşama 4 — Assigned feature üzerinde çalış
Size atanan alan dışında core mimariye doğrudan müdahale etmeyin.

### Aşama 5 — Test + PR aç
Yaptığınız iş:
- test içermeli,
- açıklanabilir olmalı,
- kendi branch'inizde geliştirilmeli,
- PR ile gönderilmelidir.

---

## 3. Kurulum

### Gereksinimler
- Python 3.10
- `uv`

### Setup
```bash
uv sync
```

### Testleri çalıştır
```bash
uv run pytest -v
```

### Lint
```bash
uv run ruff check .
```

---

## 4. Proje yapısını nasıl okuyacaksınız?

### `config/`
Config nesneleri, YAML loading ve validation burada yer alır.

### `layers/`
En küçük mimari bileşenler burada bulunur:
- embeddings
- attention
- feed forward
- pooling

### `blocks/`
Layer'ların birleştiği daha büyük yapı burada bulunur:
- `EncoderBlock`

### `models/`
Task-level ve model-level yapılar burada bulunur:
- `Encoder`
- `SequenceClassificationHead`
- `EncoderForSequenceClassification`

### `factories/`
Config tabanlı model kurulum mantığı burada bulunur.

### `training/`
Training loop ve metrics burada bulunur.

### `tests/`
Her önemli modül için contract testleri burada bulunur.

---

## 5. Bu projede nasıl düşünmeniz gerekiyor?

Bu repository'de amaç sadece “çalışan kod” yazmak değildir.
Sizden şu sorulara cevap verebilen katkılar beklenir:

- Bu değişiklik neden gerekli?
- Kodun sorumluluğu doğru yerde mi?
- Test ekledim mi?
- Mevcut mimariyi bozuyor muyum, yoksa genişletiyor muyum?
- Bu katkıyı başka bir öğrenci okuyabilir mi?

---

## 6. Zorunlu çalışma kuralları

### 6.1 Doğrudan `main` branch üzerinde çalışmayın
Her öğrenci kendi feature branch'i üzerinde çalışmalıdır.

Örnek branch isimleri:
- `feature/student-a-data-pipeline`
- `feature/student-b-training-script`
- `feature/student-c-metrics-docs`

### 6.2 Anlamlı commit mesajı kullanın
İyi örnekler:
- `add batch collation for padded sequence classification data`
- `implement experiment script for sequence classification training`
- `extend metrics with macro f1 support`

Kötü örnekler:
- `update`
- `fix`
- `last version`

### 6.3 Test eklemeden PR açmayın
Yeni feature varsa test de olmalıdır.

### 6.4 README / docs etkileniyorsa güncelleyin
Kod değiştiyse dokümantasyon da güncellenmelidir.

### 6.5 Core dosyalara izinsiz büyük müdahale yapmayın
Özellikle şu dosyalara kontrollü yaklaşın:
- `models/model.py`
- `models/encoder.py`
- `blocks/encoder_block.py`
- `layers/attention.py`

Bu dosyalarda büyük refactor veya davranış değişikliği gerekiyorsa önce tartışın.

---

## 7. Kod standardı

Bu projede aşağıdaki standart beklenir:

- Type hint kullan
- OOP ve SRP'yi koru
- Girdi validasyonu yap
- Gereksiz abstraction ekleme
- Private method'ları yalnızca gerçekten anlamlıysa kullan
- Public class ve fonksiyonlara açıklayıcı docstring yaz
- Şekil (`shape`) contract'larını açık tut

Bu projede hedef:
**senior disiplin, gereksiz enterprise karmaşıklığı değil**

---

## 8. Öğrenci Görev Dağılımı — Encoder-Only-Transformer

## Genel Bakış

Bu proje, modüler geliştirme, net sorumluluk dağılımı ve birlikte çalışma alışkanlığı kazandırmak amacıyla üç öğrenci arasında bölünmüştür.

Her öğrenci belirli bir alt sistemden sorumludur. Amaç sadece kod yazmak değil, aynı zamanda yapılan tasarım kararlarını anlamak ve açıklayabilmektir.

---

## Öğrenci A — Data / Input Pipeline

### Ana Görevler
- `datasets.py` dosyasını implement et
- `collate_fn` yaz
- Padding stratejisi tasarla
- Küçük bir toy dataset loader oluştur

### Ek Görevler

#### 1. Dynamic vs Static Padding Karşılaştırması
- Sabit `max_seq_len` padding uygula
- Batch bazlı dynamic padding uygula
- Kısa bir karşılaştırma yaz (performans, memory)

#### 2. Attention Mask Üretimi
- `padding_mask` üret
- (Opsiyonel) Attention uyumlu forma çevir:
  `(batch_size, 1, 1, seq_len)`

### Amaç
Ham verinin modelin anlayabileceği forma nasıl dönüştüğünü anlamak.

---

## Öğrenci B — Training Workflow / Experiments

### Ana Görevler
- `experiments/train_sequence_classification.py` yaz
- Config üzerinden model kur
- Checkpoint save/load implement et
- Çalışan bir training loop oluştur

### Ek Görevler

#### 1. Basit Logging
- Her epoch için:
  - loss
  - accuracy
- Console çıktısı ver

Örnek:
```
Epoch 1 | loss: 0.65 | acc: 0.72
```

#### 2. Deterministic Training Opsiyonu
- Reproducibility sağla:
  - `torch.manual_seed(...)`
- Opsiyonel config flag ekle

### Amaç
Bir modelin nasıl eğitildiğini, izlendiğini ve tekrar üretilebilir hale getirildiğini anlamak.

---

## Öğrenci C — Evaluation / Docs / Analysis

### Ana Görevler
- `precision`, `recall`, `f1` implement et
- Evaluation helper fonksiyonları yaz
- Pooling yöntemlerini karşılaştır
- README örneklerini geliştir
- Architecture notları yaz

### Ek Görevler

#### 1. Confusion Matrix Utility
- Torch veya numpy ile implement et
- Prediction sonuçlarını matrix olarak üret

#### 2. Mini Deney Raporu
- Şunları karşılaştır:
  - mean pooling
  - max pooling
  - CLS token
- 1–2 sayfalık kısa rapor yaz
- `docs/` klasörüne koy

### Amaç
Model performansının nasıl ölçüldüğünü ve yorumlandığını anlamak.

---

## Genel Kurallar

- Her öğrenci ayrı bir branch üzerinde çalışmalıdır
- `main` branch’e direkt commit atılmaz
- Her feature şunları içermelidir:
  - temiz kod
  - type hint
  - mümkünse test
- Pull Request açılmalı ve review alınmalıdır

---

## Beklenen Çıktı

Bu aşamanın sonunda:
- Projede:
  - çalışan bir data pipeline
  - çalıştırılabilir bir training script
  - doğru evaluation metrikleri olacak
- Öğrenciler:
  - pipeline’ın tamamını anlamış olacak
  - gerçek bir codebase’e katkı sunmuş olacak
  - ekip çalışması deneyimi kazanacak

---

## Son Not

Öncelik sırası:
- anlaşılabilirlik
- doğruluk
- modülerlik

Hızlı yazılmış ama anlaşılmayan kodun hiçbir değeri yoktur.

Bu çalışma sadece kod yazma değil, mühendislik pratiğidir.

---

## 9. Çalışmaya başlamadan önce yapmanız gereken kısa kontrol listesi

Aşağıdaki soruların hepsine “evet” diyebiliyor olmalısınız:

- Repo'yu lokalimde ayağa kaldırdım mı?
- Testleri çalıştırdım mı?
- Mimari akışı açıklayabiliyor muyum?
- Hangi klasör hangi işi yapıyor biliyor muyum?
- Bana atanan görev alanı net mi?
- Hangi dosyaları değiştirmem gerektiğini biliyor muyum?

---

## 10. PR açmadan önce kontrol listesi

- [ ] Branch'im doğru mu?
- [ ] Değişiklik kapsamım atanan görev ile uyumlu mu?
- [ ] Test ekledim mi?
- [ ] Tüm testleri çalıştırdım mı?
- [ ] Gerekli import ve package path'leri doğru mu?
- [ ] Dokümantasyon etkileniyorsa güncelledim mi?
- [ ] Commit mesajlarım anlamlı mı?
- [ ] PR açıklamam neyi neden yaptığımı anlatıyor mu?

---

## 11. Bu projede kaçınmanız gereken şeyler

- Aynı anda çok fazla dosyayı rastgele değiştirmek
- Çekirdek mimariyi anlamadan refactor yapmak
- Testsiz geliştirme
- README / docs güncellemeden feature eklemek
- Hazır internet kodunu yapıştırıp mantığını açıklayamamak
- “Çalışıyor gibi” ama contract'ı bozan değişiklikler

---

## 12. Sizden gerçekten beklenen şey

Bu projede amaç yalnızca “bir feature eklemek” değildir.
Sizden beklenen:
- mevcut sistemi anlamanız,
- teknik olarak anlatmanız,
- kontrollü bir şekilde genişletmeniz,
- profesyonel GitHub çalışma disiplini göstermenizdir.

Bu repository bir ödev deposu gibi değil, küçük ölçekli bir **mentored engineering project** gibi ele alınmalıdır.
