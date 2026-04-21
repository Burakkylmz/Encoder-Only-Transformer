# TASKS

Bu dosya, **Encoder-Only-Transformer** projesinin görev takibi için kullanılır.

Amaç:
- mevcut proje durumunu görünür kılmak,
- sıradaki işleri netleştirmek,
- öğrenci katkı alanlarını açık şekilde ayırmak,
- milestone seviyesinde ilerlemeyi takip etmektir.

Bu dosyada küçük commit düzeyindeki her değişiklik tutulmaz. Yalnızca anlamlı görev alanları, aktif çalışma başlıkları ve tamamlanan aşamalar izlenir.

---

## Project Status

- **Current phase:** Development → Student Contribution Phase
- **Current focus:** Student task execution, dataset/data pipeline, experiment script, evaluation expansion
- **Last updated:** 2026-04-20

---

## Current Priority

Bu aşamada öncelik sırası aşağıdaki gibidir:

1. Öğrencilerin repository'yi ayağa kaldırması
2. Mevcut mimariyi teknik olarak anlaması ve açıklaması
3. Görevlerin branch bazlı dağıtılması
4. Öğrenci implementasyonlarının başlaması
5. Experiment pipeline ve documentation tarafının olgunlaştırılması

---

## Student Task Assignment

Bu bölüm, öğrencilere atanacak görevleri açık biçimde tanımlar.  
Her öğrenci kendi sorumluluk alanında çalışmalıdır. Aynı çekirdek dosyalara eş zamanlı müdahale edilmemelidir.

### Öğrenci A — Data / Input Pipeline

#### Core Tasks
- [ ] `training/datasets.py` oluştur
- [ ] `collate_fn` implement et
- [ ] Padding stratejisini belirle
- [ ] Küçük bir toy dataset loader yaz

#### Extended Tasks
- [ ] Dynamic padding yaklaşımını implement et
- [ ] Static padding yaklaşımını implement et
- [ ] Dynamic vs static padding farkını kısa not ile açıkla
- [ ] `padding_mask` üret
- [ ] Gerekirse mask'i attention uyumlu forma dönüştür

#### Deliverables
- Çalışan data pipeline
- Doğru padding ve mask üretimi
- Kısa teknik not veya açıklama dosyası

---

### Öğrenci B — Training Workflow / Experiments

#### Core Tasks
- [ ] `experiments/train_sequence_classification.py` yaz
- [ ] Config üzerinden model kur
- [ ] Training loop oluştur
- [ ] Checkpoint save/load desteği ekle

#### Extended Tasks
- [ ] Epoch bazlı logging ekle
- [ ] Loss ve accuracy çıktısı üret
- [ ] Deterministic training opsiyonu ekle
- [ ] Basit evaluation step ekle

#### Deliverables
- Çalıştırılabilir training script
- Log üreten training akışı
- Checkpoint mekanizması
- Basit example run çıktısı

---

### Öğrenci C — Evaluation / Docs / Analysis

#### Core Tasks
- [ ] `precision`, `recall`, `f1` metriklerini ekle veya genişlet
- [ ] Evaluation helper fonksiyonları yaz
- [ ] Pooling stratejilerini karşılaştır
- [ ] README kullanım örneklerini güncelle
- [ ] `docs/architecture_notes.md` dosyasını genişlet

#### Extended Tasks
- [ ] Confusion matrix utility yaz
- [ ] Mean pooling / max pooling / first token pooling karşılaştırmasını yap
- [ ] Kısa deney raporu hazırla
- [ ] Raporu `docs/` altında uygun bir dosya olarak ekle

#### Deliverables
- Genişletilmiş metrics modülü
- Analiz notları
- Güncellenmiş dokümantasyon
- Kısa karşılaştırma raporu

---

## Student Workflow Requirements

Her öğrenci aşağıdaki akışı takip etmelidir:

- [ ] Repository'yi clone et
- [ ] Environment'ı kur
- [ ] Tüm mevcut testleri çalıştır
- [ ] Mevcut mimariyi teknik olarak açıkla
- [ ] Kendi task alanı için branch aç
- [ ] Geliştirmeyi branch üzerinde yap
- [ ] Gerekli testleri ekle veya güncelle
- [ ] Pull Request aç
- [ ] Review sonrası merge sürecine gir

---

## Backlog

Bu bölüm öğrenci görevlerinden bağımsız, genel proje backlog'unu içerir.

- [ ] Küçük ama öğretici gerçek dataset seç
- [ ] End-to-end training örneği ekle
- [ ] README içine tam training usage örneği koy
- [ ] Example run output ekle
- [ ] Checkpoint usage örneği ekle
- [ ] Evaluation sonucu örneği ekle

---

## In Progress

- [ ] Student onboarding tamamlanıyor
- [ ] Görev dağılımı netleştirildi
- [ ] Öğrenci çalışma alanları branch bazlı ayrılıyor
- [ ] Katkı süreci guide ve task dokümanlarıyla hizalanıyor

---

## Blocked

- [ ] Gerçek dataset seçimi
  - **Reason:** Önce student ownership ve feature implementasyonları netleşmeli
  - **Needed action:** Eğitim açısından uygun, küçük ve anlaşılır bir sequence classification dataset seç

---

## Done

- [x] Repository kuruldu
- [x] `uv` environment kuruldu
- [x] `pyproject.toml` ve YAML config yapısı kuruldu
- [x] `ModelConfig`, `TrainingConfig`, `ProjectConfig` tasarlandı
- [x] `TokenEmbedding` yazıldı
- [x] `SinusoidalPositionalEncoding` yazıldı
- [x] `EncoderInputEmbedding` yazıldı
- [x] `ScaledDotProductAttention` yazıldı
- [x] Mask desteği eklendi
- [x] `SelfAttention` yazıldı
- [x] `PositionwiseFeedForward` yazıldı
- [x] `EncoderBlock` yazıldı
- [x] `Encoder` yazıldı
- [x] Pooling stratejileri eklendi
- [x] `SequenceClassificationHead` yazıldı
- [x] `EncoderForSequenceClassification` modeli yazıldı
- [x] `ModelFactory` eklendi
- [x] Package export düzeni yapıldı
- [x] Klasör refactor tamamlandı
- [x] `SequenceClassificationTrainer` yazıldı
- [x] `SequenceClassificationMetrics` yazıldı
- [x] Trainer ve metrics entegrasyonu tamamlandı
- [x] Tüm mevcut testler başarılı hale getirildi

---

## Phase Breakdown

### Phase 1 — Foundation and Architecture
- [x] Problem statement yazıldı
- [x] Learning use case netleştirildi
- [x] Scope tanımlandı
- [x] Modüler mimari ayrıştırıldı
- [x] Temel test stratejisi kuruldu

### Phase 2 — Core Encoder Implementation
- [x] Input representation katmanı yazıldı
- [x] Attention çekirdeği yazıldı
- [x] Self-attention katmanı yazıldı
- [x] Feed-forward katmanı yazıldı
- [x] Encoder block oluşturuldu
- [x] Stacked encoder oluşturuldu
- [x] Pooling ve classification head eklendi
- [x] Full sequence classification modeli kuruldu

### Phase 3 — Refactor and Package Structure
- [x] Flat structure'dan subpackage yapısına geçildi
- [x] Import path'leri düzeltildi
- [x] Public package API güncellendi
- [x] Test suite refactor sonrası tekrar yeşile getirildi

### Phase 4 — Training Foundation
- [x] `training/trainer.py` oluşturuldu
- [x] `training/metrics.py` oluşturuldu
- [x] Loss + accuracy akışı bağlandı
- [ ] Dataset / dataloader katmanı eklenecek
- [ ] Experiment script eklenecek

### Phase 5 — Student Contribution Phase
- [ ] Öğrenciler repository'yi ayağa kaldıracak
- [ ] Öğrenciler mevcut mimariyi teknik olarak anlatacak
- [ ] Görevler öğrencilere branch bazlı dağıtılacak
- [ ] Student implementasyonları başlayacak
- [ ] Pull Request review süreci işletilecek

### Phase 6 — Finalization
- [ ] README son kez güncellenecek
- [ ] CHANGELOG güncellenecek
- [ ] Example run çıktıları eklenecek
- [ ] Demo akışı hazırlanacak
- [ ] Son cleanup yapılacak

---

## Milestones

### Milestone 1 — Repository and Core Setup
- [x] Repository açıldı
- [x] Config sistemi tamamlandı
- [x] Test altyapısı kuruldu

### Milestone 2 — Encoder Core Ready
- [x] Core encoder bileşenleri yazıldı
- [x] Full model çalışır hale geldi
- [x] Temel unit test coverage tamamlandı

### Milestone 3 — Refactor and Training Base
- [x] Package refactor tamamlandı
- [x] Trainer eklendi
- [x] Metrics eklendi
- [x] Tüm testler tekrar başarılı hale getirildi

### Milestone 4 — Student Contribution Phase
- [ ] Student branch'leri açıldı
- [ ] Assignment ownership netleşti
- [ ] İlk implementasyonlar başladı
- [ ] İlk PR'ler açıldı

### Milestone 5 — Learning Repository Ready
- [ ] Experiment script hazır
- [ ] Example training akışı hazır
- [ ] README ve docs güncellendi
- [ ] Repository öğrenci walkthrough ve contribution için hazır hale geldi

---

## Notes

Bu dosyanın amacı öğrencileri yönlendirmek ve ilerlemeyi görünür kılmaktır.  
Teknik kararların gerekçeleri için `DECISIONS.md`, proje bağlamı için `PROJECT_CONTEXT.md`, çalışma kuralları için `GUIDE.md`, önemli değişiklikler için `CHANGELOG.md` referans alınmalıdır.
