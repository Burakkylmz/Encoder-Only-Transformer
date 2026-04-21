# Changelog

Bu dosya, proje sürecindeki anlamlı milestone seviyesindeki değişiklikleri özetler.

## [2026-04-19] — Repository initialized and project foundation created
### Added
- `uv` tabanlı proje kurulumu yapıldı
- `pyproject.toml` oluşturuldu
- YAML config yapısı oluşturuldu
- İlk README omurgası hazırlandı

### Notes
- Proje educational encoder-only transformer repository olarak konumlandırıldı.

---

## [2026-04-19] — Core encoder architecture completed
### Added
- `TokenEmbedding`, `SinusoidalPositionalEncoding`, `EncoderInputEmbedding`
- `ScaledDotProductAttention`
- `SelfAttention`
- `PositionwiseFeedForward`
- `EncoderBlock`
- `Encoder`
- Pooling stratejileri
- `SequenceClassificationHead`
- `EncoderForSequenceClassification`
- `ModelFactory`

### Changed
- Core architecture modüler OOP yapısı ile organize edildi

### Notes
- Sequence classification için uçtan uca çalışan encoder-only core tamamlandı.

---

## [2026-04-19] — Test coverage expanded across architecture
### Added
- Config testleri
- Embedding testleri
- Attention testleri
- Self-attention testleri
- Feed-forward testleri
- Encoder block testleri
- Encoder testleri
- Pooling testleri
- Head testleri
- Full model testleri
- Factory testleri
- Package export testleri

### Fixed
- 3D attention mask broadcasting bug'ı düzeltildi
- Refactor sonrası kırılan import path'leri düzeltildi

### Notes
- Tüm mevcut çekirdek modüller testlerle doğrulandı.

---

## [2026-04-19] — Package refactor completed
### Added
- `config/`, `layers/`, `blocks/`, `models/`, `factories/`, `training/` klasör yapısı oluşturuldu
- Subpackage export düzeni yapıldı

### Changed
- Flat source structure, maintainable subpackage yapısına taşındı
- Import path'leri yeni yapıya göre güncellendi

### Notes
- Refactor, training/data katmanı büyümeden önce kontrollü şekilde tamamlandı.

---

## [2026-04-19] — Training and metrics utilities added
### Added
- `training/trainer.py`
- `training/metrics.py`
- Loss + accuracy akışı
- `SequenceClassificationBatch`, `StepResult`, `EpochResult`

### Changed
- Trainer, metric computation katmanını ayrı modülden kullanacak şekilde güncellendi

### Notes
- Repo artık yalnızca mimari gösterimi değil, temel training workflow'u da desteklemektedir.

---

## [2026-04-19] — Student contribution planning started
### Added
- Student contribution strategy tanımlandı
- Documentation templates project-specific hale getirilmeye başlandı
- Öğrenci onboarding rehberi planlandı

### Notes
- Sonraki aşama: dataset/data pipeline, experiment script ve student feature ownership modeli.
