# Decisions

Bu dosya, proje boyunca alınan temel teknik ve mimari kararları özetler.

---

## 1) Project Positioning
**Decision:** Repository, full BERT reproduction olarak değil, **Encoder-only Transformer educational foundation** olarak konumlandırılacaktır.

**Reasoning:**
- Proje eğitim odaklıdır.
- Öğrencilerin mimariyi anlaması, feature parity'den daha önemlidir.
- “BERT'in yapabildiği her şeyi yapar” söylemi bu aşamada teknik olarak erken olurdu.

**Alternatives considered:**
- Full BERT clone
- Production-ready NLP library

**Consequence:**
- Scope kontrollü kalır.
- Repo daha dürüst ve öğretici olur.
- Geliştirme odağı mimari görünürlüğünde kalır.

**Status:** Confirmed

---

## 2) Start with Encoder-only Architecture
**Decision:** İlk mimari olarak decoder-only yerine **encoder-only** yapı seçildi.

**Reasoning:**
- Encoder-only yapı öğrenme açısından daha temiz bir başlangıç sağlar.
- Classification ve sentence-level understanding task'leri için uygundur.
- Generation, sampling ve KV cache gibi ek karmaşıklıklar erken aşamada scope'u büyütürdü.

**Alternatives considered:**
- Decoder-only architecture
- Encoder-decoder architecture

**Consequence:**
- Mimari öğretimi daha sade hale gelir.
- İlk downstream task olarak sequence classification seçimi doğal hale gelir.

**Status:** Confirmed

---

## 3) Build from Scratch with PyTorch
**Decision:** Core architecture, hazır high-level model sınıfları yerine **custom PyTorch modules** ile from scratch yazılacaktır.

**Reasoning:**
- Amaç black-box kullanımı değil, mimari görünürlüğüdür.
- Her bileşenin test edilip açıklanabilmesi istenmektedir.
- Öğrenci katkısı için modüllerin okunabilir olması önemlidir.

**Alternatives considered:**
- Hugging Face Transformers tabanlı wrapper yaklaşımı
- Notebook-first prototyping

**Consequence:**
- Öğrenme değeri artar.
- Kod miktarı artar ancak mimari şeffaflık kazanılır.

**Status:** Confirmed

---

## 4) Use uv + pyproject.toml + YAML
**Decision:** Environment management için `uv`, proje yapılandırması için `pyproject.toml`, model/training ayarları için YAML kullanılacaktır.

**Reasoning:**
- Modern Python proje yönetimi için temiz bir kurulum sağlar.
- Proje metadata ile deneysel config değerlerini ayırmak daha doğrudur.
- Öğrenciler için reproducible setup daha kolay hale gelir.

**Alternatives considered:**
- `venv` + `requirements.txt`
- Tek dosyada tüm config değerlerini tutmak

**Consequence:**
- Kurulum ve geliştirme akışı daha düzenli olur.
- Config sistemi daha maintainable hale gelir.

**Status:** Confirmed

---

## 5) Modular OOP Design with Explicit Validation
**Decision:** Kod, OOP tabanlı ve sorumlulukları ayrılmış modüller halinde yazılacaktır; input / shape validation açık şekilde yapılacaktır.

**Reasoning:**
- Repo öğrenci katkısına açık olacak.
- SRP, test edilebilirlik ve maintainability hedeflenmektedir.
- Transformer implementasyonlarında hataların büyük kısmı shape seviyesinde olur.

**Alternatives considered:**
- Tek dosyada hızlı prototip
- Validation'sız sade forward akışı

**Consequence:**
- Kod biraz daha verbose olur.
- Ancak bakım, anlatım ve debugging ciddi biçimde kolaylaşır.

**Status:** Confirmed

---

## 6) Pooling as Strategy
**Decision:** Pooling mantığı `BasePooling` contract'ı üzerinden ayrı stratejiler halinde modellenmiştir.

**Reasoning:**
- Mean / first-token / max gibi yaklaşımlar downstream task açısından karşılaştırılabilir olmalıdır.
- Classification head, pooling mantığını içine gömmemelidir.

**Alternatives considered:**
- Head içine sabit pooling gömmek
- Sadece mean pooling kullanmak

**Consequence:**
- Strategy Pattern benzeri genişleyebilir yapı kazanıldı.
- Öğrenciler pooling davranışlarını daha rahat karşılaştırabilir.

**Status:** Confirmed

---

## 7) Separate Trainer and Metrics
**Decision:** Training logic ile metric computation logic ayrı modüllerde tutulacaktır.

**Reasoning:**
- Trainer yalnızca training/evaluation akışını yönetmelidir.
- Accuracy ve gelecekteki metrikler bağımsız katmanda genişletilebilmelidir.

**Alternatives considered:**
- Accuracy hesaplarını trainer içinde gömmek

**Consequence:**
- `trainer.py` daha temiz kaldı.
- `metrics.py` future extension için hazırlandı.

**Status:** Confirmed

---

## 8) Full Model Builds Attention Mask Internally
**Decision:** Full model, dışarıdan `padding_mask` alacak; attention seviyesinde kullanılacak 4D mask'i kendi içinde oluşturacaktır.

**Reasoning:**
- Kullanıcı API'sini sadeleştirir.
- Padding semantics ile attention implementation detayını ayırır.

**Alternatives considered:**
- Kullanıcının doğrudan 4D attention mask vermesi

**Consequence:**
- Model kullanımı daha ergonomik hale gelir.
- İç mask dönüşümleri merkezi hale gelir.

**Status:** Confirmed

---

## 9) Refactor to Subpackages Before Data/Training Expansion
**Decision:** Core architecture tamamlandıktan sonra, training/data katmanı büyümeden önce subpackage refactor yapılacaktır.

**Reasoning:**
- Flat structure ilk aşamada öğretici olsa da ölçeklenebilir son yapı değildir.
- `training/`, `layers/`, `models/`, `blocks/`, `factories/` ayrımı büyümeyi kolaylaştırır.

**Alternatives considered:**
- Flat structure ile devam etmek
- Daha en başta erken refactor yapmak

**Consequence:**
- Refactor kontrollü zamanda yapıldı.
- Sonraki feature'lar daha temiz klasör yapısına eklenecek.

**Status:** Confirmed

---

## 10) Student Contributions Should Extend, Not Destabilize, the Core
**Decision:** Öğrenci katkıları, mümkün olduğunca core architecture'ı bozmak yerine data, experiment, evaluation ve docs alanlarında yönlendirilecektir.

**Reasoning:**
- Çekirdek mimari şu an oturmuş durumda.
- Aynı core dosyalara aynı anda çok kişi dokunursa repo dağılabilir.
- Eğitim açısından önce sistemi anlamaları, sonra kontrollü alanlarda genişletmeleri daha doğru.

**Alternatives considered:**
- Core model dosyalarını doğrudan öğrencilere bölmek

**Consequence:**
- Merge conflict ve mimari bozulma riski azalır.
- Öğrenci katkıları daha yönetilebilir hale gelir.

**Status:** Confirmed
