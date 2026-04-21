# Encoder-Only-Transformer

**Encoder-only Transformer** mimarisini **from scratch** ve **step by step** inşa etmeyi amaçlayan eğitici bir projedir.

Bu repository’nin amacı, mimariyi sadece kullanmak değil; bileşenlerini tek tek kurarak gerçekten anlamaktır. Proje, **clean code**, **modular design**, test odaklı geliştirme ve öğretici anlatım yaklaşımıyla hazırlanmıştır.

---

## İçindekiler

- [Proje Özeti](#proje-özeti)
- [Bu Proje Neden Var](#bu-proje-neden-var)
- [Projenin Amaçları](#projenin-amaçları)
- [Kapsam](#kapsam)
- [Kapsam Dışı Olanlar](#kapsam-dışı-olanlar)
- [Mevcut Durum](#mevcut-durum)
- [Repository Yapısı](#repository-yapısı)
- [Kurulum](#kurulum)
- [Configuration](#configuration)
- [Nasıl Çalışır](#nasıl-çalışır)
- [Geliştirme Yaklaşımı](#geliştirme-yaklaşımı)
- [Öğrenci / Contributor Workflow](#öğrenci--contributor-workflow)
- [Roadmap](#roadmap)
- [Kimler İçin Uygun](#kimler-için-uygun)
- [Katkı](#katkı)
- [Lisans](#lisans)

---

## Proje Özeti

Transformer tabanlı modeller bugün NLP dünyasının temel yapı taşlarından biridir. Ancak pratikte bu mimariler çoğu zaman hazır kütüphaneler üzerinden kullanıldığı için modelin iç mekanizması yeterince görünür olmaz.

Bu proje, **Encoder-only Transformer** mimarisini bir **black box** gibi kullanmak yerine, onu bileşenlerine ayırarak adım adım inşa etmeyi hedefler.

Buradaki ana fikir şudur:

> Bir mimariyi gerçekten anlamanın en iyi yollarından biri, onu kontrollü kapsamla sıfırdan kurmaktır.

Bu yüzden bu repository’nin amacı sadece çalışan bir model üretmek değil, aynı zamanda şu sorulara net cevap verebilmektir:

- **Token Embedding** neden gereklidir?
- **Positional Encoding** neden eklenir?
- **Self-Attention** nasıl çalışır?
- **Multi-Head Attention** neden tek bir attention mekanizmasından daha güçlüdür?
- **Encoder Block** nasıl oluşur?
- **Stacked Encoder** nasıl kurulur?
- **Pooling** neden gerekir?
- Bir **Encoder-only Transformer**, sequence classification task’lerinde nasıl kullanılır?
- Bu mimari etrafında temiz ve test edilebilir bir kod tabanı nasıl kurulur?

---

## Bu Proje Neden Var

Bugün birçok geliştirici **Transformer** mimarisini kullanıyor, ancak daha az kişi onun iç yapısını gerçekten açıklayabiliyor.

Hazır framework’ler güçlü olsa da öğrenme sürecinde şu sorunları doğurabilir:

- model çalışır ama neden çalıştığı belirsiz kalır
- bileşenler görünmez hale gelir
- teori ile implementation arasındaki bağ zayıflar
- learner, mimariyi tüketir ama içselleştiremez

Bu proje tam olarak bu noktada devreye girer.

Amaç:

- mimariyi görünür hale getirmek
- her bileşeni ayrı ayrı anlamak
- teoriyi doğrudan kod ile ilişkilendirmek
- öğrencinin modeli sadece kullanmasını değil, açıklayabilmesini sağlamak
- küçük ama ciddi bir AI engineering codebase disiplini göstermek

---

## Projenin Amaçları

Bu proje şunları hedefler:

- **Encoder-only Transformer** mimarisini **from scratch** inşa etmek
- her ana bileşeni bağımsız ve modüler şekilde geliştirmek
- mimarinin yalnızca “ne yaptığını” değil, “neden var olduğunu” da açıklamak
- teorik bilgiyi doğrudan implementation ile bağlamak
- öğrenciler, geliştiriciler ve eğitmenler için açık bir öğrenme kaynağı oluşturmak
- public olarak incelenebilecek temiz, testli ve öğretici bir GitHub projesi ortaya koymak

---

## Kapsam

Bu repository’nin temel odağı **Encoder-only Transformer** mimarisidir.

Mevcut ve hedeflenen ana bileşenler:

- **Token Embedding**
- **Positional Encoding**
- **Scaled Dot-Product Attention**
- **Self-Attention**
- **Feed Forward Network**
- **Encoder Block**
- **Stacked Encoder**
- **Pooling**
- **Sequence Classification Head**
- **Full Encoder-based Classification Model**
- **Factory Pattern** ile model üretimi
- **Trainer**
- **Metrics**

Bu yapı, küçük ama anlamlı **text understanding** ve **sequence classification** task’leri için temel oluşturur.

---

## Kapsam Dışı Olanlar

Bu repository’nin mevcut aşamasında bilinçli olarak kapsama alınmayan konular:

- **Decoder-only generation**
- **Large-scale pretraining**
- **Full BERT reproduction**
- **Masked Language Modeling**
- **Advanced tokenizer training**
- **Production deployment**
- çok büyük dataset ve çok maliyetli training pipeline’ları

Bu karar bilinçlidir. Bu repo bir “everything NLP framework” değil, kontrollü kapsamlı bir **öğrenme laboratuvarı**dır.

---

## Mevcut Durum

Şu anda repository şu seviyeye ulaşmıştır:

- modüler **Encoder-only architecture core** kurulmuştur
- ana bileşenler ayrı katmanlar halinde yazılmıştır
- **SequenceClassificationHead** ve tam model akışı eklenmiştir
- **Factory Pattern** ile model kurulum katmanı eklenmiştir
- **trainer.py** ve **metrics.py** ile training foundation oluşturulmuştur
- klasör yapısı refactor edilerek daha ölçeklenebilir hale getirilmiştir
- kapsamlı **unit test** yapısı kurulmuştur

Başka bir deyişle: bu repo artık yalnızca mimari parçaların anlatıldığı bir taslak değil; **çalışan, test edilen ve genişletilebilir bir Encoder-only sequence classification foundation** haline gelmiştir.

---

## Repository Yapısı

```text
Encoder-Only-Transformer/
├── README.md
├── GUIDE.md
├── TASKS.md
├── PROJECT_CONTEXT.md
├── DECISIONS.md
├── CHANGELOG.md
├── pyproject.toml
├── uv.lock
├── .gitignore
├── .python-version
│
├── config/
│   └── default.yaml
│
├── docs/
│   └── architecture_notes.md
│
├── src/
│   └── encoder_only_transformer/
│       ├── __init__.py
│       │
│       ├── config/
│       │   ├── __init__.py
│       │   └── config.py
│       │
│       ├── layers/
│       │   ├── __init__.py
│       │   ├── attention.py
│       │   ├── embeddings.py
│       │   ├── feed_forward.py
│       │   └── pooling.py
│       │
│       ├── blocks/
│       │   ├── __init__.py
│       │   └── encoder_block.py
│       │
│       ├── models/
│       │   ├── __init__.py
│       │   ├── encoder.py
│       │   ├── heads.py
│       │   └── model.py
│       │
│       ├── factories/
│       │   ├── __init__.py
│       │   └── factories.py
│       │
│       └── training/
│           ├── __init__.py
│           ├── trainer.py
│           └── metrics.py
│
└── tests/
    ├── __init__.py
    ├── test_attention.py
    ├── test_config.py
    ├── test_encoder.py
    ├── test_encoder_block.py
    ├── test_embeddings.py
    ├── test_factories.py
    ├── test_feed_forward.py
    ├── test_heads.py
    ├── test_metrics.py
    ├── test_model.py
    ├── test_package_exports.py
    ├── test_pooling.py
    ├── test_self_attention.py
    └── test_trainer.py
```

### Klasör Açıklamaları

- `config/`  
  Runtime configuration dosyaları. Şu an `default.yaml` burada tutulur.

- `docs/`  
  Teknik notlar, mimari açıklamalar ve ek dökümanlar.

- `src/encoder_only_transformer/config/`  
  Typed config objects, config loading ve validation logic.

- `src/encoder_only_transformer/layers/`  
  En küçük mimari bileşenler: embedding, attention, feed forward, pooling.

- `src/encoder_only_transformer/blocks/`  
  Daha büyük yapılar. Şu an `EncoderBlock` burada bulunur.

- `src/encoder_only_transformer/models/`  
  Encoder stack, head ve full model burada tutulur.

- `src/encoder_only_transformer/factories/`  
  Config-driven model construction logic.

- `src/encoder_only_transformer/training/`  
  Training loop ve metric logic.

- `tests/`  
  Her ana modül için unit test dosyaları.

---

## Kurulum

Bu projede environment ve dependency yönetimi için **uv** kullanılmaktadır.

### Gereksinimler

- Python **3.10**
- **uv**

### Proje kurulumu

```bash
uv sync
```

### Testleri çalıştırma

```bash
uv run pytest -v
```

### Lint çalıştırma

```bash
uv run ruff check .
```

---

## Configuration

Bu projede iki ayrı configuration katmanı vardır.

### 1. `pyproject.toml`

Bu dosya aşağıdakiler için kullanılır:

- project metadata
- dependency management
- pytest configuration
- ruff configuration

### 2. `config/default.yaml`

Bu dosya model ve training parametrelerini tutar.

Örnek alanlar:

- `vocab_size`
- `max_seq_len`
- `d_model`
- `n_heads`
- `ff_hidden_dim`
- `dropout`
- `n_layers`
- `batch_size`
- `learning_rate`
- `weight_decay`
- `epochs`

Bu ayrım bilinçlidir. Proje metadata ile deneysel model parametrelerini aynı yerde tutmak yerine iki farklı katmanda yönetmek daha sağlıklıdır.

---

## Nasıl Çalışır

Repository içindeki temel akış şu şekildedir:

1. **input_ids** alınır
2. `EncoderInputEmbedding` ile token representation + positional information üretilir
3. Bu temsil, çok katmanlı `Encoder` içine girer
4. Encoder çıktısı sequence-level representation’a indirgenir
5. `SequenceClassificationHead` ile logits üretilir
6. `SequenceClassificationTrainer` ile loss ve metric hesaplanır

Kavramsal akış:

```text
input_ids
  -> TokenEmbedding
  -> PositionalEncoding
  -> EncoderBlock x N
  -> Pooling
  -> Classification Head
  -> logits
  -> loss / accuracy
```

---

## Geliştirme Yaklaşımı

Bu proje bilinçli olarak **step by step** geliştirilmektedir.

Her yeni aşamada şu yaklaşım izlenir:

1. önce ilgili bileşenin amacı netleştirilir
2. sonra minimal ama doğru implementation yazılır
3. ardından unit test eklenir
4. gerekirse kısa teknik açıklama ile dokümante edilir
5. bir sonraki bileşene ancak önceki yapı anlaşılır hale geldikten sonra geçilir

Bu yaklaşımın nedeni şudur:

Bir mimariyi anlamaya çalışırken aynı anda çok fazla dosya, çok fazla abstraction ve çok fazla feature görmek öğrenmeyi zorlaştırır. Bu repository’de hızdan çok **anlaşılırlık**, **sorumluluk ayrımı** ve **test edilebilirlik** önemlidir.

---

## Öğrenci / Contributor Workflow

Bu repository, sadece bireysel öğrenme için değil; mentor kontrollü contributor geliştirme süreci için de uygundur.

Önerilen contributor akışı:

1. repository’yi clone et
2. environment’ı kur
3. tüm testleri geçir
4. mevcut mimariyi teknik olarak açıkla
5. atanmış feature alanında kendi branch’in üzerinde çalış
6. test ekle
7. PR aç
8. review sonrası merge al

### Branch yaklaşımı

- `main` yalnızca kontrollü ve temiz sürümler için korunmalıdır
- contributor’lar feature branch üzerinde çalışmalıdır

Örnek branch isimleri:

- `feature/data-pipeline`
- `feature/training-script`
- `feature/evaluation-metrics`

Detaylı contributor ve öğrenci çalışma kuralları için:

- `GUIDE.md`
- `TASKS.md`
- `DECISIONS.md`
- `CHANGELOG.md`
- `PROJECT_CONTEXT.md`

dosyalarına bakılmalıdır.

---

## Roadmap

### Tamamlanan Ana Aşamalar

- project setup
- config system
- embeddings
- attention core
- feed forward
- encoder block
- stacked encoder
- pooling
- classification head
- full model
- factory layer
- trainer
- metrics
- package refactor
- test stabilization

### Sıradaki Muhtemel Aşamalar

- end-to-end training example script
- toy dataset / dataset utility
- evaluation script
- checkpoint save/load
- daha zengin metrics
- sentence pair task
- ablation / pooling comparison experiments
- docs ve architecture visual improvements

---

## Kimler İçin Uygun

Bu repository özellikle şu kişiler için uygundur:

- **Transformer architecture** öğrenmek isteyen öğrenciler
- teoriyi kodla bağlamak isteyen geliştiriciler
- derslerinde öğretici bir repo kullanmak isteyen eğitmenler
- hazır kütüphaneleri kullanmadan önce encoder mimarisinin içini görmek isteyen mühendisler

Bu repo tamamen başlangıç seviyesi için tasarlanmış değildir. En çok faydayı, kod okuyarak öğrenmeye açık ve mimariyi gerçekten anlamak isteyen kişiler alır.

---

## Katkı

Bu proje eğitim odaklı bir repository’dir.

Katkı yapmak isteyenler şu alanlarda katkı sunabilir:

- test kapsamını genişletme
- training example ekleme
- metrics iyileştirme
- docs ve architecture notes geliştirme
- bug fix
- küçük deneyler ve analizler
- contributor workflow iyileştirmeleri

Katkı sunmadan önce repository’nin eğitim odaklı yapısını dikkate alın. Öncelik her zaman:

- anlaşılabilirlik
- modülerlik
- test edilebilirlik
- öğretici değer

olmalıdır.

---

## Lisans

Bu proje **MIT License** ile paylaşılacaktır.

---

## Son Not

Bu repository’nin amacı sadece çalışan bir model yazmak değildir.

Buradaki asıl hedef:

- bir mimariyi görünür hale getirmek
- bileşenleri tek tek anlamak
- temiz engineering disiplini göstermek
- öğrenciyi kademeli olarak daha güçlü AI engineering işlerine hazırlamak
- ve public olarak gerçekten faydalı bir öğrenme kaynağı oluşturmaktır
