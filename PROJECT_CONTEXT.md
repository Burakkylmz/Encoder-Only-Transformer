# Project Context

Bu dosya, **Encoder-Only-Transformer** projesinin arka planını, kapsamını, teknik yönünü ve temel varsayımlarını açıklamak için hazırlanmıştır.

## 1. Project Title
Encoder-Only-Transformer

## 2. Project Goal
Bu projenin temel amacı, **Encoder-only Transformer** mimarisini sıfırdan, modüler ve öğretici bir şekilde inşa etmek; öğrencilerin hem mimariyi anlamasını hem de bu yapıyı sequence classification gibi görevlerde genişletebilmesini sağlamaktır.

## 3. Business / Use Case Context
Bu proje bir **eğitim ve öğrenme repository**'sidir.
Ana kullanım alanı:
- AI / NLP / Transformer eğitimi veren eğitmenler
- Encoder-only mimariyi anlamak isteyen öğrenciler
- PyTorch ile from-scratch mimari görmek isteyen geliştiriciler

Bu repository doğrudan production product geliştirmek için değil, **BERT-style encoder mantığını görünür kılmak** için tasarlanmıştır.

## 4. Problem Statement
Transformer tabanlı encoder modeller çoğu zaman hazır framework'ler üzerinden kullanılır. Bu da öğrencilerin:
- attention mekanizmasını,
- encoder block yapısını,
- pooling ve classification ilişkisini,
- modelin task pipeline'a nasıl bağlandığını

yeterince görünür şekilde anlamasını zorlaştırır.

Bu proje, bu problemi çözmek için mimariyi **parça parça ve test odaklı** biçimde inşa eder.

## 5. Target User
- Encoder-only Transformer mimarisini derinlemesine öğrenmek isteyen öğrenciler
- Teknik eğitim alan katılımcılar
- Eğitmen gözetiminde feature geliştirecek student contributors

## 6. Target Workflow
Örnek workflow:

**Input IDs → Embedding → Encoder → Pooling → Classification Head → Trainer / Metrics → Output**

Detay akış:
1. Token ID'ler modele girer.
2. Token embeddings ve positional encoding uygulanır.
3. Sequence, stacked encoder blokları içinden geçirilir.
4. Pooling ile sequence-level representation çıkarılır.
5. Classification head logits üretir.
6. Trainer loss ve accuracy hesaplar.
7. Sonuç, eğitim / evaluation sürecinde raporlanır.

## 7. In Scope
Bu proje kapsamında yapılacaklar:
- Encoder-only Transformer mimarisinin from-scratch implementasyonu
- Modüler PyTorch bileşenleri
- Sequence classification pipeline
- Trainer ve metric katmanı
- Test odaklı geliştirme
- Öğrenci contribution workflow'u
- Küçük demo / experiment script'leri

## 8. Out of Scope
Bu proje kapsamında yapılmayacaklar:
- Full BERT reproduction
- Masked Language Modeling pretraining (ilk aşamada)
- Large-scale distributed training
- Production deployment
- Heavy tokenizer research pipeline
- Decoder-only generation

## 9. Input Types
Projede şu input türleri kullanılacaktır:
- Token ID tensörleri
- Padding mask tensörleri
- Gelecek aşamada basit text classification dataset örnekleri

## 10. Expected Output
Sistemin üretmesi beklenen çıktılar:
- Classification logits
- Attention weight'leri
- Loss
- Accuracy
- Eğitim / evaluation özetleri

## 11. Technical Direction
### Planned components
- Config loader
- Input embedding katmanı
- Attention katmanları
- Feed-forward katmanı
- Encoder block
- Stacked encoder
- Pooling stratejileri
- Classification head
- Full model
- Trainer
- Metrics
- Future dataset / experiment scripts

### Initial tech choices
- **Language:** Python 3.10
- **Framework:** Custom PyTorch implementation
- **Deep Learning Library:** PyTorch
- **Config:** `pyproject.toml` + YAML
- **Testing:** pytest
- **Linting:** ruff
- **Environment / dependency management:** uv
- **Interface:** CLI / script-based

## 12. Constraints
Projeyi etkileyen kısıtlar:
- Eğitim odaklı olduğu için kod çok karmaşık olmamalı
- Compute maliyeti düşük tutulmalı
- Öğrenciler tarafından ayağa kaldırılabilir olmalı
- Core mimari gereksiz dependency ile şişirilmemeli
- Scope büyümesi kontrol altında tutulmalı

## 13. Assumptions
Projede başlangıçta kabul edilen varsayımlar:
- Öğrenci temel Python ve OOP bilgisine sahiptir
- Öğrenci pytest çalıştırabilir ve repo'yu lokalinde ayağa kaldırabilir
- Sequence classification, ilk task olarak yeterince öğreticidir
- Modüler tasarım, öğrenci katkılarını yönetmeyi kolaylaştıracaktır

## 14. Risks
Muhtemel riskler:
- Scope creep
- Öğrencilerin core mimariyi anlamadan genişletmeye çalışması
- Aynı dosyalarda fazla çakışma nedeniyle merge conflict oluşması
- Dokümantasyonun kodla birlikte güncellenmemesi
- Eğitim repo'sunun gereksiz enterprise karmaşıklığına kayması

## 15. Success Criteria
Bu projenin başarılı sayılması için minimum kriterler:
- Repo sıfırdan kurulup testler başarılı şekilde çalıştırılabiliyor olmalı
- Encoder-only core mimari açıklanabilir ve incelenebilir olmalı
- Sequence classification pipeline çalışıyor olmalı
- Trainer + metrics akışı çalışıyor olmalı
- Öğrenciler repo'yu anlatıp belirli feature'ları ekleyebilmeli
- Repo, public learning resource olarak düzenli ve okunur görünmeli

## 16. Notes
Bu repository şu an için “BERT replacement” olarak değil, **BERT-style encoder systems için güçlü bir educational foundation** olarak konumlandırılmalıdır.
