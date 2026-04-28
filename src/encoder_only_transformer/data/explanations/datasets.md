
#### Data / Input Pipeline
Ham token dizilerini modelin işleyebileceği formata dönüştürür.
- Pipeline:
```
    raw text  →  token_ids (list[int])  →  Dataset  →  DataLoader (collate_fn)
                                                        ↓
                                            input_ids  (batch, seq_len)      ← padded
                                            labels     (batch,)
                                            padding_mask (batch, seq_len)    ← 1=gerçek, 0=pad
                                                        ↓  [opsiyonel]
                                            attention_mask (batch, 1, 1, seq_len)
```

#### def make_collate_fn():
- DataLoader için collate_fn üretir.
```
    Padding stratejisi:
        max_seq_len verilmişse  -> Statik padding: her batch sabit uzunlukta
        max_seq_len=None ise    -> Dinamik padding: batch içindeki en uzun diziye göre padding

    Döndürür: (input_ids, labels, padding_mask)
        input_ids    : (batch_size, seq_len)  — long
        labels       : (batch_size,)           — long
        padding_mask : (batch_size, seq_len)  — long, 1=gerçek token 0=pad
```

#### def build_padding_mask():
2D padding mask üretir.
```
    Giriş : input_ids (batch_size, seq_len)
    Çıkış : padding_mask (batch_size, seq_len)
                1 → gerçek token
                0 → padding
```

#### def build_attention_mask():
- Padding mask'i attention mekanizmasının beklediği 4D forma sokar.
```

    Giriş : padding_mask (batch_size, seq_len)
    Çıkış : attention_mask (batch_size, 1, 1, seq_len)
```
- 1 ve 2. unsqueeze'ler sayesinde n_heads ve query konumlarına broadcast edilebilir hale gelir.

#### def make_toy_dataloader():
- Eğitim dışı hızlı test için küçük bir DataLoader döndürür.
```
    max_seq_len=None  → dinamik padding (varsayılan)
    max_seq_len=<int> → statik padding
```

#### compare_padding_strategies():
- Aynı sample listesi için statik ve dinamik padding istatistiklerini hesaplar.
```
Statik: her örnek max_seq_len'e kadar padding alır (+ truncate)
    Dinamik: her batch içindeki en uzun diziye göre padding yapılır (burada tüm veri tek batch kabul edilir)
```

### Dinamik vs Statik Padding
##### STATİK PADDING (max_seq_len sabit)
- Her dizi, veri setindeki en uzun diziye veya belirlenen bir üst sınıra kadar pad edilir. Tensor boyutu her batch'te aynı kalır.
- `Bellek:`Kısa diziler bile tam uzunluğa pad aldığı için gereksiz token işlenir. Örneğin max_seq_len=128, gerçek uzunluk=5 ise 123 sıfır taşınır; bu da ~25x fazla bellek demektir.
- `Hız:` Tensor boyutu sabit olduğu için GPU/TPU kernel başlatma maliyeti tekdüzedir. XLA derleyicisi (TPU) veya torch.compile() gibi araçlar bu durumda daha iyi optimize edebilir çünkü shape değişmez.


- Ne zaman tercih edilir?
  - TPU eğitimi (XLA sabit shape ister)
  - torch.compile() ile statik grafikler
  - Veri setindeki dizi uzunlukları birbirine yakınsa

 #### DİNAMİK PADDING (batch içindeki en uzun diziye göre)
- Her batch kendi içinde en uzun diziye göre pad edilir. Farklı batch'ler farklı `seq_len`'e sahip olabilir.

- `Bellek:` Her batch sadece kendi içindeki maksimum uzunluğa kadar padding alır. Uzunluklar çok değişkense (örn. 2–50 arası), ortalama bellek tüketimi statik padding'e göre %40–80 daha az olabilir.
- `Hız:` collate_fn sırasında max() hesabı ve değişken tensor boyutu küçük bir CPU maliyeti getirir. GPU'da ise kısa batch'lerde daha az işlem yapıldığı için hız genellikle eşit ya da daha iyi. Ancak her batch'te shape değiştiği için JIT/XLA optimizasyonu güçleşir.


- Ne zaman tercih edilir?
  - CPU/GPU eğitimi (özellikle NLP görevlerinde standart seçim)
  - Dizi uzunlukları çok değişkense (örn. soru-cevap, özetleme)
  - Bellek kısıtı varsa

#### ÖZET
```
   ┌─────────────┬──────────────┬──────────────┐
   │     Kriter  │   Statik     │   Dinamik    │
   ├─────────────┼──────────────┼──────────────┤
   │ Bellek      │ Yüksek       │ Düşük        │
   │ Hız (GPU)   │ ≈ eşit       │ ≈ eşit/iyi   │
   │ Hız (TPU)   │ Daha iyi     │ Sorunlu      │
   │ Kod karmaş. │ Düşük        │ Düşük        │
   │ JIT uyumu   │ İyi          │ Kötü         │
   └─────────────┴──────────────┴──────────────┘
```