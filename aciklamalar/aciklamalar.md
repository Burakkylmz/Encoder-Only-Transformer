


### Token Embedding:
- kelimeler `d_model` sayesinde anlam kazanır.
- batch_size -> cümle sayısı 
- seq_len -> token sayısı 
- d_model -> tokenların boyutu
  - (2, 3, 4) -> 2 cümle (batch_size), her cümle 3 token (seq_len), her token 4 boyutlu (d_model)



### Positional Encoding: 
- kelimeler sıra yapısını da öğrenir
- Token Embedding'den çıkan vektörlerde kelime anlamı var ama kelime sırası yok. Çünkü Attention tüm tokenlara aynı anda bakıyor. Bunun için tokenlara sıra bilgisi eklenir.
- her sıra için vektör üretiliyor ve token embedding vektörünün üzerinde toplanıyor.
* `PE(pos, çift indeks) = sin(pos / 10000^(i/d_model))`  
* `PE(pos, tek indeks)  = cos(pos / 10000^(i/d_model))`
    - her sıra için farklı sin/cos dalgaları üretiyor `[-1, +1]`. bu sayede her sıra benzersiz oluyor..!
```
input_ids (batch, seq_len)
        ↓ TokenEmbedding
token embeddings (batch, seq_len, d_model)   <- anlam
        ↓ + PE
encoder input (batch, seq_len, d_model)   <- anlam + sıra
```


### Self Attention (Multi-Head):
- her token cümledeki diğer tokenlara bakarak sorgular.
- her token'dan 3 farklı vektör üretilir.
    - *Q -> Query (sorgu)*
    - *K -> Key (eşleşen anahtar)*
    - *V -> Value (sonuç)*
- Q ve K karşılaştırılır = score
- score * V => sonuç

Multi-Head Attention ile model aynı anda farklı ilişkileri öğrenebilir..!
- Q, K, V üretilir ve ayrı linear katmanlardan geçirilip head'lere bölünür. sonra her head için attention hesaplanır ve headler birleştirilir. böylece her head kendi bakış açısını öğrenir ve bu bilgiler ortak şekilde harmanlanır..!



### Feed Forward Network: (forward propagation)
- mantık olarak forward propagation ile aynı. normalde FFN tüm sekansı bir bütün olarak işler. Attention mekanizasına özgü farklılık olarak ise her tokeni bağımsız olarak FFN işler ve bu sayede Attention tokenlar arası ilişkileri öğrenir, FFN'de her token'in temsilini derinleştirir..!
- kod kısmındaki önce genişletip sonra daraltmanın sebebi geniş ara katmanın modele daha fazla öğrenme kapasitesi vermesidir..!
- Not: Transformer mimarisinde `GELU (x < 0 yani küçük negatif değerler kalsın)` daha iyi sonuç verir. (PRelu, Leaky Relu vb.)
```
(batch, seq_len, d_model)
    ↓ Linear
(batch, seq_len, hidden_dim) <- genişledi
    ↓ GELU → Dropout
    ↓ Linear
(batch, seq_len, d_model) <- daraldı, shape aynıya döndü
```


### Encoder Block:
- self attention ve FFN'i alıp Residual Connection (artık bağlantı) + LayerNorm ile birleştirir.
```
    x
    ↓
    Self-Attention
    ↓
    x + attention_output ← Residual Connection
    ↓
    LayerNorm
    ↓
    Feed Forward
    ↓
    x + ff_output ← Residual Connection
    ↓
    LayerNorm
    ↓
    output
```
- Residual Connection => girdiyi çıktıya toplayarak geçirir ve böylece vanishing gradient sorunun çözer.
- x = x + attention_output vb.
- Keras için `Add()` katmanı ile yapılır.
- Layer Normalization => her token'ın vektörünü normalize eder.
- Keras için `BatchNormalization()` ile benzer mantıktadır.
- BatchNormalization() -> batch boyunca normalize eder yani batch'e bağımlıdır
- Layer Normalization -> her token'i kendi içinde normalize eder yani batch'ten bağımsızdır.
- Transformer'larda Layer Normalization kullanılır çünkü seq_len değişken olabilir. eğer batchnorm kullanılsaydı batch bağımlılık sorun yaratabilirdi..!
```
  x (batch, seq_len, d_model)
    ↓ Self-Attention
  attention_output (batch, seq_len, d_model)
    ↓ Dropout + Residual + LayerNorm
  x (batch, seq_len, d_model)
    ↓ Feed Forward
  ff_output (batch, seq_len, d_model)
    ↓ Dropout + Residual + LayerNorm
  output (batch, seq_len, d_model)   <- aynı
```


Stacked Encoder:
- Aynı encoder bloğu N kez üst üste konar.
- Her bir N encoder bloğu öncekinin çıktısını alıp daha zengin bir temsil üretir.
```
    embedding çıktısı
        ↓
    Encoder Block 1   <- basit ilişkiler
        ↓
    Encoder Block 2   <- daha karmaşık ilişkiler
        ↓
    Encoder Block N   <- en soyut yüksek seviye temsil
        ↓
    output
```


### Pooling Layer:
- (CNN'de kullanılan MaxPooling, AveragePooling vb. boyutu küçültüp önemli feature'lar korunurdu)
- `seq_len` yerine tek vektör üretilir ve tüm sequence bir cümle temsiline indirilir..!
- `(batch_size, seq_len, d_model) -> (batch_size, d_model)`    -> yani seq_len kaybolur.
- `FirsTokenPooling` -> sadece ilk token'ı alır.
- `MaxPooling` -> her boyutta max değeri alır
- `MeanPooling` -> seq_len boyutunu ortalar
```
(batch_size, seq_len, d_model)
    ↓ Pooling
(batch_size, d_model) <- seq_len silindi, tek vektör kaldı
```


### Classification Head:
- (self._classifier = nn.Linear(d_model, num_classes) = keras -> Dense(num_classes))
- Pooling'ten gelen `(batch_size, d_model)` boyutundaki tek vektörü sınıf sayısındaki *logitlere* dönüştürür.
- Pytorch'da loss function içinde softmax olduğundan ayrıca softmax uygulanmaz..!
```
(batch, seq_len, d_model)
    ↓ Pooling
(batch, d_model)
    ↓ Dropout
(batch, d_model)
    ↓ Linear
(batch, num_classes) <- logitler
```


### Tam Model:
```
input_ids                    (batch, seq_len)
    ↓ TokenEmbedding
                            (batch, seq_len, d_model)
    ↓ PositionalEncoding
                            (batch, seq_len, d_model)
    ↓ EncoderBlock × N
                            (batch, seq_len, d_model)
    ↓ Pooling
                            (batch, d_model)
    ↓ Dropout + Linear
logits                      (batch, num_classes)
```