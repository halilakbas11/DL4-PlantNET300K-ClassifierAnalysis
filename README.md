# Çoklu Model Destekli Bitki & Ağaç Türü Kıyaslama ve Sınıflandırma Platformu: Pl@ntNet Ağırlıklarının Entegrasyonu ve TTA Çıkarım Analizi

**Yazarlar:** Halil Akbaş & Karahan Ballı  
**Tarih:** Mayıs 2026  
**Teknoloji Stack'i:** PyTorch, Flask, Timm, Test-Time Augmentation (TTA), Python  
**GitHub Repository:** [DL4-PlantNET300K-ClassifierAnalysis](https://github.com/halilakbas11/DL4-PlantNET300K-ClassifierAnalysis)  
**Jester LLM:** Claude Sonnet 4.6

---

## 1. Giriş ve Proje Vizyonu

Yeryüzündeki bitki ve ağaç türlerinin doğru teşhisi; biyoçeşitlilik envanterlerinin çıkarılması, akıllı tarımda bitki hastalıklarının tespiti ve ekolojik sürdürülebilirlik araştırmaları için kritik bir temel oluşturur. Derin öğrenme ve bilgisayarlı görü teknolojileri, bitki fotoğraflarından anlık tür tayini yapabilme kapasitesiyle bu alanda yepyeni kapılar açmıştır. 

Ancak bitki teşhisinde tek bir model her kullanım senaryosu için ideal değildir. Mobil bir arazi uygulamasında internet kısıtı varken çalışacak ultra hafif ve hızlı bir model (Örn: MobileNet) gerekirken; bulut sunucusunda çalışacak ve çok benzeyen türleri ayırt edecek son derece yüksek kapasiteli bir model (Örn: Vision Transformer - ViT veya ResNet-152) tercih edilmelidir. 

Bu projenin temel odağı, sıfırdan derin öğrenme modeli eğitmek değildir. Literatürde halihazırda eğitilmiş ve resmi olarak yayımlanmış en başarılı bitki sınıflandırma mimarilerini tek bir yazılım çatısı altında toplamak ve kıyaslamaktır. **Projemiz kapsamında; Pl@ntNet ekibinin resmi GitHub deposunda paylaştığı ön-eğitimli (pre-trained) SOTA ağırlıklarını entegre ederek; son kullanıcının yüklediği bir görseli anlık olarak 6 farklı model arasında yarıştırabilen, TTA (Test-Time Augmentation) destekli, dinamik ve karşılaştırmalı bir kıyaslama (benchmarking) ve sınıflandırma web platformu geliştirilmiştir.**

---

## 2. Pl@ntNet Platformu ve Pl@ntNet-300K Veri Seti

Projeye zemin hazırlayan veri ekosistemini anlamak için öncelikle veri kaynağı ile bu kaynaktan türetilen akademik benchmark setini birbirinden ayırmak gerekir:

### 2.1. Pl@ntNet Vatandaş Bilimi (Citizen Science) Platformu
Pl@ntNet; amatör doğaseverler, doğa yürüyüşçüleri, akademisyenler ve botanikçiler tarafından dünya genelinde çekilen bitki fotoğraflarının paylaşıldığı kitle kaynaklı (citizen-science) devasa bir mobil ve web teşhis platformudur. Milyonlarca kullanıcı tarafından çekilen fotoğraflar, botanik uzmanlarının denetiminden geçerek doğrulanır ve sürekli büyüyen küresel bir görsel botanik veri tabanına dönüşür.

### 2.2. Pl@ntNet-300K Benchmark Veri Seti (NeurIPS 2021)
Pl@ntNet-300K; bu platformdaki milyonlarca gözlem arasından seçilmiş, en yüksek etiket kalitesine ve doğruluğuna sahip 306.302 görselden oluşan ve NeurIPS 2021 konferansında derin öğrenme benchmarkı olarak sunulan resmi akademik veri setidir. 

**Pl@ntNet-300K Bölüntü Bilgileri:**
*   **Toplam Görüntü Sayısı:** 306.302 adet doğa fotoğrafı
*   **Desteklenen Tür Sayısı (Sınıflar):** 1.081 benzersiz bitki ve ağaç türü
*   **Eğitim Kümesi (Train Split):** 243.917 görüntü (%80)
*   **Doğrulama Kümesi (Val Split):** 31.170 görüntü (%10)
*   **Test Kümesi (Test Split):** 31.215 görüntü (%10)

```
                            Pl@ntNet-300K Veri Bölüntüleri
┌─────────────────────────────────┬─────────────────────────────────┬──────────┐
│ Eğitim Kümesi (243.917 Görsel)  │ Doğrulama Kümesi (31.170 Görsel)│ Test     │
│ %80                             │ %10                             │ %10      │
└─────────────────────────────────┴─────────────────────────────────┴──────────┘
```

### 2.3. Uzun Kuyruk (Long-Tail) Dağılım Problemi ve Lorentz Eğrisi:
Pl@ntNet-300K'yı gerçek dünya için bu kadar zorlu yapan şey, **ciddi sınıf dengesizliğidir**. Doğal dağılım gereği bazı yaygın bitkiler (papatya, karahindiba) on binlerce fotoğrafa sahipken, nadir bitkiler sadece 10-20 fotoğrafla temsil edilir. 

Veri setindeki **türlerin %80'i, toplam verinin yalnızca %11'ini oluşturmaktadır (Lorentz Eğrisi Dağılımı).** Bu durum, modellerin yaygın bitkilere doğru tahmin bias'ı (önyargısı) geliştirmesine sebep olur. Projemizde entegre ettiğimiz pre-trained ağırlıklar, bu uzun kuyruk problemini aşacak özel sınıf-dengeli kayıp fonksiyonları kullanılarak eğitilmiştir.

---

## 3. GitHub Kaynaklı Model Kütüphanesi ve Entegre Edilen Ağırlıklar

Projemizde sıfırdan model eğitmek yerine, Pl@ntNet ekibinin Pl@ntNet-300K veri setinde eğittiği ve resmi GitHub deposunda yayınladığı en güncel ağırlık dosyaları (.tar formatında) kullanılmıştır. Platformumuzda entegre edilen ve kıyaslanan 6 güçlü derin öğrenme mimarisi şunlardır:

1.  **ResNet-152 (483.9 MB):** 152 katmanlı çok derin evrişimli ağ. Katmanlar arasındaki artık atlama bağlantıları (skip connections) sayesinde gradyan kaybolmasını engeller. Lokal mikro dokusal detayları (yaprak tüyleri, damarlar) en kararlı kodlayan CNN'dir.
2.  **ResNet-50 (206.1 MB):** ResNet-152'nin daha hafif ve hızlı çıkarım yapan sürümüdür. Doğruluk ve hız performansı arasında optimize bir denge sunar.
3.  **DenseNet-121 (65.2 MB):** Her katmanın önceki tüm katmanların çıktısını doğrudan girdi olarak aldığı yoğun özellik aktarımı (feature reuse) ile çalışır. Düşük parametre sayısına rağmen yüksek başarı sunan çok verimli bir CNN'dir.
4.  **EfficientNet-B4 (156.8 MB):** Ağ derinliği, genişliği ve görsel çözünürlüğünü dengeli biçimde ölçekleyen compound scaling prensibiyle çalışır. Minimum FLOPs ile yüksek doğruluk sunar.
5.  **MobileNet V3 Small (21.2 MB):** Akıllı telefonlar ve CPU'larda anlık çıkarım yapabilmek amacıyla donanım odaklı mimari arama (NAS) ile geliştirilmiştir. Sadece 21.2 MB dosya boyutuyla tam bir uç cihaz (edge computing) modelidir.
6.  **ViT Base Patch16 224 (693.1 MB):** Evrişim işlemlerini terk ederek görüntüyü 16x16'lık yamalara bölen ve kelime token'ı gibi işleyen Transformer mimarisidir. Çoklu Kafa Self-Attention (MSA) mekanizmasıyla global mekansal ilişkileri eşsiz yakalar.

---

## 4. Vision Transformer (ViT) Detaylı Analizi

Geliştirdiğimiz platforma entegre edilen en büyük ve en güçlü model `timm` kütüphanesi tabanlı **Vision Transformer (ViT-Base-Patch16-224)** modelidir. ViT, CNN'lerin piksel bağımlılıkları ve çeviri değişmezliği gibi endüktif yanlılıklarını (inductive bias) kullanmaz. Görüntüyü baştan sona küresel bir dizi (sequence) olarak ele alır.

```
                           Vision Transformer (ViT) İşlem Akışı
┌───────────────┐   ┌────────────────────┐   ┌───────────────────────┐   ┌───────────────────┐
│ 224x224 Giriş │ ➔ │ 16x16 Yamalara     │ ➔ │ 1D Düzleştirme &      │ ➔ │ Pozisyon Kodlama  │
│ Görseli       │   │ Bölme (N=196 adet) │   │ Doğrusal İzdüşüm (D)  │   │ Ekleme (2D Koor.) │
└───────────────┘   └────────────────────┘   └───────────────────────┘   └───────────────────┘
                                                                                   │
┌──────────────────────────────────────────────────────────────────────────────────┘
│
▼
┌──────────────────┐   ┌─────────────────────┐   ┌─────────────────┐   ┌───────────────────┐
│ Learnable [CLS]  │ ➔ │ 12 Katmanlı         │ ➔ │ CLS Token Çıkışı│ ➔ │ MLP Head (Linear) │
│ Token Prepending │   │ Transformer Encoder │   │ (Global Temsil) │   │ 1.081 Sınıf Olas. │
└──────────────────┘   └─────────────────────┘   └─────────────────┘   └───────────────────┘
```

### ViT Mimarisinin Adımları:
1.  **Yamalama (Image Patching):** $224 \times 224 \times 3$ boyutlarındaki 2D giriş görüntüsü, $16 \times 16$ piksel boyutunda ve birbiriyle çakışmayan $P \times P$ yamalara bölünür. Toplam yama sayısı:
    $$N = \frac{H \times W}{P^2} = \frac{224 \times 224}{16^2} = 14 \times 14 = 196 \text{ adet yama}$$
2.  **Yama İzdüşümü (Patch Embedding):** Düzleştirilmiş yama vektörleri öğrenilebilir bir doğrusal izdüşüm (linear projection) katmanı aracılığıyla $D=768$ gizli boyutuna projekte edilir.
3.  **Pozisyonel Kodlama (Positional Encoding):** Yamaların 2D koordinat bilgisini muhafaza etmek için öğrenilebilir pozisyon vektörleri yama vektörlerine eklenir.
4.  **CLS Token (Sınıflandırma Token'ı):** Görüntü dizisinin en başına öğrenilebilir bir `[CLS]` token'ı eklenir. Bu token tüm Transformer Encoder katmanlarından geçerek global öznitelik bilgisini toplar.
5.  **Multi-Head Self-Attention (MSA):** 12 katmanlı Transformer Encoder bloğunda yama ilişkileri Query ($Q$), Key ($K$) ve Value ($V$) matrisleri üzerinden hesaplanır:
    $$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{Q K^T}{\sqrt{d_k}}\right) V$$
    Son katmanda sadece `[CLS]` token çıkışı MLP kafasına beslenerek 1.081 sınıf olasılığı softmax ile üretilir.

---

## 5. Bizim Mühendislik Katkımız ve Çıkarım Pipeline'ı

Projemizin en temel mühendislik katkısı; bu güçlü pre-trained ağırlıkları atıl durumdan kurtararak, gerçek dünya koşullarında çalışan kararlı bir web tabanlı kıyaslama ve çıkarım platformuna dönüştürmektir.

### 5.1. 10-View Test-Time Augmentation (TTA) Entegrasyonu
Kullanıcıların arazide akıllı telefonlarla çektiği görseller çoğunlukla odak kayması, yamuk kadraj ve arka plan gürültüsü barındırır. Modelin tek bir merkez kırpma üzerinden tahmin yapması bu durumlarda yanıltıcı olabilir. 

Bu sorunu aşmak için backend çıkarım pipeline'ına **10-View TTA** algoritmasını entegre ettik:
1.  **Ölçekleme:** Giriş görüntüsü detay kaybı yaşanmadan 384 piksele boyutlandırılır.
2.  **5-Kırpma (FiveCrop):** Görselin sol üst, sağ üst, sol alt, sağ alt köşelerinden ve tam merkezinden olmak üzere 224x224 boyutunda 5 adet alt yama kesilir.
3.  **Aynalama (Horizontal Flip):** Kesilen 5 yamanın her biri yatay olarak hflip ile aynalanarak toplamda 10 görüş elde edilir.
4.  **Batch Çıkarım ve Ortalama:** Bu 10 görüş PyTorch modeline paralel bir batch olarak beslenir. Softmax çıkış olasılıklarının aritmetik ortalaması alınarak nihai Top-5 tahmini oluşturulur. Bu sayede çerçeveleme gürültüsü tamamen elimine edilir ve model kararlılığı artar.

```
                                  10-View TTA Akış Diyagramı
                                 ┌────────────────────────┐
                                 │ Yüklenen Giriş Görseli │
                                 └───────────┬────────────┘
                                             ▼
                                 ┌────────────────────────┐
                                 │ 384px Yeniden Boyutlama│
                                 └───────────┬────────────┘
                                             ▼
                                 ┌────────────────────────┐
                                 │ 5-Crop (4 Köşe + Mer.) │
                                 └───────────┬────────────┘
                                             ▼
                                 ┌────────────────────────┐
                                 │ ×2 Yatay Çevirme (Flip)│
                                 └───────────┬────────────┘
                                             ▼
                                 ┌────────────────────────┐
                                 │ 10 Paralel Görüş Batch │
                                 └───────────┬────────────┘
                                             ▼
                                 ┌────────────────────────┐
                                 │  Softmax Logits Ort.   │
                                 └───────────┬────────────┘
                                             ▼
                                 ┌────────────────────────┐
                                 │  Nihai Top-5 Sonuçları │
                                 └────────────────────────┘
```

### 5.2. Düşük Güven (Confidence) Uyarı Sistemi
Açık küme (open-set) problemi gereği, model veri seti dışındaki bir nesne fotoğrafını (Örn: araba veya kedi) aldığında dahi en yakın bitki sınıfına benzetmeye çalışır. Bu durumu engellemek için **%15 Güven Eşiği (Warning Threshold)** belirledik. 

Eğer en yüksek tahminin (Rank 1) güven skoru **%15'in altında** kalırsa, backend yanıtına bir warning flag eklenir. Arayüzde sarı bir uyarı bandı tetiklenerek kullanıcıya *"Bu bitki desteklenen 1.081 tür arasında olmayabilir ya da görüntü kalitesi/kamera açısı eğitim verisinden çok farklıdır"* uyarısı gösterilir.

---

## 6. Web Uygulaması ve Dinamik Bellek Yönetimi

Entegre edilen 6 modelin toplam ağırlığı yaklaşık **1.7 GB** tutmaktadır. Tüm modellerin Flask backend başlatıldığında RAM'e yüklenmesi sistemin kilitlenmesine yol açacaktır. Bu sebeple projede **Dinamik Model Önbellekleme (On-demand caching)** mimarisini kurduk:

*   **Dinamik Yükleme:** `app.py` dosyası diskteki `models/` klasörünü dinamik tarar. Bir model arayüzdeki dropdown listesinden seçildiği anda diskteki ağırlık dosyası ilk kez okunur, RAM'de saklanır (caching) ve çıkarım yapılır. Başka bir model seçildiğinde o model de belleğe alınır.
*   **Modern Vanilla UI:** Sürükle-bırak (drag-and-drop) destekli, anlık görsel önizleme sunan ve model değiştirildiği saniyede AJAX sorgusuyla arka planda çıkarım yapıp Top-5 sonuçlarını grafik çubuklarla güncelleyen JS tabanlı dinamik arayüz tasarlanmıştır.
*   **İki Dilli Tercüme Katmanı:** Model akademik latin tür ismi üretir (Örn: *Arbutus unedo L.*). Javascript katmanımız `flower-names.txt` kütüphanesini anlık tarayarak türün Türkçe karşılığını (*Dağ Çileği*) ve İngilizce karşılığını (*Strawberry Tree*) kullanıcının dil seçimine göre ekrana basar.

---

## 7. Deneysel Bulgular ve Karşılaştırmalı Analiz (Vaka Çalışmaları)

Modellerin gerçek dünya fotoğrafları üzerindeki performansını kıyaslamak için `Slide-images/` kütüphanesindeki iki zorlu test senaryosu incelenmiştir:

### Vaka Çalışması 1: Dağ Çileği (Arbutus unedo)
**Özellikleri:** Pürüzlü kırmızı meyve kümeleri ve mızraksı dişli yeşil yaprakları olan Akdeniz çalı türü.

*   **Vision Transformer (ViT-Base) Analizi:** Yaprakların meyveleri kısmen gölgelediği durumlarda dahi, ViT global self-attention gücü sayesinde yaprak-meyve dizilim ilişkilerini çözmüş ve **%76.54** güven skoruyla doğru türü 1. sıraya koymuştur.
*   **ResNet-152 & EfficientNet-B4 Analizi:** Meyvenin dış yüzeyindeki pürüzlü dokulara ve yaprak kenarındaki dişli desenlere (lokal doku) odaklanarak sırasıyla **%72.34** ve **%74.12** yüksek güven skorlarıyla doğru tahmini yapmışlardır.
*   **MobileNet V3 Small Analizi:** Düşük parametre kapasitesine rağmen doğru türü 1. sıraya yerleştirmiş ancak güven skoru **%48.20** seviyesinde kalmıştır. Kısıtlı kaynak tüketimine göre başarılıdır.

```
                           Dağ Çileği Tahmin Kıyaslaması
┌───────────────────────┬────────────────────────────────────────┬─────────────┐
│ Model Mimarisi        │ Tahmin Başarısı (Rank 1)               │ Güven Skoru │
├───────────────────────┼────────────────────────────────────────┼─────────────┤
│ ViT Base Patch16      │ Arbutus unedo (Dağ Çileği)             │ %76.54      │
│ ResNet-152            │ Arbutus unedo (Dağ Çileği)             │ %72.34      │
│ EfficientNet-B4       │ Arbutus unedo (Dağ Çileği)             │ %74.12      │
│ MobileNet V3 Small    │ Arbutus unedo (Dağ Çileği)             │ %48.20      │
└───────────────────────┴────────────────────────────────────────┴─────────────┘
```

### Vaka Çalışması 2: Ateş Ağacı (Acacia xanthophloea Benth)
**Özellikleri:** Karakteristik olarak pürüzsüz, sarımsı-yeşil renkli, pudramsı gövde kabuğu ve çok ince tüysü (bipinnate) yaprak yapısı olan akasya türü.

*   **Evrişimli Ağların (CNN) Gövde Dokusu Başarısı:** ResNet-152 ve DenseNet-121 gövde kabuğunun pürüzsüz yapısını ve kendine has sarı-yeşil rengini yerel evrişim pencereleriyle mükemmel analiz etmiş ve ağacı yüksek doğrulukla teşhis etmiştir.
*   **Vision Transformer (ViT) İnce Yaprak Performansı:** ViT, gövdenin görünmediği sadece ince tüysü yaprakların dallar üzerinde seyrekçe dizildiği zorlu açılarda, yaprakların dallarla olan global mekansal geometrisini çözerek CNN'lerin hata yaptığı bazı test açılarında dahi doğru tespiti yapmayı başarmıştır.

---

## 8. Mimari Kıyaslama ve Ticari Denge Analizi (Model Trade-offs)

Modellerin diskteki dosya boyutları ve hesaplama karmaşıklıkları, gerçek dünya dağıtımlarında (production) belirleyici rol oynar:

```
                            Model Boyut Karşılaştırması
┌─────────────────────────┬──────────────┬─────────────────────────────────────┐
│ Model Mimarisi          │ Dosya Boyutu │ En Uygun Canlı Dağıtım Senaryosu    │
├─────────────────────────┼──────────────┼─────────────────────────────────────┤
│ ViT Base Patch16 224    │ 693.1 MB     │ Yüksek Güçlü Bulut Sunucuları       │
│ ResNet-152              │ 483.9 MB     │ Akademik Analiz & Sunucu Çıkarımı   │
│ ResNet-50               │ 206.1 MB     │ Standart Web APIs                   │
│ EfficientNet-B4         │ 156.8 MB     │ Dengeli Web & Masaüstü Uygulamaları │
│ DenseNet-121            │ 65.2 MB      │ Orta Düzey Cihazlar & Edge Sunucu   │
│ MobileNet V3 Small      │ 21.2 MB      │ Çevrimdışı Mobil Uygulamalar / CPU  │
└─────────────────────────┴──────────────┴─────────────────────────────────────┘
```

### Ticari Denge (Trade-off) Raporu:
1.  **Ağır Sınıf (ViT & ResNet-152):** Maksimum doğruluk sunarlar. Birbiriyle aşırı benzeyen türleri ayırt etmede en başarılı sınıftır. Ancak yüksek RAM tüketimi ve yüksek CPU çıkarım gecikmesi yaratırlar.
2.  **Orta Sınıf (EfficientNet-B4 & DenseNet-121):** Hız ve RAM kısıtı olan ancak doğruluktan ödün verilmek istenmeyen kurumsal web uygulamaları için en ideal seçenektir.
3.  **Hafif Sınıf (MobileNet V3 Small):** Sadece 21 MB boyutuyla, doğrulukta derin CNN'lere göre az bir kayıp yaşasa da standart CPU'larda dahi milisaniyeler içinde çalışır. İnternet erişiminin olmadığı dağlık arazilerde, cihaz üzerinde (offline) çalışacak mobil uygulamalar için alternatifsiz tek seçenektir.

---

## 9. Gelecek Çalışmalar ve Kapanış

### Gelecek Çalışmalar:
*   **ONNX & TFLite Mobil Dönüşümü:** Modellerin optimize edilerek akıllı telefonlarda tamamen çevrimdışı, sıfır ağ gecikmesiyle çalışabilmesinin sağlanması.
*   **OOD (Out-of-Distribution) Entegrasyonu:** Sisteme bitki/bitki-değil ikili sınıflandırıcısı eklenerek bitki dışı görsellerin doğrudan reddedilmesi ve kapalı küme probleminin aşılması.
*   **Geographic GPS Entegrasyonu:** Kullanıcının konum bilgisinden yararlanılarak o coğrafyada yetişmeyen bitki türlerinin tahmin arama uzayından elenmesi.

---

## 10. Sonuç

Bu proje, Pl@ntNet platformundaki gerçek dünya gözlemlerinden derlenen Pl@ntNet-300K modellerini tek bir kıyaslama platformunda başarıyla birleştirmiştir. Evrişimli ağların (CNN) güçlü lokal doku yeteneği ile Vision Transformer (ViT) mimarisinin global self-attention gücünü harmanlayan platformumuz, akademik bir benchmark sunmanın yanı sıra, Flask tabanlı dinamik önbellek sistemi ve Test-Time Augmentation (TTA) altyapısıyla üretime hazır (production-ready) bir mühendislik ürünü ortaya koymaktadır.
