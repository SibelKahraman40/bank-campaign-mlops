# Banka Pazarlama Kampanyası Dönüşüm Tahmini

Bu projede müşteri özelliklerinden banka pazarlama kampanyası dönüşümünü (ikili sınıflandırma) tahmin eden ve veri kaymasını (drift) izleyen bir ML hattı kurdum. Lojistik regresyonu başlangıç modeli olarak kullandım ve Random Forest ile karşılaştırdım. Son model 120 ağaç ve `min_samples_leaf=20` yapılandırmasından oluşuyor.

## Veri ve yöntem

- Canlı dönem öncesinde sonuçları kesinleşen geçmiş kayıtları birleştirerek eğitim verisini (`X_gecmis`, `y_gecmis`) oluşturup genişlettim.
- Sayısal özniteliklere `StandardScaler`, kategorik özniteliklere `OneHotEncoder(handle_unknown="ignore")` uygulayarak `ColumnTransformer` ön işlemcisi kurdum.
- Etkileşim öznitelikleri türettim: `job` ve `education` sütunlarını birleştirerek `job_education`, `contact` ve `poutcome` sütunlarını birleştirerek `contact_outcome` özniteliklerini oluşturdum. `campaign` arama sayılarını ise `[0, 1, 2, 4, inf]` aralıklarına ayırarak binned kategorilere dönüştürdüm.
- Modelin canlı dönemdeki veri kayması (drift) ve genelleştirme yeteneğini ölçmek için canlı etiketleri (`y_live`) yalnızca geriye dönük nihai değerlendirme aşamasında kullandım.

Veri dosyalarını `data/` klasörüne yerleştirip `lesson.ipynb` dosyasını sırayla çalıştırın.

## Sonuçlar

| Model | Canlı ROC-AUC | Canlı Average Precision (PR-AUC) |
|---|---:|---:|
| Lojistik regresyon (Baseline) | 0,563 | - |
| Lojistik regresyon + Sinir ağı ensemble | 0,539 | - |
| **Random Forest + genişletilmiş veri + etkileşimler** | **0,689** | **0,576** |

Lojistik regresyon ile sinir ağının skorlarını birleştirmek test ROC-AUC’sini 0,661’den 0,682’ye çıkardı. Ancak canlı dönemde lojistik regresyon 0,563, birleşik model 0,539 verdi. Testteki iyileşme son dönemde tekrarlanmadığı için lojistik regresyonu bir süre ana model olarak korudum.

En büyük iyileşmeyi, canlı dönemden önce sonuçları belli olan kayıtları eğitime ekleyerek elde ettim. Canlı dönem öncesi verilerin tamamını eğitime dahil ederek elde ettiğim genişletilmiş veri setinde, Random Forest modeli lojistik regresyona kıyasla daha yüksek ROC-AUC ve Average Precision performansına ulaştı. 

Canlı dönem etiketleri yalnızca değerlendirme adımı için saklanmış olup, geriye dönük karşılaştırma amacıyla **Canlı ROC-AUC skoru 0,689**, **Canlı Average Precision skoru ise 0,576** olarak raporlanmıştır.

## Docker ile çalıştırma

Docker Desktop'ı açın. Windows PowerShell'de notebook ile veri dosyalarının bulunduğu klasöre girip şu komutu çalıştırın:

```powershell
docker run --name banka-jupyter --rm -it -p 127.0.0.1:8890:8890 --mount "type=bind,source=$($PWD.Path),target=/work" -w /work python:3.10-slim sh -c "pip install notebook pandas scikit-learn joblib matplotlib && jupyter notebook --ip=0.0.0.0 --port=8890 --no-browser --allow-root"