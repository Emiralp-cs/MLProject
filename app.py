import joblib
import pandas as pd
import numpy as np
from flask import Flask, render_template, request

# ======================================================================
# 1. MODELLERİ GÜVENLİ YÜKLEME
# ======================================================================

model = None
preprocessor = None
all_input_features = [] 

print("🔄 Başlatılıyor...")

try:
    # Modelleri yükle
    print("   -> Model dosyaları okunuyor...")
    model = joblib.load('final_model.pkl')
    preprocessor = joblib.load('preprocessor.pkl')
    
    # Özellik listesini yükle ve all_input_features değişkenine ata
    all_input_features = joblib.load('final_features.pkl')
    
    print("✅ Modeller ve özellik listesi başarıyla yüklendi.")
    print(f"   -> Beklenen özellik sayısı: {len(all_input_features)}")

except ImportError as e:
    print(f"❌ KÜTÜPHANE HATASI: {e}")
    print("Lütfen 'scikit-learn' kütüphanesini kurduğunuzdan emin olun.")
except FileNotFoundError as e:
    print(f"❌ DOSYA HATASI: {e}")
    print("Lütfen .pkl dosyalarının (final_model, preprocessor, final_features) app.py ile aynı klasörde olduğundan emin olun.")
except Exception as e:
    print(f"❌ YÜKLEME HATASI: {e}")

# ======================================================================
# 2. FLASK UYGULAMASI
# ======================================================================

app = Flask(__name__)

# DİKKAT: Hatalı olan 'all_input_features = final_features' satırı BURADAN SİLİNDİ.

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction_result = None
    
    # Model yüklenmediyse uyarı ver
    if not all_input_features or model is None:
        return render_template('index.html', features=[], result={
            'status': "SİSTEM HATASI",
            'error': "Model dosyaları yüklenemediği için işlem yapılamıyor. Terminali kontrol edin.",
            'risk_level': "danger"
        })

    if request.method == 'POST':
        try:
            # 1. Formdan verileri al
            form_data = request.form.to_dict()
            
            # 2. Veri Setine Uygun DataFrame Oluşturma
            data_row = {}
            
            # 'german.data' yapısına ve eğitim kodunuza göre SADECE BU 7 KOLON SAYISALDIR
            numeric_columns = [
                'Duration', 
                'CreditAmount', 
                'InstallmentRate', 
                'PresentResidenceSince', 
                'Age', 
                'NumberExistingCredits', 
                'NumberPeopleMaintenance'
            ]

            for feature in all_input_features:
                value = form_data.get(feature, None)

                if feature in numeric_columns:
                    # Sayısal dönüşüm (Boş gelirse 0 veya ortalama yerine NaN atıyoruz, model halleder veya hata verir)
                    if value and str(value).strip() != '':
                        try:
                            data_row[feature] = float(value)
                        except ValueError:
                            data_row[feature] = np.nan # Sayı değilse boş geç
                    else:
                        data_row[feature] = np.nan
                else:
                    # Kategorik dönüşüm (String)
                    data_row[feature] = str(value) if value else ''

            # DataFrame oluştur
            input_df = pd.DataFrame([data_row], columns=all_input_features)

            # 3. Ön İşleme (Preprocessing)
            # Not: preprocessor.transform() sadece eğitilmiş kolonları dönüştürür
            processed_data = preprocessor.transform(input_df)

            # 4. Tahmin Yap
            prediction = model.predict(processed_data)[0]
            proba = model.predict_proba(processed_data)[:, 1][0]

            # 5. Sonucu Formatla
            if prediction == 1:
                status = "✅ KREDİ ONAYLANDI (DÜŞÜK RİSK)"
                style = "success"
            else:
                status = "❌ KREDİ REDDEDİLDİ (YÜKSEK RİSK)"
                style = "danger"

            prediction_result = {
                'status': status,
                'probability': f"%{proba * 100:.2f}",
                'risk_level': style
            }

        except Exception as e:
            print(f"Tahmin Hatası Detayı: {e}")
            prediction_result = {
                'status': "İŞLEM HATASI",
                'error': f"Tahmin sırasında bir hata oluştu: {str(e)}",
                'risk_level': "warning"
            }

    return render_template('index.html', features=all_input_features, result=prediction_result)

if __name__ == '__main__':
    app.run(debug=True)