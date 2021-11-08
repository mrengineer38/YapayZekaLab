# Değerlendirme Soruları
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import losses
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
## 1) Verilen kodu inceleyerek kendi modelinizi oluşturunuz
model = Sequential()
model.add(Dense(16, input_dim=20, activation="relu"))
model.add(Dense(12, activation="relu"))
model.add(Dense(8, activation="relu"))
model.add(Dense(4, activation="softmax"))
model.summary()
## 2) Verilen kodu inceleyerek çapraz doğrulama işlemi yapınız ve buna göre başarımı değerlendiriniz
### 2.1) Verilerin hazırlanması
data = pd.read_csv("telefon_fiyat_degisimi.csv")

label_encoder = LabelEncoder().fit(data.price_range)
labels = label_encoder.transform(data.price_range)

x = data.drop(["price_range"], axis=1)
y = labels
sc = StandardScaler()
x = sc.fit_transform(x)

train_data, test_data, train_target, test_target = train_test_split(
    x, y, test_size=0.2)
train_target = to_categorical(train_target)
test_target = to_categorical(test_target)
### 2.2) Çapraz doğrulama
k = 4
num_val_samples = len(train_data) // k
num_epochs = 150
all_scores = []

model.compile(loss="categorical_crossentropy",
                  optimizer="adam",
                  metrics=["accuracy"])

for i in range(k):
    print('processing fold  #', i)
    val_data = train_data[i * num_val_samples: (i + 1) * num_val_samples]
    val_target = train_target[i * num_val_samples: (i + 1) * num_val_samples]

    partial_train_data = np.concatenate([train_data[:i * num_val_samples],
                                         train_data[(i + 1) * num_val_samples:]], axis=0)

    partial_train_target = np.concatenate([train_target[:i * num_val_samples],
                                           train_target[(i + 1) * num_val_samples:]], axis=0)

    model.fit(partial_train_data, partial_train_target,
              epochs=num_epochs, batch_size=1, verbose=0)

    val_categorical_crossentropy, val_accuracy = model.evaluate(
        val_data, val_target, verbose=0)

    all_scores.append(val_accuracy)

all_scores
## 3) Verisetinde (telefon_fiyat_degisimi) toplam 20 adet özellik bulunmaktadır. Bu verisetinden "blue", "fc", "int_memory", "ram" ve "wifi" değerlerini çıkarıp, sınıflandırma işlemini tekrar yapınız
### 3.1) Veriyi hazırlama
data = pd.read_csv("telefon_fiyat_degisimi.csv")
data = data.drop(["blue", "fc", "int_memory", "ram", "wifi"], axis=1)
label_encoder = LabelEncoder().fit(data.price_range)
labels = label_encoder.transform(data.price_range)

x = data.drop(["price_range"], axis=1)
y = labels
sc = StandardScaler()
x = sc.fit_transform(x)

train_data, test_data, train_target, test_target = train_test_split(
    x, y, test_size=0.2)

train_target = to_categorical(train_target)
test_target = to_categorical(test_target)

### 3.2) Model
model = Sequential()
model.add(Dense(12, input_dim=15, activation="relu"))
model.add(Dense(8, activation="relu"))
model.add(Dense(4, activation="softmax"))
model.summary()
### 3.3) Modeli derleme
model.compile(loss="categorical_crossentropy",
              optimizer="adam",
              metrics=["accuracy"])
### 3.4) Eğitim
model.fit(train_data, train_target, validation_data=(test_data, test_target), epochs=150, verbose=0)
### 3.5) Grafik gösterimi
plt.plot(model.history.history["accuracy"])
plt.plot(model.history.history["val_accuracy"])
plt.title("Model Başarımları")
plt.ylabel("Başarım")
plt.xlabel("Epok sayısı")
plt.legend(["Eğitim", "Test"], loc="upper left")
plt.show()

plt.plot(model.history.history["loss"])
plt.plot(model.history.history["val_loss"])
plt.title("Model Kayıpları")
plt.ylabel("Kayıp")
plt.xlabel("Epok sayısı")
plt.legend(["Eğitim", "Test"], loc="upper left")
plt.show()
## 4) Diyabet verisetini kullanarak bir YSA modeli oluşturunuz. Bu YSA modeline, eğitim ve doğrulama işlemlerinin başarım ve kayıplarını belirleyiniz. Bu değerleri bir grafik üzerinde gösteriniz
### 4.1) Verisetinin hazırlanması
data = pd.read_csv("diyabet.csv")
label_encoder = LabelEncoder().fit(data.output)
labels = label_encoder.transform(data.output)
data = data.drop(["output"], axis=1)
data = StandardScaler().fit_transform(data)
### 4.2) Eğitim ve test verilerinin hazırlanması
train_data, test_data, train_target, test_target = train_test_split(
    data, labels, test_size=0.2)
### 4.3) Modelin oluşturulması
model = Sequential()
model.add(Dense(8, input_dim=8, activation="relu"))
model.add(Dense(16, activation="relu"))
model.add(Dense(4, activation="relu"))
model.add(Dense(1, activation="sigmoid"))
model.summary()
### 4.4) Modelin derlenmesi
model.compile(loss=losses.binary_crossentropy,
              optimizer="adam",
              metrics=["accuracy"])
### 4.5) Modelin eğitilmesi
history = model.fit(train_data, train_target, validation_data=(
    test_data, test_target), epochs=110)

### 4.6) Eğitim ve doğrulama başarımlarının grafiği
# Başarım
plt.plot(model.history.history["accuracy"])
plt.plot(model.history.history["val_accuracy"])

plt.title("Model Başarımları")
plt.ylabel("Başarım")
plt.xlabel("Epok sayısı")
plt.legend(["Eğitim", "Test"], loc="upper left")
plt.show()

# Kayıp
plt.plot(model.history.history["loss"])
plt.plot(model.history.history["val_loss"])

plt.title("Model Kayıpları")
plt.ylabel("Kayıp")
plt.xlabel("Epok sayısı")
plt.legend(["Eğitim", "Test"], loc="upper left")
plt.show()
## 5) Soru 4'te oluşturduğunuz modelin ROC eğrisini çiziniz
target_pred = model.predict(test_data).ravel()
fpr, tpr, thresholds = roc_curve(test_target, target_pred)
auc_keras = auc(fpr, tpr)

plt.plot(fpr, tpr, marker='.')
plt.title("ROC")