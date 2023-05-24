import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

# veriyi yükleme kısmı
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'Class']
dataset = pd.read_csv("iris.data", names=names)

# veriyi yükledikten sonra etiketlere ve özelliklere bölüyor.
X = dataset.iloc[:, 0:4].values
y = dataset.iloc[:, 4].values

# %20 luk test ve %80 lik eğitim modeli alındı.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# test ve eğitim transform fonksiyonuna sokuluyor.
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# LDA modeki tanımlanır ve ilk model ile transform edilmiş model tekrar transforma uğratılıyor.
lda = LDA(n_components=1)
X_train = lda.fit_transform(X_train, y_train)
X_test = lda.transform(X_test)

# performans ölçümleri
classifier = RandomForestClassifier(max_depth=2, random_state=0)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)

cm = confusion_matrix(y_test, y_pred)
print(cm)
print('Accuracy' + str(accuracy_score(y_test, y_pred)))