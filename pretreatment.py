from tensorflow.keras.datasets import cifar10

# Charger le jeu de données
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Normaliser les données
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

# One-hot encoding des étiquettes
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)
