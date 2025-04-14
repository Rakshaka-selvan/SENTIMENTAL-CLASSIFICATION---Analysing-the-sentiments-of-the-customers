from tensorflow.keras.datasets import imdb

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding #layers are the main foundation
#embeddings would be the first layer in neural networks which is to hv words with similar meanings together
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.preprocessing import sequence
from numpy import array
import keras
import json


#data is loaded here into training and testing data , to avoid overfitting we use different datas to test and train
#x is positive and y is negative
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=5000)

#preprocessing the data
word_to_id = keras.datasets.imdb.get_word_index()
word_to_id = {k:(v+3) for k,v in word_to_id.items()}
word_to_id["<PAD>"] = 0
word_to_id["<START>"] = 1
word_to_id["<UNK>"] = 2

x_train = sequence.pad_sequences(x_train, maxlen=300)
x_test = sequence.pad_sequences(x_test, maxlen=300)


network = Sequential()
network.add(Embedding(5000, 32, input_length=300)) #with this add we could add neural network layers to our model
network.add(Flatten()) #flattens our input
network.add(Dense(1, activation='sigmoid'))
network.compile(loss="binary_crossentropy", optimizer='Adam', metrics=['accuracy'])

#fitting or training the model
network.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=3, batch_size=64)

network.save('model.h5')

with open("word_index.json", "w") as f:
    json.dump(word_to_id, f)

print("Model and word index saved!")
result = network.evaluate(x_test, y_test, verbose=0)

negative = "this movie was bad"
positive = "i had fun"
negative2 = "this movie was terrible"
positive2 = "i really liked the movie"

for review in [positive, positive2, negative, negative2]:
    temp = []
    for word in review.split(" "):
        temp.append(word_to_id[word])
    temp_padded = sequence.pad_sequences([temp], maxlen=300)
    print(review + " -- Sent -- " + str(network.predict(array([temp_padded][0]))[0][0]))


