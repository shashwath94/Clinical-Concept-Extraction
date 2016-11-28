import os
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

texts = []  # list of text samples
labels_index = {}  # dictionary mapping label name to numeric id
labels = []  # list of label ids
for name in sorted(os.listdir('data/train/txt/')):
    fpath = os.path.join('data/train/txt/', name)
    
    
    
    
    f = open(fpath)
    texts.append(f.read())
    f.close()
    

print('Found %s texts.' % len(texts))


tokenizer = Tokenizer(nb_words=MAX_NB_WORDS)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)

word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))
