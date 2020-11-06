#%%
import numpy as np
import pandas as pd
import matplotlib.pylab as plt
import re
import time
import codecs
import os

# %%
# 데이터셋 로드
file_dir = os.getcwd() # 현재 파일 경로 추출
file_dir = os.path.dirname(file_dir) # 상위 경로 추출 - 코드 파일과 단어 리스트 파일 위치가 틀려서

train_data = pd.read_csv(file_dir + "/data/train_dataset_1007.csv") 
test_data = pd.read_csv(file_dir + "/data/test_dataset_1007.csv")
#%%
train_data['label'].value_counts().plot(kind='bar')
test_data['label'].value_counts().plot(kind='bar')
# %%
# 텍스트 전처리
from konlpy.tag import *

stopwords = ['의', '가', '이', '은', '들', '는', '좀', '잘', '걍', '과', '도', '를', '으로', '자', '에', '와', '한', '하다']

okt = Okt() 
X_train = [] 
for sentence in train_data['title']: 
    temp_X = [] 
    temp_X = okt.morphs(sentence, stem=True) # 토큰화 
    temp_X = [word for word in temp_X if not word in stopwords] # 불용어 제거 
    X_train.append(temp_X)

X_test = [] 
for sentence in test_data['title']: 
    temp_X = [] 
    temp_X = okt.morphs(sentence, stem=True) # 토큰화 
    temp_X = [word for word in temp_X if not word in stopwords] # 불용어 제거 
    X_test.append(temp_X)
#%%
# 토큰화 된 단어 정수 인코딩
from keras.preprocessing.text import Tokenizer 
max_words = 35000 
tokenizer = Tokenizer(num_words = max_words) 
tokenizer.fit_on_texts(X_train) 
X_train = tokenizer.texts_to_sequences(X_train) 
X_test = tokenizer.texts_to_sequences(X_test)
#%%
# 데이터 분포 확인
print("제목의 최대 길이 : ", max(len(l) for l in X_train)) 
print("제목의 평균 길이 : ", sum(map(len, X_train))/ len(X_train)) 
plt.hist([len(s) for s in X_train], bins=50) 
plt.xlabel('length of Data') 
plt.ylabel('number of Data') 
plt.show()
# %%
# 라벨값인 y를 원핫 인코딩 합니다.
y_train = [] 
y_test = [] 
for i in range(len(train_data['label'])): 
    if train_data['label'].iloc[i] == 1: 
        y_train.append([0, 0, 1]) 
    elif train_data['label'].iloc[i] == 0: 
        y_train.append([0, 1, 0]) 
    elif train_data['label'].iloc[i] == -1: 
        y_train.append([1, 0, 0]) 
        
for i in range(len(test_data['label'])): 
    if test_data['label'].iloc[i] == 1: 
        y_test.append([0, 0, 1]) 
    elif test_data['label'].iloc[i] == 0: 
        y_test.append([0, 1, 0]) 
    elif test_data['label'].iloc[i] == -1:
         y_test.append([1, 0, 0]) 
         
y_train = np.array(y_train) 
y_test = np.array(y_test)
# %%
from keras.layers import Embedding, Dense, LSTM 
from keras.models import Sequential 
from keras.preprocessing.sequence import pad_sequences 

max_len = max(len(l) for l in X_train) # 전체 데이터의 길이를 20로 맞춘다 
X_train = pad_sequences(X_train, maxlen=max_len) # pad_sequences를 활용하여 모든 데이터의 길이를 20으로 통일
X_test = pad_sequences(X_test, maxlen=max_len) # pad_sequences를 활용하여 모든 데이터의 길이를 20으로 통일

model = Sequential() 
model.add(Embedding(max_words, 100)) 
model.add(LSTM(128)) 
model.add(Dense(3, activation='softmax')) 
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy']) 
history = model.fit(X_train, y_train, epochs=10, batch_size=10, validation_split=0.1)

print("\n 테스트 정확도: {:.2f}%".format(model.evaluate(X_test, y_test)[1] * 100))
# %%
predict = model.predict(X_test)

predict_labels = np.argmax(predict, axis=1)
original_labels = np.argmax(y_test, axis=1)
for i in range(30):
    print("기사제목 : ", test_data['title'].iloc[i], "/\t 원래 라벨 : ", original_labels[i], "/\t예측한 라벨 : ", predict_labels[i])


# %%
def tokenize(doc):
    # norm은 정규화, stem은 근어로 표시하기를 나타냄
    return ['/'.join(t) for t in okt.pos(doc, norm=True, stem=True)]
#%%
import nltk
text = nltk.Text(tokens, name='NMSC')

selected_words = [f[0] for f in text.vocab().most_common(10000)]

def term_frequency(doc):
    return [doc.count(word) for word in selected_words]
#%%
def predict_pos_neg(review):
    token = tokenize(review)
    tf = term_frequency(token)
    data = np.expand_dims(np.asarray(tf).astype('float32'), axis=0)
    score = float(model.predict(data))
    if(score > 0.5):
        print("[{}]는 {:.2f}% 확률로 긍정 리뷰이지 않을까 추측해봅니다.^^\n".format(review, score * 100))
    else:
        print("[{}]는 {:.2f}% 확률로 부정 리뷰이지 않을까 추측해봅니다.^^;\n".format(review, (1 - score) * 100))
#%%

predict_pos_neg("올해 최고의 영화! 세 번 넘게 봐도 질리지가 않네요.")