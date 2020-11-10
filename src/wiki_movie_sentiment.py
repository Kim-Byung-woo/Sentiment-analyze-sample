#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import urllib.request
from konlpy.tag import Okt
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import os
# %%
# 데이터셋 로드
file_dir = os.getcwd() # 현재 파일 경로 추출
file_dir = os.path.dirname(file_dir) # 상위 경로 추출

train_data = pd.read_table(file_dir + "/data/ratings_train.txt")
test_data = pd.read_table(file_dir + "/data/ratings_test.txt")
#%%
# 데이터 개수 확인
print('훈련 데이터 리뷰 개수 :',len(train_data)) # 리뷰 개수 출력
print('테스트 데이터 리뷰 개수 :',len(test_data)) # 리뷰 개수 출력
# %%
# 텍스트 전처리
# 훈련용 리뷰 데이터 중 중복값 제거
train_data.drop_duplicates(subset=['document'], inplace=True) # document 열에서 중복인 내용이 있다면 중복 제거
# 훈련용 리뷰 데이터 중 한글과 공백을 제외하고 모두 제거
train_data['document'] = train_data['document'].str.replace("[^ㄱ-ㅎㅏ-ㅣ가-힣 ]","")
# 훈련용 리뷰 데이터 중 모두 제거된 데이터는 Nan값으로 대체
train_data['document'].replace('', np.nan, inplace=True)
# 훈련용 리뷰 데이터 중 Nan값 제거
train_data = train_data.dropna(how = 'any') # Null 값이 존재하는 행 제거
print(train_data.isnull().values.any()) # Null 값이 존재하는지 확인
print('전처리 후 훈련용 데이터 개수: ', len(train_data))

# 테스트용 리뷰 데이터 중 중복값 제거
test_data.drop_duplicates(subset=['document'], inplace=True) # document 열에서 중복인 내용이 있다면 중복 제거
#테스트용  리뷰 데이터 중 한글과 공백을 제외하고 모두 제거
test_data['document'] = test_data['document'].str.replace("[^ㄱ-ㅎㅏ-ㅣ가-힣 ]","")
#테스트용  리뷰 데이터 중 모두 제거된 데이터는 Nan값으로 대체
test_data['document'].replace('', np.nan, inplace=True)
#테스트용  리뷰 데이터 중 Nan값 제거
test_data = test_data.dropna(how = 'any') # Nan 값이 존재하는 행 제거
print(test_data.isnull().values.any()) # Nan 값이 존재하는지 확인
print('전처리 후 테스트용 데이터 개수: ', len(test_data))
#%%
# 토큰화
# 토큰(Token)이란 문법적으로 더 이상 나눌 수 없는 언어요소를 뜻합니다. 텍스트 토큰화(Text Tokenization)란 말뭉치(Corpus)로부터 토큰을 분리하는 작업을 뜻합니다.
from konlpy.tag import *

stopwords = ['의', '가', '이', '은', '들', '는', '좀', '잘', '걍', '과', '도', '를', '으로', '자', '에', '와', '한', '하다']

okt = Okt() 
X_train = [] 
for sentence in train_data['document']: 
    temp_X = [] 
    temp_X = okt.morphs(sentence, stem=True) # 토큰화 
    temp_X = [word for word in temp_X if not word in stopwords] # 불용어 제거 
    X_train.append(temp_X)

X_test = [] 
for sentence in test_data['document']: 
    temp_X = [] 
    temp_X = okt.morphs(sentence, stem=True) # 토큰화 
    temp_X = [word for word in temp_X if not word in stopwords] # 불용어 제거 
    X_test.append(temp_X)
#%%
# 토큰화 된 단어 정수 인코딩
tokenizer = Tokenizer()
tokenizer.fit_on_texts(X_train)
print(tokenizer.word_index) # 총 단어가 43000개 넘게 존재
#%%
# 단어 등장 빈도수가 3회 미만인 단어의 비중을 확인합니다.
threshold = 3
total_cnt = len(tokenizer.word_index) # 단어의 수
rare_cnt = 0 # 등장 빈도수가 threshold보다 작은 단어의 개수를 카운트
total_freq = 0 # 훈련 데이터의 전체 단어 빈도수 총 합
rare_freq = 0 # 등장 빈도수가 threshold보다 작은 단어의 등장 빈도수의 총 합

# 단어와 빈도수의 쌍(pair)을 key와 value로 받는다.
for key, value in tokenizer.word_counts.items():
    total_freq = total_freq + value # 빈도수(value)를 카운팅

    # 단어의 등장 빈도수가 threshold보다 작으면
    if(value < threshold):
        rare_cnt = rare_cnt + 1
        rare_freq = rare_freq + value

# 희귀 단어 = 등장 빈도수가 threshold 보다 작은 단어
print('단어 집합(vocabulary)의 크기 :',total_cnt)
print('등장 빈도가 %s번 이하인 희귀 단어의 수: %s'%(threshold - 1, rare_cnt))
print("단어 집합에서 희귀 단어의 비율:", (rare_cnt / total_cnt)*100)
print("전체 등장 빈도에서 희귀 단어 등장 빈도 비율:", (rare_freq / total_freq)*100) 

# 전체 단어 개수 중 빈도수 2이하인 단어 개수는 제거.
vocab_size = total_cnt - rare_cnt + 2 # 0번 패딩 토큰과 1번 OOV 토큰을 고려하여 +2
print('단어 집합의 크기 :',vocab_size)

tokenizer = Tokenizer(vocab_size, oov_token = 'OOV') # OOV: Out of Vocabulary
tokenizer.fit_on_texts(X_train)
X_train = tokenizer.texts_to_sequences(X_train)
X_test = tokenizer.texts_to_sequences(X_test)

y_train = np.array(train_data['label'])
y_test = np.array(test_data['label'])
#%%
# 희귀 단어들로만 이루어진 샘플들 제거
drop_train = [index for index, sentence in enumerate(X_train) if len(sentence) < 1] # enumerate를 활용해서 길이가 1보다 작은 샘플의 인덱스를 저장합니다.

# 빈 샘플들을 제거
X_train = np.delete(X_train, drop_train, axis=0)
y_train = np.delete(y_train, drop_train, axis=0)
print(len(X_train))
print(len(y_train))
#%%
# 패딩
print('리뷰의 최대 길이 :',max(len(l) for l in X_train))
print('리뷰의 평균 길이 :',sum(map(len, X_train))/len(X_train))
plt.hist([len(s) for s in X_train], bins=50)
plt.xlabel('length of samples')
plt.ylabel('number of samples')
plt.show()

def below_threshold_len(max_len, nested_list):
  cnt = 0
  for s in nested_list:
    if(len(s) <= max_len):
        cnt = cnt + 1
  print('전체 샘플 중 길이가 %s 이하인 샘플의 비율: %s'%(max_len, (cnt / len(nested_list))*100))

max_len = 35
below_threshold_len(max_len, X_train)

# %%
# 모델 생성 및 훈련
# LSTM 모델로 영화 리부 감성 분류
from keras.layers import Embedding, Dense, LSTM 
from keras.models import Sequential
from tensorflow.keras.models import load_model
from keras.preprocessing.sequence import pad_sequences 
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

model = Sequential()
model.add(Embedding(vocab_size, 100))
model.add(LSTM(128))
model.add(Dense(1, activation='sigmoid'))

es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=4)
mc = ModelCheckpoint('best_model.h5', monitor='val_acc', mode='max', verbose=1, save_best_only=True)

model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])
# 모델 훈련
history = model.fit(X_train, y_train, epochs=15, callbacks=[es, mc], batch_size=60, validation_split=0.2)

#%%
# 모델 검증
loaded_model = load_model('best_model.h5')
print("\n 테스트 정확도: %.4f" % (loaded_model.evaluate(X_test, y_test)[1]))

# %%