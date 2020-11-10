#%%
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
file_dir = os.path.dirname(file_dir) # 상위 경로 추출 - 코드 파일과 단어 리스트 파일 위치가 틀려서

train_df = pd.read_table(file_dir + "/data/ratings_train.txt")
test_df = pd.read_table(file_dir + "/data/ratings_test.txt")

# 데이터 개수 확인
print('훈련 데이터 리뷰 개수 :',len(train_df)) # 리뷰 개수 출력
print('테스트 데이터 리뷰 개수 :',len(test_df)) # 리뷰 개수 출력
#%%
okt = Okt()

def tokenize(doc):
    #형태소와 품사를 join
    return ['/'.join(t) for t in okt.pos(doc, norm=True, stem=True)]
#%%
train_df.isnull().any() #document에 null값이 있다.
train_df['document'] = train_df['document'].fillna(''); #null값을 ''값으로 대체

test_df.isnull().any()
test_df['document'] = test_df['document'].fillna(''); #null값을 ''값으로 대체
#%%
train_df = train_df.head(2000)
test_df = test_df.head(100)

print('훈련 데이터 리뷰 개수 :',len(train_df)) # 리뷰 개수 출력
print('테스트 데이터 리뷰 개수 :',len(test_df)) # 리뷰 개수 출력

#%%
import json
if os.path.isfile(file_dir + "/data/train_docs.json"):
    print('Json File is already')
    with open(file_dir + "/data/train_docs.json", encoding='UTF8') as f:
        train_docs = json.load(f)
    with open(file_dir + "/data/test_docs.json", encoding='UTF8') as f:
        test_docs = json.load(f)
else:
    train_docs = [(tokenize(row[1]), row[2]) for row in train_df.values]
    test_docs = [(tokenize(row[1]), row[2]) for row in test_df.values]

    # JSON 파일로 저장
    with open(file_dir + "/data/train_docs.json", 'w', encoding="utf-8") as make_file:
        json.dump(train_docs, make_file, ensure_ascii=False, indent="\t")
    with open(file_dir + "/data/test_docs.json", 'w', encoding="utf-8") as make_file:
        json.dump(test_docs, make_file, ensure_ascii=False, indent="\t")
        
from pprint import pprint
pprint(train_docs[0])
#%%
tokens = [t for d in train_docs for t in d[0]]
print("토큰개수:", len(tokens))

import nltk
text = nltk.Text(tokens, name='NMSC')

#토큰개수
print(len(text.tokens))

#중복을 제외한 토큰개수
print(len(set(text.tokens)))

#출력빈도가 높은 상위 토큰 20개
print(text.vocab().most_common(20))


from matplotlib import font_manager, rc
plt.rc('font', family='Malgun Gothic') # Window 의 한글 폰트 설정
plt.figure(figsize=(20,10))
text.plot(20)

#%%
FREQUENCY_COUNT = 1000; #시간적 여유가 있다면 10000개를 해보도록~
selected_words = [f[0] for f in text.vocab().most_common(FREQUENCY_COUNT)]


#단어리스트 문서에서 상위 1000개들중 포함되는 단어들이 개수
def term_frequency(doc):
    return [doc.count(word) for word in selected_words]

#문서에 들어가는 단어 개수
x_train = [term_frequency(d) for d,_ in train_docs]
x_test = [term_frequency(d) for d,_ in test_docs]
#라벨(1 or 0)
y_train = [c for _,c in train_docs]
y_test = [c for _,c in test_docs]


x_train = np.asarray(x_train).astype('float32')
x_test = np.asarray(x_test).astype('float32')

y_train = np.asarray(y_train).astype('float32')
y_test = np.asarray(y_test).astype('float32')


#%%
#학습 프로세스 설정
import tensorflow as tf

#레이어 구성
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(FREQUENCY_COUNT,)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=0.001),
    loss=tf.keras.losses.binary_crossentropy,
    metrics=[tf.keras.metrics.binary_accuracy])

#학습 데이터로 학습
model.fit(x_train, y_train, epochs=10, batch_size=512)

# 모델 평가
results = model.evaluate(x_test, y_test)

# 모델 예측
review = "개노잼이네"
token = tokenize(review)
tfq = term_frequency(token)
data = np.expand_dims(np.asarray(tfq).astype('float32'), axis=0)
score = float(model.predict(data))
if(score > 0.5):
        print(f"{review} ==> 긍정 ({round(score*100)}%)")
else:
    print(f"{review} ==> 부정 ({round((1-score)*100)}%)")

#%%
#모델 저장
model.save('movie_review_model.h5')
