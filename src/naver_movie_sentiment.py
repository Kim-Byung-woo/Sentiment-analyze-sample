#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import urllib.request

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# %%
# 데이터셋 로드
import os
file_dir = os.getcwd() # 현재 파일 경로 추출
file_dir = os.path.dirname(file_dir) # 상위 경로 추출 - 코드 파일과 단어 리스트 파일 위치가 틀려서

train_df = pd.read_table(file_dir + "/data/ratings_train.txt")
test_df = pd.read_table(file_dir + "/data/ratings_test.txt")
#%%
train_df.isnull().any() #document에 null값이 있다.
train_df['document'] = train_df['document'].fillna(''); #null값을 ''값으로 대체

test_df.isnull().any()
test_df['document'] = test_df['document'].fillna(''); #null값을 ''값으로 대체
#%%
# 데이터 개수가 너무 많아 임의로 축소
train_df = train_df.head(2000)
test_df = test_df.head(100)

print('훈련 데이터 리뷰 개수 :',len(train_df)) # 리뷰 개수 출력
print('테스트 데이터 리뷰 개수 :',len(test_df)) # 리뷰 개수 출력
#%%
import json

from konlpy.tag import Okt
okt = Okt()

def tagging(doc):
    #형태소와 품사를 join
    return ['/'.join(t) for t in okt.pos(doc, norm=True, stem=True)] # norm: 정규화 -> 표현이 방법이 다른 단어들을 통합시켜 같은 단어로 만듭니다. stem: 단어의 어간 추출

# 기존에 태깅한 데이터가 있으면 불러옵니다.
if os.path.isfile(file_dir + "/data/train_docs.json"):
    print('Json File is already')
    with open(file_dir + "/data/train_docs_ori.json", encoding='UTF8') as f:
        train_docs = json.load(f)
    with open(file_dir + "/data/test_docs_ori.json", encoding='UTF8') as f:
        test_docs = json.load(f)
else: # 태깅한 데이터 없는 경우 태깅 실시
    train_docs = [(tagging(row[1]), row[2]) for row in train_df.values]
    test_docs = [(tagging(row[1]), row[2]) for row in test_df.values]

    # JSON 파일로 저장
    with open(file_dir + "/data/train_docs.json", 'w', encoding="utf-8") as make_file:
        json.dump(train_docs, make_file, ensure_ascii=False, indent="\t")
    with open(file_dir + "/data/test_docs.json", 'w', encoding="utf-8") as make_file:
        json.dump(test_docs, make_file, ensure_ascii=False, indent="\t")
        
from pprint import pprint
pprint(train_docs[0])
#%%

# 분석한 데이터의 토큰(문자열을 분석을 위한 작은 단위)의 갯수를 확인
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
FREQUENCY_COUNT = 1000; # 시간적 여유가 있다면 10000개를 해보기
selected_words = [f[0] for f in text.vocab().most_common(FREQUENCY_COUNT)] # 빈도수가 높은 1000개 단어 추출

# 단어리스트 문서에서 상위 1000개들중 포함되는 단어들이 개수
def term_frequency(doc):
    return [doc.count(word) for word in selected_words]


# list comprehension을 사용할 경우
'''
train_docs의 구조가 [[[토큰],라벨], [[토큰],라벨] ,[[토큰],라벨], [[토큰],라벨]...] 이렇게 되있습니다
따라서 for문에서 for d,_ in test_docs를 하게 되면 d에는 토큰, _에는 라벨 정보가 순차적으로 반복됩니다.

term_frequency(d)는 상위 1000개 단어들 중 토큰의 빈도수를 반환합니다.
ex. 1000개의 단어 중 '영화'라는 단어가 있고 토큰에서 '영화'라는 단어가 3개 있으면 빈도수는 3
'''
x_train = [term_frequency(d) for d,_ in train_docs]
x_test = [term_frequency(d) for d,_ in test_docs]

'''
# list comprehension을 사용안할 경우

x_train = []
for idx in range(len(train_docs)):
    x_train.append(term_frequency(train_docs[idx][0]))

x_test = []
for idx in range(len(test_docs)):
    x_test.append(term_frequency(test_docs[idx][0]))
'''

# 라벨 데이터 전처리
y_train = [c for _,c in train_docs] # train_docs에서 라벨 정보만 가져와 list comprehension 실행
y_test = [c for _,c in test_docs] # test_docs에서 라벨 정보만 가져와 list comprehension 실행
#%%
x_train_a = np.asarray(x_train).astype('float32')
x_test_a = np.asarray(x_test).astype('float32')

y_train_a = np.asarray(y_train).astype('float32')
y_test_a = np.asarray(y_test).astype('float32')


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
#%%
# 모델 예측
review = "개노잼이네"
token = tagging(review)
tfq = term_frequency(token)
tfq_arr = np.asarray(tfq).astype('float32')

# x_test->는 2차원 tfq_arr-> 1차원입니다.
# 데이터 형태를 맞춰 주기 위해 np.expand_dims 사용 
data = np.expand_dims(tfq_arr, axis=0) # expand_dims를 사용하여 tfq_arr을 2차원으로 변환

'''
# 데이터 형태를 맞춰 주기 위해 np.expand_dims 사용
data = np.reshape(tfq_arr, (1, len(tfq_arr)))
'''
score = float(model.predict(data))
if(score > 0.5):
    print(f"{review} ==> 긍정 ({round(score*100)}%)")
else:
    print(f"{review} ==> 부정 ({round((1-score)*100)}%)")

#%%
#모델 저장
model.save('movie_review_model.h5')