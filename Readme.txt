news title.py
- 크롤링  댓글 라벨링
- 훈련/테스트 데이터 분류
- 모델 학습
- 테스트 데이터 감성분석 -> 라벨링과 얼마나 차이가 있는지 비교 -> 여기서 말하는 정확도는 기존 라벨링과의 정확도이지 긍정/부정의 정확도가 아님. 따라서 사용하기 어려움

comment_labeling_dictionary.py
- 구축한 단어사전을 기반으로 하여 크롤링 댓글 라벨링

movie_review_sentiment_naver.py(https://cyc1am3n.github.io/2018/11/10/classifying_korean_movie_review.html 참조)
- 네이버 영화 리뷰 데이터를 토대로 모델을 생성
- 라벨링 댓글 데이터 토대로 모델 생성
- 샘플 데이터(라벨링 하지 않은 크롤링 댓글) 감성분석

movie_review_sentiment_wiki.py(https://wikidocs.net/44249, 위키독스 - 딥러닝을 이용한 자연어 처리 입문 참조)
- 네이버 영화 리뷰 데이터를 토대로 모델을 생성
- 라벨링 댓글 데이터 토대로 모델 생성
- 샘플 데이터(라벨링 하지 않은 크롤링 댓글) 감성분석

참고 사이트
Sentiment Anlayze
https://somjang.tistory.com/entry/Keras%EA%B8%B0%EC%82%AC-%EC%A0%9C%EB%AA%A9%EC%9D%84-%EA%B0%80%EC%A7%80%EA%B3%A0-%EA%B8%8D%EC%A0%95-%EB%B6%80%EC%A0%95-%EC%A4%91%EB%A6%BD-%EB%B6%84%EB%A5%98%ED%95%98%EB%8A%94-%EB%AA%A8%EB%8D%B8-%EB%A7%8C%EB%93%A4%EC%96%B4%EB%B3%B4%EA%B8%B0
https://cyc1am3n.github.io/2018/11/10/classifying_korean_movie_review.html
https://wikidocs.net/44249
https://github.com/hoho0443/classify_comment_emotion
https://devtimes.com/nlp-korea-movie-review
https://colab.research.google.com/drive/1tIf0Ugdqg4qT7gcxia3tL7und64Rv1dP#scrollTo=P58qy4--s5_x

https://m.blog.naver.com/samsjang/220982297456
https://techblog-history-younghunjo1.tistory.com/111
https://shinminyong.tistory.com/13
https://mjdeeplearning.tistory.com/67
https://soyoung-new-challenge.tistory.com/46
https://pinkwink.kr/1025

BERT
http://docs.likejazz.com/bert/

OOV
https://wikidocs.net/31766

감성사전
https://github.com/park1200656/KnuSentiLex
https://projectlog-eraser.tistory.com/19