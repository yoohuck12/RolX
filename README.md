ReFeX and RolX
===============================================
[![codebeat badge](https://codebeat.co/badges/f688b042-0641-4aa7-a122-9719e3372ca9)](https://codebeat.co/projects/github-com-benedekrozemberczki-rolx-master) [![repo size](https://img.shields.io/github/repo-size/benedekrozemberczki/RolX.svg)](https://github.com/benedekrozemberczki/RolX/archive/master.zip)⠀[![benedekrozemberczki](https://img.shields.io/twitter/follow/benrozemberczki?style=social&logo=twitter)](https://twitter.com/intent/follow?screen_name=benrozemberczki)

<p align="justify">
ReFex 는 대형 그래프에서 노드의 구조적 속성을 묘사하기 위한 바이너리 특색을 만들어내는 구조적 그래프 특색 추출 알고리즘이다.
첫번째로 연속된 특색들이 이웃의 묘사가능한 통계에 근거하여 추출된다.
이러한 통계는 재귀적으로 모이게 된다.
원래의 알고리즘은 재귀적으로 실행되는 동안 더 진보된 기술 통계를 추출할 수 있도록 확장되었다.
게다가 특색 추출 재귀 실행의 횟수와 이진 binning 역시 제어가능한 파라미터로 제공된다.
강하게 연결된 특색들은 임의적으로 설정된 threshold 에 의해 drop 될 수도 있다.
</p>

<p align="center">
  <img width="720" src="structural.jpeg">
</p>

<p align="justify">
롤렉스는 ReFeX 로부터 추출된 특색들을 이용해서 작은 dimension 의 구조적 노드 표현방법을 생성하기 위해서 이진 노드 특색 매트릭스를 생성한다.
비슷한 구조적 특색을 갖는 노드는 이 잠재적 공간안에서 함께 클러스터링 된다.
원래의 모델은 음수가 아닌 매트릭스를 사용하였지만, 이 리파지토리에서 우리는 경사 하강의 강력한 변종으로 훈련된 암시적 매트릭스 생성 모델을 사용했다.
우리 모델은 GPU 에서도 동작한다.
</p>

이 리파지토리에서는 다음 논문들이 묘사한 모델의 커스텀 구현을 제공한다.

> **It's who you know: graph mining using recursive structural features.**
> Keith Henderson, Brian Gallagher, Lei Li, Leman Akoglu, Tina Eliassi-Rad, Hanghang Tong and Christos Faloutsos.
> Proceedings of the 17th ACM SIGKDD international conference on Knowledge discovery and data mining.
> [[Paper]](http://www.cs.cmu.edu/~leili/pubs/henderson-kdd2011.pdf)


> RolX: Structural Role Extraction & Mining in Large Graphs
> Keith Henderson, Brian Gallagher, Tina Eliassi-Rad, Hanghang Tong, Sugato Basu, Leman Akoglu, Danai Koutra, Christos Faloutsos and Lei Li.
> Proceedings of the 18th ACM SIGKDD international conference on Knowledge discovery and data mining.
> [[Paper]](https://web.eecs.umich.edu/~dkoutra/papers/12-kdd-recursiverole.pdf)

또 다른 구현은 다음 링크에서 확인 가능하다: [[here]](https://github.com/dkaslovsky/GraphRole).

### Requirements

The codebase is implemented in Python 2.7.
package versions used for development are just below.
```
networkx          1.11
tqdm              4.19.5
numpy             1.13.3
pandas            0.20.3
tensorflow-gpu    1.3.0
jsonschema        2.6.0
texttable         1.2.1
```

### Datasets

<p align="justify">
csv 파일 형식으로 된 input graph 를 인자로 받는다.
모든 행은 두 노드 사이의 edge 를 가리킨다.
첫번째 열은 헤더다.
노드들은 0부터 인덱싱 되어야 한다.
예제로 페이스북 티비쇼 데이터셋이 'data' 디렉토리에 있다.

### Logging

<p align="justify">
모델은 각 epoch 마다 파라미터 세팅, 추출된 특색, 그리고 생성(factorization) loss 를 로깅한다.
다음의 것들을 로그한다.

```
1. Hyperparameter settings.                  실험에 사용된 하이퍼파라미터를 저장한다.
3. Number of extracted features per epoch.   프루닝 전후로 특색의 개수를 입력으로 받는다.
2. Cost per epoch.                           재건 비용이 매 iteration 별로 저장된다.
4. Runtime.                                  특색 추출과 최적화에 들어간 시간을 초 단위로 측정한다.
```

### Options

<p align="justify">
특색 추출 및 생성은 'src/main.py' 스크립트를 통해서 다뤄진다. 다음은 command line 인자들이다.

#### Input and output options

```
  --input                        STR   Input graph path.           Default is `data/tvshow_edges.csv`.
  --embedding-output             STR   Embeddings path.            Default is `output/embeddings/tvhsow_embedding.csv`.
  --recursive-features-output    STR   Recursive features path.    Default is `output/features/tvhsow_features.csv`.
  --log-output                   STR   Log path.                   Default is `output/logs/tvhsow.log`.
```

#### ReFeX options

```
  --recursive-iterations  INT      Number of recursions.                                Default is 3.
  --bins                  INT      Number of binarization bins.                         Default is 4.
  --aggregator            STR      Aggregation strategy (simple/complex).               Default is `simple`.
  --pruning-cutoff        FLOAT    Absolute correlation for feature dropping.           Default is 0.9.
```

#### RolX options

```
  --epochs                  INT       Number of epochs.                           Default is 10.
  --batch-size              INT       Number of edges in batch.                   Default is 32.
  --dimensions              INT       Number of dimensions.                       Default is 16.
  --initial-learning-rate   FLOAT     Initial learning rate.                      Default is 0.01.
  --minimal-learning-rate   FLOAT     Final learning rate.                        Default is 0.001.
  --annealing-factor        FLOAT     Annealing factor for learning rate.         Default is 1.0.
  --lambd                   FLOAT     Weight regularization penalty.              Default is 10**-3.
```

### Examples

<p align="justify">
다음의 명령어들은 구조적 특색을 생성하고, 그래프 임베딩을 배우며, 디스크에 이것을 기록한다.
노드 표현은 ID 순서로 되어있다. </p>

<p align="justify">
기본 하이퍼파라미터 설정으로 기본 데이터셋의 RolX 임베딩을 만든다.
ReFeX 특색, RolX 임베딩, 그리고 로그 파일을 기본 경로에 저장한다. </p>

```
python src/main.py
```
Creating an embedding of an other dataset the `Facebook Companies`. Saving the output and the log in a custom place.

```
python src/main.py --input data/company_edges.csv  --embedding-output output/embeddings/company_embedding.csv --recursive-features-output output/features/company_features.csv --log-output output/logs/company_log.json
```

Creating an embedding of the default dataset in 128 dimensions with 8 binary feature bins.

```
python src/main.py --dimensions 128 --bins 8
```



--------------------------------------------------------------------------------

**License**

- [GNU](https://github.com/benedekrozemberczki/RolX/blob/master/LICENSE)
