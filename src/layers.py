import math
import numpy as np
import tensorflow as tf

class Factorization:
    """
    Factorization layer class.
    """
    def __init__(self, args, user_size, feature_size):
        """
        Layer 를 적절한 매트릭스와 bias 를 사용해 초기화한다.
        인풋 변수도 여기서 초기화된다.
        """
        self.args = args
        self.user_size = user_size
        self.feature_size = feature_size 

        # 실제 그래프에서 사용될 인자를 선언한다.
        self.target = tf.placeholder(tf.int64, shape=[None, None])
        self.edge_indices_left = tf.placeholder(tf.int64, shape=[None])
        self.edge_indices_right = tf.placeholder(tf.int64, shape=[None])

        # Node 별로 dimensions 씩 embedding 한다.
        self.embedding_node = tf.Variable(tf.random_uniform([self.user_size, self.args.dimensions],
                                            -0.1/self.args.dimensions, 0.1/self.args.dimensions))

        # Feature 별로 dimensions 씩 embedding 한다.
        self.embedding_feature = tf.Variable(tf.random_uniform([self.feature_size, self.args.dimensions],
                                            -0.1/self.args.dimensions, 0.1/self.args.dimensions))
      
    def __call__(self):
        """
        Loss 를 정의한다.
        """
        # Embedding_node 에서 특정 노드에 해당하는 embedding 을 가져온다.
        self.embedding_left = tf.nn.embedding_lookup(self.embedding_node , self.edge_indices_left) 
        # Embedding_feature 에서 edge_indices_right 에 해당하는 embedding 을 가져온다.
        # RolX 에서 edge_indices_right 에는 feature index 가 들어오게 된다.
        self.embedding_right = tf.nn.embedding_lookup(self.embedding_feature, self.edge_indices_right)
        # 위의 두 embedding 을 곱하고 sigmoid 함수를 통과시켜 prediction 을 얻게된다.
        # 노드의 Node embedding * Feature Embedding 인데
        # RolX 에서는 특정 노드 Embedding * Feature Embedding 을 구하게 된다.
        self.embedding_predictions = tf.sigmoid(tf.matmul(self.embedding_left, tf.transpose(self.embedding_right)))
        # Target 과 위에서 구한 prediction 의 log_loss 를 구한뒤에 평균을 구한다.
        # Target 은 refex 로 구한 node 들의 feature vector 들이 된다.
        # RolX 에서는 노드 Embedding 과 Feature Embedding 을 사용해서 실제 Feature Embedding 을 갖게 만들고 싶어한다.
        return tf.reduce_mean(tf.losses.log_loss(self.target,self.embedding_predictions))
