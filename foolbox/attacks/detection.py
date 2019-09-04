import numpy as np
import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, GlobalAveragePooling2D, Activation, InputLayer
from collections import OrderedDict

class Detector(object):

    def __init__(self, threshold, K=50, size=None, chunk_size=1000, name="detector", weights_path="./encoder_1.h5", ith_query=1, save_queries=False):
        self.threshold = threshold
        self.K = K
        self.size = size
        self.num_queries = 0
        self.ith_query = ith_query
        self.name = name

        # Stats
        self.prev_query = None
        self.dist_to_prev = 0
        self.curr_k_dist = 0

        # Save queries
        self.save_queries = save_queries
        self.queries = []

        self.buffer = []
        self.memory = []
        self.chunk_size = chunk_size

        self.history = [] # Tracks number of queries (t) when attack was detected
        self.history_by_attack = []
        self.detected_dists = [] # Tracks knn-dist that was detected

        self._init_encoder(weights_path)

    def _init_encoder(self, weights_path):
        raise NotImplementedError("Must implement your own encode function!")

    def process(self, queries, num_queries_so_far, encoded=False):
        if not encoded:
            queries = self.encode(queries)
        if self.save_queries:
            self.queries.append(queries)

        for query in queries:
            self.process_query(query, num_queries_so_far)
            num_queries_so_far += 1

    def process_query(self, query, num_queries_so_far):
        if self.num_queries % self.ith_query != 0:
            self.num_queries += 1
            return

        if len(self.memory) == 0 and len(self.buffer) < self.K:
            self.buffer.append(query)
            self.num_queries += 1
            return

        if self.prev_query is not None:
            self.dist_to_prev = np.linalg.norm(query - self.prev_query)
        self.prev_query = query

        k = self.K
        all_dists = []

        if len(self.buffer) > 0:
            queries = np.stack(self.buffer, axis=0)
            dists = np.linalg.norm(queries - query, axis=-1)
            all_dists.append(dists)

        for queries in self.memory:
            dists = np.linalg.norm(queries - query, axis=-1)
            all_dists.append(dists)

        dists = np.concatenate(all_dists)
        k_nearest_dists = np.partition(dists, k - 1)[:k, None]
        k_avg_dist = np.mean(k_nearest_dists)
        self.curr_k_dist = k_avg_dist

        self.buffer.append(query)
        self.num_queries += 1

        if len(self.buffer) >= self.chunk_size:
            self.memory.append(np.stack(self.buffer, axis=0))
            self.buffer = []

        # print("[debug]", num_queries_so_far, k_avg_dist)
        is_attack = k_avg_dist < self.threshold
        if is_attack:
            self.history.append(self.num_queries)
            self.history_by_attack.append(num_queries_so_far + 1)
            self.detected_dists.append(k_avg_dist)
            # print("[encoder] Attack detected:", str(self.history), str(self.detected_dists))
            self.clear_memory()

    def clear_memory(self):
        self.buffer = []
        self.memory = []

    def get_detections(self):
        history = self.history
        epochs = []
        for i in range(len(history) - 1):
            epochs.append(history[i + 1] - history[i])

        return epochs

class L2Detector(Detector):
    def _init_encoder(self, weights_path):
        self.encode = lambda x : x.reshape((x.shape[0], -1))

class SimilarityDetector(Detector):
    def _init_encoder(self, weights_path):
        encoder = cifar10_encoder()
        encoder.load_weights(weights_path, by_name=True)
        self.encoder = encoder
        self.encode = lambda x : encoder.predict(x)

class ExperimentDetectors():
    def __init__(self, active=True, detectors=None):
        self.active = active

        if detectors is None:
            detectors = [
                ("similarity", SimilarityDetector(threshold=1.44, K=50, weights_path="./encoders/encoder_all.h5")),
                ("l2", L2Detector(threshold=5.069, K=50)),
                ("sim-no-brightness", SimilarityDetector(threshold=1.56, K=50, weights_path="./encoders/encoder_no_brightness.h5")),
            ]

        self.detectors = OrderedDict({})
        for d_name, detector in detectors:
            self.detectors[d_name] = detector

    def process(self, queries, num_queries_so_far, encoded=False):
        if not self.active:
            return

        for name, detector in self.detectors.items():
            detector.process(queries, num_queries_so_far, encoded=encoded)
            # Log stats
            # print("[detection-stats]: %s: k-dist %.5f; prev-dist %.5f" % (name, detector.curr_k_dist, detector.dist_to_prev))


    def process_query(self, query, num_queries_so_far):
        if not self.active:
            return

        for _, detector in self.detectors.items():
            detector.process_query(query, num_queries_so_far)


class MultiAttackDetectors(ExperimentDetectors):
    def __init__(self, active=True, detectors=None):
        """
        detectors = [
            ("l2-k=50-i=1", SimilarityDetector(threshold=1.44, K=50, name="sim-k=50-i=1", weights_path="./encoders/encoder_all.h5")),
             ("l2-k=25-i=1", SimilarityDetector(threshold=1.26, K=25, weights_path="./encoders/encoder_all.h5")),
             ("l2-k=10-i=1", SimilarityDetector(threshold=1.02, K=10, weights_path="./encoders/encoder_all.h5")),
#            ("sim-k=50-i=5", SimilarityDetector(threshold=1.44, K=50, name="sim-k=50-i=5", weights_path="./encoders/encoder_all.h5", ith_query=5)),
#            ("sim-k=50-i=10", SimilarityDetector(threshold=1.44, K=50, name="sim-k=50-i=10", weights_path="./encoders/encoder_all.h5", ith_query=10)),
#            ("sim-k=50-i=25", SimilarityDetector(threshold=1.44, K=50, name="sim-k=50-i=25", weights_path="./encoders/encoder_all.h5", ith_query=25)),
#            ("sim-k=50-i=50", SimilarityDetector(threshold=1.44, K=50, name="sim-k=50-i=50", weights_path="./encoders/encoder_all.h5", ith_query=50)),
#             ("sim-k=50-i=100", SimilarityDetector(threshold=1.44, K=50, weights_path="./encoders/encoder_all.h5", ith_query=100)),
#            ("sim-k=10-i=50", SimilarityDetector(threshold=1.02, K=10, weights_path="./encoders/encoder_all.h5", ith_query=50)),

        ]
"""

        detectors = []
        for k, delta in [(5, 0.799), (10, 1.02), (25, 1.26), (50, 1.44)]:
            for i in [1]:
                save_queries = False
                detectors.append(("sim-k=%d-i=%d" % (k, i), SimilarityDetector(threshold=delta, K=k, ith_query=i, name="sim-k=%d-i=%d" % (k, i), save_queries=save_queries,  weights_path="./encoders/encoder_all.h5")))

#        self.encode_once = detectors[0][1].encode
        # ExperimentDetectors.__init__(self, detectors=detectors)

        # detectors = [
        #    ("sim-k=50-i=1", SimilarityDetector(threshold=1.44, K=50, name="sim-k=50-i=1", save_queries=True, weights_path="./encoders/encoder_all.h5")),
        #    ("l2-k=50-i=1", L2Detector(threshold=5.07, K=50, name="l2-k=50-i=1", weights_path="./encoders/encoder_all.h5")),
        #    ("l2-k=25-i=1", L2Detector(threshold=4.68, K=25, name="l2-k=25-i=1", weights_path="./encoders/encoder_all.h5")),
        #    ("l2-k=10-i=1", L2Detector(threshold=4.07, K=10, name="l2-k=10-i=1", weights_path="./encoders/encoder_all.h5")),
        #    ("l2-k=5-i=1", L2Detector(threshold=3.50, K=5, name="l2-k=5-i=1", weights_path="./encoders/encoder_all.h5")),
        # ]
        ExperimentDetectors.__init__(self, detectors=detectors)
        self.encode_once = detectors[0][1].encode

    def process(self, queries, num_queries_so_far):
        # queries = self.encode_once(queries)
        ExperimentDetectors.process(self, queries, num_queries_so_far, encoded=False)

def cifar10_encoder(encode_dim=256):
    model = Sequential()
#     model.add(InputLayer(input_tensor=input_placeholder,
#                      input_shape=(32, 32, 3)))

    model.add(Conv2D(32, (3, 3), padding='same', name='conv2d_1', input_shape=(32, 32, 3)))
    model.add(Activation('relu', name='activation_1'))
    model.add(Conv2D(32, (3, 3), name='conv2d_2'))
    model.add(Activation('relu', name='activation_2'))
    model.add(MaxPooling2D(pool_size=(2, 2), name='max_pooling2d_1'))
    model.add(Dropout(0.25, name='dropout_1'))

    model.add(Conv2D(64, (3, 3), padding='same', name='conv2d_3'))
    model.add(Activation('relu', name='activation_3'))
    model.add(Conv2D(64, (3, 3), name='conv2d_4'))
    model.add(Activation('relu', name='activation_4'))
    model.add(MaxPooling2D(pool_size=(2, 2), name='max_pooling2d_2'))
    model.add(Dropout(0.25, name='dropout_2'))

    model.add(Flatten(name='flatten_1'))
    model.add(Dense(512, name='dense_1'))
    model.add(Activation('relu', name='activation_5'))
    model.add(Dropout(0.5, name='dropout_3'))
    model.add(Dense(encode_dim, name='dense_encode'))
    model.add(Activation('linear', name='encoding'))

    return model

