import keras
import numpy as np
import sys
import argparse
import foolbox

from foolbox.attacks import BoundaryAttack
from foolbox.attacks import PerlinBoundaryAttack

from foolbox.adversarial import Adversarial
from foolbox.criteria import TargetClass
from foolbox.sampling.sample_generator import SampleGenerator
from foolbox.distances import Linf
from foolbox.detection import transforms

from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, GlobalAveragePooling2D, Activation, InputLayer
from keras.datasets import cifar10

import tensorflow as tf

import ast
import time

CIFAR_10_WEIGHTS_PATH = "cifar10_ResNet20v1_model.h5"
INDICES_FILE = "./image_idxs.npz"

def load_cifar_model():
    model = keras.models.load_model(CIFAR_10_WEIGHTS_PATH)
    # logits = Model(inputs=model.input, outputs=model.layers[-2].output)
    # logits = cifar10_logits()
    # print(logits.output.op.inputs)
    return model


def cifar10_logits():
    model = Sequential()

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
    model.add(Dense(10, name='dense_2'))
    model.add(Activation('linear', name='logits'))

    return model

def pseudorandom_target(index, total_indices, true_class):
    rng = np.random.RandomState(index)
    target = true_class
    while target == true_class:
        target = rng.randint(0, total_indices)
    return target

### NP Transforms ###
def transform_brightness(C):
    def brightness(x):
        k = x.shape[0]
        signs = np.random.choice([-1, 1], size=(k,))
        scalings = np.random.uniform(low=-C, high=C, size=(k,)) * signs
        scalings = np.reshape(scalings, (k,) + (1, 1, 1))
        return np.clip(x + scalings, a_min=0, a_max=1)

    return brightness

def transform_pixel_scale(C):
    def pixel_scale(x):
        k = x.shape[0]
        scalings = 1 + np.random.uniform(low=-C, high=C, size=(k,))
        scalings = np.reshape(scalings, (k,) + (1, 1, 1))
        return np.clip(x * scalings, a_min=0, a_max=1)

    return pixel_scale

# For each channel, this Op computes the mean of the image pixels in the channel
# and then adjusts each component x of each pixel to (x - mean) * contrast_factor + mean.
def transform_contrast(C):
    def contrast(x):
        k = x.shape[0]
        scalings = 1 + np.random.uniform(low=C, high=1, size=(k,))
        scalings = np.reshape(scalings, (k,) + (1, 1, 1))
        return np.clip(x * scalings, a_min=0, a_max=1)

    return contrast

## TF Transforms ##
def wrap_tf_transform(sess, transformer):
    points = tf.placeholder(tf.float32, [None, 32, 32, 3])
    transform_t = transformer.generate_samples(points, 1, (32, 32, 3))
    def transform(x):
        return sess.run(transform_t, feed_dict={points: x})
    return transform

def get_test_model_correct(model):
    (_, _), (x_test, y_test) = cifar10.load_data()
    return x_test, y_test
    """
    x_test_norm = x_test / 255.
    scores = model.predict(x_test_norm)
    preds = scores.argmax(axis=-1)
    model_correct = (preds == y_test.flatten())
    return x_test[model_correct], y_test[model_correct]
    """

### Create Model ###

keras.backend.set_learning_phase(0)
kmodel = load_cifar_model()
model = foolbox.models.KerasModel(kmodel, bounds=(0, 1))


### Parse Args ###

parser = argparse.ArgumentParser()
parser.add_argument('--save-name', type=str)
parser.add_argument('--num_iters', type=int, default=12500)
parser.add_argument('--normal_factor', type=float, default=1.0)
parser.add_argument('--transform', type=str, default=None)
parser.add_argument('--transform_param', type=float)
parser.add_argument('--idx-range', type=int, nargs=2)

args = parser.parse_args()
save_name = args.save_name

start_idx, end_idx = 0, 100
idx_range_specified = args.idx_range is not None
if idx_range_specified:
    start_idx, end_idx = args.idx_range[0], args.idx_range[1]

transform = args.transform
if transform is not None:
    # variable is transform name
    sess = keras.backend.get_session()
    print("Wrapping up transform", args.transform)
    transform = wrap_tf_transform(sess, transforms.get_transform(args.transform, args.transform_param))

### Init Boundary Attack ###

# TODO: Targeted attack
x_test, y_test = get_test_model_correct(kmodel)
x_test_idx = np.arange(x_test.shape[0])
dets = []
dists = []
l2_distortions = []
linf_distortions = []
successes = []
failures = []
errors = []
double_check = []

indices_provided = INDICES_FILE is not None

if indices_provided:
    # f = open(INDICES_FILE, 'r')
    # img_indices = ast.literal_eval(f.readline().strip())
    img_idxs = []
    npz = np.load(INDICES_FILE)
    for orig_i, target_i, target_class in zip(npz['original_idxs'], npz['target_idxs'], npz['target_classes']):
        img_idxs.append((orig_i, target_i, target_class))

indices_targets = []

img_shape = (32, 32, 3)
start_time = time.time()
with SampleGenerator(shape=img_shape, n_threads=1, queue_lengths=100) as sample_gen:
    for i in range(start_idx, end_idx):
        print("TRIAL", i)
        if indices_provided:
            img_index, target_i, target_class = img_idxs[i]
        else:
            img_index = np.random.randint(0, x_test.shape[0])

        x, y = x_test[img_index, None][0], y_test[img_index][0]
        orig_class = y
        initial_img = x
        initial_img = initial_img.astype(np.float32) / 255.0

        if not indices_provided:
            target_class = pseudorandom_target(img_index, 10, orig_class)
            mask = (y_test == target_class).flatten()
            x_test_target_class = x_test[mask]
            target_i = np.random.randint(0, x_test_target_class.shape[0])
            starting_img = x_test_target_class[target_i, None][0]
        else:
            starting_img = x_test[target_i, None][0]

        starting_img = starting_img.astype(np.float32) / 255.0

        adv = Adversarial(
            model=model,
            criterion=TargetClass(target_class=target_class),
            original_image=initial_img,
            original_class=orig_class,
            distance=Linf,
            threshold=0.05
        )

        try:
            attack = PerlinBoundaryAttack()
            attack(adv, starting_point=starting_img, iterations=args.num_iters, verbose=False, log_every_n_steps=1000, sample_gen=sample_gen,
                    normal_factor=args.normal_factor,
                    detection_transform=transform
                    # detection_transform=transform_brightness(0.7)
                    # spherical_step=0.3, source_step=0.3, step_adaptation=1.1
                    )

            #if i < 40:
            #    pred = np.argmax(kmodel.predict(np.expand_dims(adv.image, 0)), axis=-1)
            #    double_check.append((pred == target_class, pred, target_class))

            print("[detections]", len(attack.detector.get_detections()), np.mean(attack.detector.get_detections()))
            print("[detections]", adv.adversarial_class == adv.target_class())
            print("[distortions]", adv.distance, np.linalg.norm(adv.original_image - adv.image))
            dets.append(len(attack.detector.get_detections()))
            dists.append(np.mean(attack.detector.get_detections()))
            linf_distortions.append(adv.distance.value)
            l2_distortions.append(np.linalg.norm(adv.original_image - adv.image))
            successes.append(adv.adversarial_class == adv.target_class())
            # indices_targets.append((img_index, target_idx, orig_class, target_class))
        except (AssertionError, AttributeError) as e:
            # failures.append((img_index, target_idx, orig_class, target_class))
            errors.append(e)
            continue
np.savez_compressed(save_name, detections=dets, dists=dists, linf_distortions=linf_distortions, l2_distortions=l2_distortions, indices_targets=indices_targets, failures=failures)
    ### Extract Detection Results ###
    # print("DETECTIONS:")

def process(npz):
    x = np.load(npz)
    success_mask = x['linf_distortions'] <= 0.05
    failure_mask = x['linf_distortions'] > 0.05

    queries = x['detections'] * x['dists']
    detections = x['detections']
    # Success rate, l2 distortion, number of queries, median detections
    success_rate = success_mask.sum() / 100
    l2_distortion = x['l2_distortions'][success_mask]
    mean_queries, median_queries = np.mean(queries[success_mask]), np.median(queries[success_mask])
    mean_detections, median_detections = np.mean(detections[success_mask]), np.median(detections[success_mask])
    print("Success Rate", success_rate)
    print("l2_distortion", np.mean(l2_distortion), np.median(l2_distortion))
    print("mean, median queries:", mean_queries, median_queries)
    print("mean, median detections:", mean_detections, median_detections)

print("TIME:", time.time() - start_time)
print("[detections]", dets)
print("[dists]", dists)
print("[distortions]", l2_distortions, linf_distortions)
process(save_name)
# print("[successes]", successes)
# print("[indices]", indices_targets)
# print("[failures]", failures)
print("[errors]", errors)
# print("[double_check]", double_check)
    # print("[detections]", len(attack.detector.get_detections()), np.mean(attack.detector.get_detections()))
