import keras
import numpy as np
import sys
import foolbox

from foolbox.attacks import BoundaryAttack
from foolbox.attacks import PerlinBoundaryAttack

from foolbox.adversarial import Adversarial
from foolbox.criteria import TargetClass
from foolbox.sampling.sample_generator import SampleGenerator

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, GlobalAveragePooling2D, Activation, InputLayer
from keras.datasets import cifar10

import ast
import time

CIFAR_10_WEIGHTS_PATH = "./keras_cifar10_trained_model_weights.h5"
INDICES_FILE = "./indices_exp.txt"

def load_cifar_model():
    cifar10 = cifar10_logits()
    cifar10.load_weights(CIFAR_10_WEIGHTS_PATH, by_name=True)
    return cifar10


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

### Transforms ###
def transform_brightness(C):
    def brightness(x):
        k = x.shape[0]
        signs = np.random.choice([-1, 1], size=(k,))
        scalings = np.random.uniform(low=-C, high=C, size=(k,)) * signs
        scalings = np.reshape(scalings, (k,) + (1, 1, 1))
        return np.clip(x + scalings, a_min=0, a_max=1)

    return brightness

def get_test_model_correct(model):
    (_, _), (x_test, y_test) = cifar10.load_data()
    x_test_norm = x_test / 255.
    scores = model.predict(x_test_norm)
    preds = scores.argmax(axis=-1)
    model_correct = (preds == y_test.flatten())
    return x_test[model_correct], y_test[model_correct]

### Create Model ###

keras.backend.set_learning_phase(0)
kmodel = load_cifar_model()
model = foolbox.models.KerasModel(kmodel, bounds=(0, 1))


### Parse Args ###
args = sys.argv
save_name = str(args[1])

if len(args) >= 3:
    variable = str(args[2])
    param = float(args[3])
if len(args) >= 5:
    INDICES_FILE = str(args[4])

kwargs = {}

if variable == 'normal_factor':
    kwargs[variable] = param
elif variable == 'detection_transform':
    kwargs[variable] = transform_brightness(param)
    # kwargs['normal_factor'] = 1.0
else:
    # pass
    raise ValueError()
                    # detection_transform=transform_brightness(0.7)


### Init Boundary Attack ###

# TODO: Targeted attack
x_test, y_test = get_test_model_correct(kmodel)
x_test_idx = np.arange(x_test.shape[0])
dets = []
dists = []
successes = []
failures = []
errors = []
double_check = []

indices_provided = INDICES_FILE is not None

if indices_provided:
    f = open(INDICES_FILE, 'r')
    img_indices = ast.literal_eval(f.readline().strip())

indices_targets = []

img_shape = (32, 32, 3)
start_time = time.time()
with SampleGenerator(shape=img_shape, n_threads=1, queue_lengths=100) as sample_gen:
    for i in range(100):
        if indices_provided:
            img_index, target_idx, orig_class, target_class = img_indices[i]
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
            target_idx = x_test_idx[mask][target_i]
        else:
            starting_img = x_test[target_idx, None][0]

        starting_img = starting_img.astype(np.float32) / 255.0

        adv = Adversarial(
            model=model,
            criterion=TargetClass(target_class=target_class),
            original_image=initial_img,
            original_class=orig_class,
            threshold=0.05 * 0.05 / (32 * 32 * 3)
        )

        try:
            attack = PerlinBoundaryAttack()
            attack(adv, starting_point=starting_img, iterations=100000, verbose=False, sample_gen=sample_gen,
                    # **kwargs
                    normal_factor=0.0,
                    # detection_transform=transform_brightness(0.5)
                    # spherical_step=0.3, source_step=0.3, step_adaptation=1.1
                    )

            if i < 40:
                pred = np.argmax(kmodel.predict(np.expand_dims(adv.image, 0)), axis=-1)
                double_check.append((pred == target_class, pred, target_class))

            print("[detections]", len(attack.detector.get_detections()), np.mean(attack.detector.get_detections()))
            print("[detections]", adv.adversarial_class == adv.target_class())
            dets.append(len(attack.detector.get_detections()))
            dists.append(np.mean(attack.detector.get_detections()))
            successes.append(adv.adversarial_class == adv.target_class())
            indices_targets.append((img_index, target_idx, orig_class, target_class))
        except (AssertionError, AttributeError) as e:
            failures.append((img_index, target_idx, orig_class, target_class))
            errors.append(e)
            continue
np.savez_compressed(save_name, detections=dets, dists=dists, indices_targets=indices_targets, failures=failures)
    ### Extract Detection Results ###
    # print("DETECTIONS:")
print("TIME:", time.time() - start_time)
print("[detections]", dets)
print("[dists]", dists)
print("[successes]", successes)
print("[indices]", indices_targets)
print("[failures]", failures)
print("[errors]", errors)
print("[double_check]", double_check)
    # print("[detections]", len(attack.detector.get_detections()), np.mean(attack.detector.get_detections()))
