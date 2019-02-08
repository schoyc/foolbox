import keras
import numpy as np
import foolbox

from foolbox.attacks import BoundaryAttack

from foolbox.adversarial import Adversarial
from foolbox.criteria import TargetClass

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, GlobalAveragePooling2D, Activation, InputLayer
from keras.datasets import cifar10

CIFAR_10_WEIGHTS_PATH = "./keras_cifar10_trained_model_weights.h5"

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
        scalings = np.random.uniform(low=-C, high=C, size=(k,))
        scalings = np.reshape(scalings, (k,) + (1, 1, 1))
        return np.clip(x + scalings, a_min=0, a_max=1)

    return brightness


### Create Model ###

keras.backend.set_learning_phase(0)
kmodel = load_cifar_model()
model = foolbox.models.KerasModel(kmodel, bounds=(0, 1))


### Init Boundary Attack ###

# TODO: Targeted attack
(_, _), (x_test, y_test) = cifar10.load_data()
dets = []
dists = []
successes = []
for i in range(100):
    img_index = np.random.randint(0, x_test.shape[0])

    x, y = x_test[img_index, None][0], y_test[img_index][0]
    orig_class = y
    initial_img = x
    initial_img = initial_img.astype(np.float32) / 255.0

    target_class = pseudorandom_target(img_index, 10, orig_class)
    mask = (y_test == target_class).flatten()
    x_test_target_class = x_test[mask]
    target_i = np.random.randint(0, x_test_target_class.shape[0])
    starting_img = x_test_target_class[target_i, None][0]
    starting_img = starting_img.astype(np.float32) / 255.0

    adv = Adversarial(
        model=model,
        criterion=TargetClass(target_class=target_class),
        original_image=initial_img,
        original_class=orig_class,
        threshold=0.05 * 0.05 / (32 * 32 * 3)
    )

    try:
        attack = BoundaryAttack()
        attack(adv, starting_point=starting_img, iterations=100000, verbose=False, 
                # detection_transform=transform_brightness(0.7)
                # spherical_step=0.3, source_step=0.3, step_adaptation=1.1
                )
        print("[detections]", len(attack.detector.get_detections()), np.mean(attack.detector.get_detections()))
        print("[detections]", adv.adversarial_class == adv.target_class())
        dets.append(len(attack.detector.get_detections()))
        dists.append(np.mean(attack.detector.get_detections()))
        successes.append(adv.adversarial_class == adv.target_class())
    except (AssertionError, AttributeError) as e:
        continue

    ### Extract Detection Results ###
    # print("DETECTIONS:")
print("[detections]", dets)
print("[dists]", dists)
    # print("[detections]", len(attack.detector.get_detections()), np.mean(attack.detector.get_detections()))
