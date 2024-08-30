"""

Below is the tensorflow code for the The BasicIterativeMethod attack.


"""
import numpy as np
import tensorflow as tf
from utils_attack import optimize_linear, compute_gradient, clip_eta, random_lp_vector


def fgsm_atk(
    model_fn,
    x,
    eps,
    norm,
    loss_fn=None,
    clip_min=None,
    clip_max=None,
    y=None,
    targeted=False,
    sanity_checks=False,
):
    """
    Tensorflow 2.0 implementation of the Fast Gradient Method.
    :param model_fn: a callable that takes an input tensor and returns the model logits.
    :param x: input tensor.
    :param eps: epsilon (input variation parameter).
    :param norm: Order of the norm (mimics NumPy). Possible values: np.inf, 1 or 2.
    :param loss_fn: (optional) callable. Loss function that takes (labels, logits) as arguments and returns loss.
                    default function is 'tf.nn.sparse_softmax_cross_entropy_with_logits'
    :param clip_min: (optional) float. Minimum float value for adversarial example components.
    :param clip_max: (optional) float. Maximum float value for adversarial example components.
    :param y: (optional) Tensor with true labels. If targeted is true, then provide the
              target label. Otherwise, only provide this parameter if you'd like to use true
              labels when crafting adversarial samples. Otherwise, model predictions are used
              as labels to avoid the "label leaking" effect . Default is None.
    :param targeted: (optional) bool. Is the attack targeted or untargeted?
              Untargeted, the default, will try to make the label incorrect.
              Targeted will instead try to move in the direction of being more like y.
    :param sanity_checks: bool, if True, include asserts (Turn them off to use less runtime /
              memory or for unit tests that intentionally pass strange input)
    :return: a tensor for the adversarial example
    """
    if norm not in [np.inf, 1, 2]:
        raise ValueError("Norm order must be either np.inf, 1, or 2.")

    if loss_fn is None:
        loss_fn = tf.nn.sparse_softmax_cross_entropy_with_logits

    asserts = []

    # If a data range was specified, check that the input was in that range
    if clip_min is not None:
        asserts.append(tf.math.greater_equal(x, clip_min))

    if clip_max is not None:
        asserts.append(tf.math.less_equal(x, clip_max))

    # cast to tensor if provided as numpy array
    x = tf.cast(x, tf.float32)

    if y is None:
        # Using model predictions as ground truth to avoid label leaking
        y = tf.argmax(model_fn(x), 1)

    grad = compute_gradient(model_fn, loss_fn, x, y, targeted)

    optimal_perturbation = optimize_linear(grad, eps, norm)
    # Add perturbation to original example to obtain adversarial example
    adv_x = x + optimal_perturbation

    # If clipping is needed, reset all values outside of [clip_min, clip_max]
    if (clip_min is not None) or (clip_max is not None):
        # We don't currently support one-sided clipping
        assert clip_min is not None and clip_max is not None
        adv_x = tf.clip_by_value(adv_x, clip_min, clip_max)

    if sanity_checks:
        assert np.all(asserts)
    return adv_x


def pgd_atk(
    model_fn,
    x,
    eps,
    eps_iter,
    nb_iter,
    norm,
    loss_fn=None,
    clip_min=None,
    clip_max=None,
    y=None,
    targeted=False,
    rand_init=None,
    rand_minmax=None,
    sanity_checks=False,
):
    """
    :param model_fn: a callable that takes an input tensor and returns the model logits.
    :param x: input tensor.
    :param eps: epsilon (input variation parameter);
    :param eps_iter: step size for each attack iteration
    :param nb_iter: Number of attack iterations.
    :param norm: Order of the norm (mimics NumPy). Possible values: np.inf, 1 or 2.
    :param loss_fn: (optional) callable. loss function that takes (labels, logits) as arguments and returns loss.
                    default function is 'tf.nn.sparse_softmax_cross_entropy_with_logits'
    :param clip_min: (optional) float. Minimum float value for adversarial example components.
    :param clip_max: (optional) float. Maximum float value for adversarial example components.
    :param y: (optional) Tensor with true labels. If targeted is true, then provide the
              target label. Otherwise, only provide this parameter if you'd like to use true
              labels when crafting adversarial samples.
    :param targeted: (optional) bool. Is the attack targeted or untargeted?
              Untargeted, the default, will try to make the label incorrect.
              Targeted will instead try to move in the direction of being more like y.
    :param rand_init: (optional) float. Start the gradient descent from a point chosen
                        uniformly at random in the norm ball of radius
                        rand_init_eps
    :param rand_minmax: (optional) float. Size of the norm ball from which
                        the initial starting point is chosen. Defaults to eps
    :param sanity_checks: bool, if True, include asserts (Turn them off to use less runtime /
              memory or for unit tests that intentionally pass strange input)
    :return: a tensor for the adversarial example
    """

    assert eps_iter <= eps, (eps_iter, eps)
    if norm == 1:
        raise NotImplementedError(
            "It's not clear that FGM is a good inner loop"
            " step for PGD when norm=1, because norm=1 FGM "
            " changes only one pixel at a time. We need "
            " to rigorously test a strong norm=1 PGD "
            "before enabling this feature."
        )
    if norm not in [np.inf, 2]:
        raise ValueError("Norm order must be either np.inf or 2.")

    if loss_fn is None:
        loss_fn = tf.nn.sparse_softmax_cross_entropy_with_logits

    asserts = []

    # If a data range was specified, check that the input was in that range
    if clip_min is not None:
        asserts.append(tf.math.greater_equal(x, clip_min))

    if clip_max is not None:
        asserts.append(tf.math.less_equal(x, clip_max))

    # Initialize loop variables
    if rand_minmax is None:
        rand_minmax = eps

    if rand_init:
        eta = random_lp_vector(
            tf.shape(x), norm, tf.cast(rand_minmax, x.dtype), dtype=x.dtype
        )
    else:
        eta = tf.zeros_like(x)

    # Clip eta
    eta = clip_eta(eta, norm, eps)
    adv_x = x + eta
    if clip_min is not None or clip_max is not None:
        adv_x = tf.clip_by_value(adv_x, clip_min, clip_max)

    if y is None:
        # Using model predictions as ground truth to avoid label leaking
        y = tf.argmax(model_fn(x), 1)

    i = 0
    while i < nb_iter:
        adv_x = fgsm_atk(
            model_fn,
            adv_x,
            eps_iter,
            norm,
            loss_fn,
            clip_min=clip_min,
            clip_max=clip_max,
            y=y,
            targeted=targeted,
        )

        # Clipping perturbation eta to norm norm ball
        eta = adv_x - x
        eta = clip_eta(eta, norm, eps)
        adv_x = x + eta

        # Redo the clipping.
        # FGM already did it, but subtracting and re-adding eta can add some
        # small numerical error.
        if clip_min is not None or clip_max is not None:
            adv_x = tf.clip_by_value(adv_x, clip_min, clip_max)
        i += 1

    asserts.append(eps_iter <= eps)
    if norm == np.inf and clip_min is not None:
        # TODO necessary to cast to x.dtype?
        asserts.append(eps + clip_min <= clip_max)

    if sanity_checks:
        assert np.all(asserts)
    return adv_x


def basic_iterative_method(
    model_fn,
    x,
    eps,
    eps_iter,
    nb_iter,
    norm,
    clip_min=None,
    clip_max=None,
    y=None,
    targeted=False,
    rand_init=None,
    rand_minmax=0.3,
    sanity_checks=True,
):
    """
    The BasicIterativeMethod attack.
    """
    return pgd_atk(
        model_fn,
        x,
        eps,
        eps_iter,
        nb_iter,
        norm,
        clip_min=clip_min,
        clip_max=clip_max,
        y=y,
        targeted=targeted,
        rand_init=False,
        rand_minmax=rand_minmax,
        sanity_checks=sanity_checks,
    )


# Below are the test code：
# Load the MNIST dataset and the model
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_test = np.expand_dims(x_test, axis = -1).astype(np.float32) / 255 # randomize the input dataset

# Load the pre-trained model
mnist_model_logits = tf.keras.models.load_model("/ssd-sata1/mwt/def_project/DiffRobOT/MNIST/LeNet5_MNIST_logits.h5")
# mnist_model_logits = tf.keras.models.load_model("/ssd-sata1/mwt/def_project/DiffRobOT/MNIST/LeNet5_MNIST_normal.h5")

# Define a single MNIST image for testing
test_image = x_test[10]  # Use the first test image
test_label = y_test[10]  # Use the corresponding label

print(test_label)

# Run the BIM attack
adv_image = basic_iterative_method(
    model_fn=mnist_model_logits,
    x=tf.convert_to_tensor(np.expand_dims(test_image, axis=0)),  # Add batch dimension
    # x = test_image,
    eps=0.4,
    eps_iter=0.08,
    nb_iter=20,
    norm=np.inf,
    clip_min=0.0,
    clip_max=1.0,
    y=tf.convert_to_tensor([test_label]),  
    #y = test_label,
    targeted=False,
    rand_init=False,
    rand_minmax=0.3, #Initialize
    sanity_checks=False,
)



# Print out the original label
mnist_model = tf.keras.models.load_model("/ssd-sata1/mwt/def_project/DiffRobOT/MNIST/LeNet5_MNIST_normal.h5")

# Print out the total perturbation in L2 Norm
total_perturbation = np.linalg.norm(adv_image - test_image)

# Get original label
original_label = np.argmax(mnist_model.predict(np.expand_dims(test_image, axis = 0)))

# Get adversarial label
adv_label = np.argmax(mnist_model.predict(adv_image))

adv_image = tf.squeeze(adv_image, axis =-1) # Remove the last channel
print(adv_image)

# Output results
print(f"Original Label: {original_label}")
print(f"Adversarial Label: {adv_label}")
print(f"Total Perturbation (L2 norm): {total_perturbation}")
print(f"Total Iterations: {20}")  # The number of iterations is specified by `nb_iter'.