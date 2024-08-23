import numpy as np
import tensorflow as tf

binary_search_step = 9  # Number of times to adjust the constant with binary search
max_iter= 1000          # Number of iterations to perform gradient descent
abort_early = True      # If we stop improving, abort gradient descent early
lr = 1e-2               # Larger values converge faster to less accurate results
targeted = False        # Should we target one specific class? or just be wrong?
conf = 0                # How strong the adversarial example should be
init_const = 1e-3       # The initial constant c to pick as a first guess


# Carlini Wagner Attack, L_2
class CwL2:
    def __init__(self, model, batch_size=1, confidence=conf, targeted=targeted, learning_rate=lr, binary_search_steps=binary_search_step, max_iterations=max_iter, abort_early=abort_early, initial_const=init_const, boxmin=-0.5, boxmax=0.5):
        # image_size, num_channels, num_labels = model.image_size, model.num_channels, model.num_labels
        image_size, num_channels, num_labels = 28, 1, 10
        self.TARGETED = targeted
        self.LEARNING_RATE = learning_rate
        self.MAX_ITERATIONS = max_iterations
        self.BINARY_SEARCH_STEPS = binary_search_steps
        self.ABORT_EARLY = abort_early
        self.CONFIDENCE = confidence
        self.initial_const = initial_const
        self.batch_size = batch_size

        self.repeat = binary_search_steps >= 10

        self.I_KNOW_WHAT_I_AM_DOING_AND_WANT_TO_OVERRIDE_THE_PRESOFTMAX_CHECK = False

        self.shape = (batch_size, image_size, image_size, num_channels)
        self.boxmul = (boxmax - boxmin) / 2.
        self.boxplus = (boxmin + boxmax) / 2.
        self.model = model

    def attack(self, imgs, targets):
        """
        If self.targeted is true, then the targets represents the target labels.
        If self.targeted is false, then targets are the original class labels.
        """
        adv_res = []
        print('go up to', len(imgs))
        for i in range(0, len(imgs), self.batch_size):
            print('tick', i)
            # print("imgs[i:i + self.batch_size]", imgs[i:i + self.batch_size])
            # print("targets", targets)
            adv_res.extend(self.attack_batch(
                imgs[i:i + self.batch_size], targets))
        return np.array(adv_res)

    def attack_batch(self, imgs, labels):
        """
        Run the attack on a batch of images and labels.
        """
        # print("imgs, labs in attack_batch", imgs, labs) #shape=(1, 28, 28, 1), dtype=float32) [array([0., 0., 0., 0., 0., 0., 1., 0., 0., 0.])]
        batch_size = self.batch_size
        
        # whether attack success
        def check_success(x, y):
            if not isinstance(x, (float, int, np.int64)):
                x = x.numpy()
                x = np.copy(x)
                if self.TARGETED:
                    x[y] -= self.CONFIDENCE
                else:
                    x[y] += self.CONFIDENCE
                x = np.argmax(x)
            if self.TARGETED:
                return x == y
            else:
                return x != y

        # @tf.function
        def train_step(modifier, timg, tlab, const):
            with tf.GradientTape() as tape:
                newimg = tf.tanh(modifier + timg) * self.boxmul + self.boxplus
                output = self.model(newimg)
                output = tf.cast(output, dtype=tf.float32)
                l2dist = tf.reduce_sum(
                    tf.square(newimg - (tf.tanh(timg) * self.boxmul + self.boxplus)), [1, 2, 3])
                real = tf.math.reduce_sum((tlab) * output, 1)
                other = tf.math.reduce_max(
                    (1 - tlab) * output - (tlab * 10000), 1)
                if self.TARGETED:
                    # if targetted, optimize for making the other class most likely
                    loss1 = tf.maximum(0.0, other - real + self.CONFIDENCE)
                else:
                    # if untargeted, optimize for making this class least likely.
                    loss1 = tf.maximum(0.0, real - other + self.CONFIDENCE)

                loss2 = tf.reduce_sum(l2dist)
                loss1 = tf.reduce_sum(const * loss1)

                loss = loss1 + loss2
            optimizer = tf.optimizers.Adam(self.LEARNING_RATE)
            loss_metric = tf.keras.metrics.Mean(name='train_loss')

            grads = tape.gradient(loss, [modifier])
            optimizer.apply_gradients(zip(grads, [modifier]))
            loss_metric.update_state(loss)
            return loss, l2dist, output, newimg, loss1, loss2

        # convert to tanh-space
        imgs = np.arctanh((imgs - self.boxplus) / self.boxmul * 0.999999)
        # print(np.shape(imgs))
        lower_bound = np.zeros(batch_size)
        CONST = np.ones(batch_size) * self.initial_const
        upper_bound = np.ones(batch_size) * 1e10

        # the best l2, score, and image attack
        o_bestl2 = [1e10] * batch_size
        o_bestscore = [-1] * batch_size
        o_bestattack = [np.zeros(imgs[0].shape)] * batch_size
        print(np.shape(o_bestattack), "np.shape(o_bestattack)")  # (1, 28, 28, 1)

        for outer_step in range(self.BINARY_SEARCH_STEPS):
            batch = tf.Variable(imgs[:batch_size], dtype=tf.float32)
            batchlab = tf.Variable(labels[:batch_size], dtype=tf.float32)
            bestl2 = [1e10] * batch_size
            bestscore = [-1] * batch_size
            if self.repeat == True and outer_step == self.BINARY_SEARCH_STEPS - 1:
                CONST = upper_bound

            modifier = tf.Variable(np.zeros((1, 28, 28, 1), dtype=np.float32))
            const = tf.Variable(CONST, dtype=tf.float32)
            prev = np.inf
            for iteration in range(self.MAX_ITERATIONS):
                # perform the attack

                l, l2s, scores, nimg, loss1, loss2 = train_step(modifier, batch, batchlab, const)

                if np.all(scores >= -.0001) and np.all(scores <= 1.0001):
                    if np.allclose(np.sum(scores, axis=1), 1.0, atol=1e-3):
                        if not self.I_KNOW_WHAT_I_AM_DOING_AND_WANT_TO_OVERRIDE_THE_PRESOFTMAX_CHECK:
                            raise Exception("The output of model.predict should return the pre-softmax layer. It looks like you are returning the probability vector (post-softmax). If you are sure you want to do that, set attack.I_KNOW_WHAT_I_AM_DOING_AND_WANT_TO_OVERRIDE_THE_PRESOFTMAX_CHECK = True")

                if iteration % (self.MAX_ITERATIONS // 10) == 0:
                    print(iteration, l, loss1, loss2)
                # check if we should abort search if we're getting nowhere.
                if self.ABORT_EARLY and iteration % (self.MAX_ITERATIONS // 10) == 0:
                    if l > prev * .9999:
                        break
                    prev = l
                # adjust the best result found so far
                for e, (l2, sc, ii) in enumerate(zip(l2s, scores, nimg)):
                    # print("batchlab", np.argmax(batchlab[e]))
                    # print("(sc, np.argmax(batchlab))", sc, np.argmax(sc))
                    # print("l2 and bestl2[e]", l2, bestl2[e])
                    # print("compare(sc, tf.argmax(batchlab))",
                    #       compare(sc, tf.argmax(batchlab[e])))
                    if l2 < bestl2[e] and check_success(sc, np.argmax(batchlab[e])):
                        bestl2[e] = l2
                        bestscore[e] = np.argmax(sc)
                    if l2 < o_bestl2[e] and check_success(sc, np.argmax(batchlab[e])):
                        o_bestl2[e] = l2
                        o_bestscore[e] = np.argmax(sc)
                        o_bestattack[e] = ii

            # adjust the constant as needed
            for e in range(batch_size):
                print("bestscore[e]", bestscore[e])
                if check_success(bestscore[e], np.argmax(batchlab[e])) and bestscore[e] != -1:
                    # success, divide const by two
                    upper_bound[e] = min(upper_bound[e], CONST[e])
                    if upper_bound[e] < 1e9:
                        CONST[e] = (lower_bound[e] + upper_bound[e]) / 2
                else:
                    # failure, either multiply by 10 if no solution found yet
                    #          or do binary search with the known upper bound
                    lower_bound[e] = max(lower_bound[e], CONST[e])
                    if upper_bound[e] < 1e9:
                        CONST[e] = (lower_bound[e] + upper_bound[e]) / 2
                    else:
                        CONST[e] *= 10
        o_bestl2 = np.array(o_bestl2)
        return o_bestattack