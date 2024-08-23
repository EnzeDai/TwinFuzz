import numpy as np
import tensorflow as tf

decrease_factor = 0.9   # Range in (0, 1), rate where we shrink tau
max_iter = 100          # The number of maximum iterations to try
abort_early = True      # Abort when find the valid solution
lr = 1e-2               # Learning rate
init_const = 1e-3       # First value of constant 'c'
max_const = 2e+1        # The max value of 'c' to climb before giving up
reduce_const = False    # Try to lower c each iteration; faster to set to false
targeted = False        # Target or untarget attack
const_factor = 2.0      # Should > 1, where we increase constant

# Carlini Wagner Attack, L_infinite, from 2017 S&P
class CwLinf:
    def __init__(self, model, targeted = targeted, lr = lr, max_iterations = max_iter, abort_early = abort_early, initial_const = init_const,
                 largest_const = max_const, reduce_const = reduce_const, decrease_factor = decrease_factor, const_factor = const_factor):
        
        self.model = model
        self.TARGETED = targeted
        self.lr = lr
        self.MAX_ITERATIONS = max_iterations
        self.ABORT_EARLY = abort_early
        self.INITIAL_CONST = initial_const
        self.LARGEST_CONST = largest_const
        self.DECREASE_FACTOR = decrease_factor
        self.REDUCE_CONST = reduce_const
        self.const_factor = const_factor

        self.I_KNOW_WHAT_I_AM_DOING_AND_WANT_TO_OVERRIDE_THE_PRESOFTMAX_CHECK = False
        
        self.grad = self.gradient_descent(model)


    def gradient_descent(self, model):
        # Set constant
        image_size, num_channels, num_labels = 28, 1, 10

        # whether attack success
        def check_success(x, y):
            if self.TARGETED:
                return x == y
            else:
                return x != y
        
        shape = (1, image_size, image_size, num_channels)

        # the variable to optimize over
        modifier = tf.Variable(np.zeros(shape,dtype=np.float32))
        
        tau = tf.Variable(dtype=tf.float32)
        simg = tf.Variable(tf.float32, shape)
        timg = tf.Variable(tf.float32, shape)
        tlab = tf.Variable(tf.float32, (1, num_labels))
        const = tf.Variable(tf.float32, [])
        
        newimg = (tf.tanh(modifier + simg)/2)
        
        output = model.predict(newimg)
        orig_output = model.predict(tf.tanh(timg)/2)

        real = tf.reduce_sum((tlab)*output)
        other = tf.reduce_max((1-tlab)*output - (tlab*10000))

        if self.TARGETED:
            # if targeted attack, optimize for making the other class most likely
            loss1 = tf.maximum(0.0, other-real)
        else:
            # if untargeted attack, optimize for making this class least likely.
            loss1 = tf.maximum(0.0, real-other)

        # sum up the losses
        loss2 = tf.reduce_sum(tf.maximum(0.0,tf.abs(newimg-tf.tanh(timg)/2)-tau))
        loss = const * loss1 + loss2

        # setup the adam optimizer and keep track of variables
        optimizer = tf.keras.optimizers.Adam(self.lr)

        # @tf.function
        def train_step():
            with tf.GradientTape() as tape:
                current_loss = loss
            gradients = tape.gradient(current_loss, [modifier])
            optimizer.apply_gradients(zip(gradients, [modifier]))
            return loss, output
        

        def doit(oimgs, labs, starts, tt, CONST):
            # convert to tanh-space
            imgs = np.arctanh(np.array(oimgs)*1.999999)
            starts = np.arctanh(np.array(starts)*1.999999)
    
            # initialize the variables
            modifier.assign(np.zeros(shape, dtype=np.float32))
            tau.assign(tt)
            const.assign(CONST)

            while CONST < self.LARGEST_CONST:
                # try solving for each value of the constant
                print('try const', CONST)
                for step in range(self.MAX_ITERATIONS):
                    feed_dict={
                        timg: imgs, 
                        tlab:labs, 
                        tau: tt,
                        simg: starts,
                        const: CONST
                    }
                    
                    if step % (self.MAX_ITERATIONS // 10) == 0:
                        loss_value, loss1_value, loss2_value = train_step()
                        print(step, loss_value.numpy(), loss1_value.numpy(), loss2_value.numpy())

                    # perform the update step
                    loss_value, scores = train_step()

                    if np.all(scores>=-.0001) and np.all(scores <= 1.0001):
                        if np.allclose(np.sum(scores,axis=1), 1.0, atol=1e-3):
                            if not self.I_KNOW_WHAT_I_AM_DOING_AND_WANT_TO_OVERRIDE_THE_PRESOFTMAX_CHECK:
                                raise Exception("The output of model.predict should return the pre-softmax layer. It looks like you are returning the probability vector (post-softmax). If you are sure you want to do that, set attack.I_KNOW_WHAT_I_AM_DOING_AND_WANT_TO_OVERRIDE_THE_PRESOFTMAX_CHECK = True")

                    
                    # it worked
                    if loss_value < 0.0001 * CONST and self.ABORT_EARLY:
                        get = self.model(tf.tanh(modifier + simg) / 2, training=False)
                        works = check_success(np.argmax(get), np.argmax(labs))
                        if works:
                            scores = self.model(tf.tanh(modifier + simg) / 2, training=False)
                            origscores = self.model(tf.tanh(timg) / 2, training=False)
                            nimg = tf.tanh(modifier + simg) / 2
                            l2s = np.square(nimg - np.tanh(imgs) / 2).numpy().sum(axis=(1, 2, 3))
                            
                            return scores, origscores, nimg, CONST

                # we didn't succeed, increase constant and try again
                CONST *= self.const_factor
    
        return doit


    def attack(self, imgs, targets):
        adv_res = []

        for img,target in zip(imgs, targets):
            adv_res.extend(self.attack_single(img, target))
        
        return np.array(adv_res)
    

    def attack_single(self, img, target):

        # the previous image
        prev = np.copy(img).reshape((1,self.model.image_size,self.model.image_size,self.model.num_channels))
        tau = 1.0
        const = self.INITIAL_CONST
        
        while tau > 1./256:
            # try to solve given this tau value
            res = self.grad([np.copy(img)], [target], np.copy(prev), tau, const)
            if res == None:
                # the attack failed, we return this as our final answer
                return prev
    
            scores, origscores, nimg, const = res
            if self.REDUCE_CONST: const /= 2

            # the attack succeeded, reduce tau and try again
    
            actualtau = np.max(np.abs(nimg-img))
    
            if actualtau < tau:
                tau = actualtau
    
            print("Tau",tau)

            prev = nimg
            tau *= self.DECREASE_FACTOR
        return prev