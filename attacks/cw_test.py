import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow import keras
from keras.utils import to_categorical





class CW2:
    """ C&W Attack (L2), changed from:
    https://github.com/tagomaru/ai_security/blob/
    Attributes:
        classifier (Model) : logits output
        k (float): hyperparameter change of k
        learning_rate (float): learning rate of the Adam Optimizer
        binary_search_steps (int): Binary Search Steps
        max_iterations (int): Max iteration numbers of the adversarial generation
        initial_c (float): initial constant value
    """

    def __init__(self, classifier, k = 0, learning_rate = 0.01,
                            binary_search_steps = 9, max_iterations = 1000,
                            initial_c = 0.001):

        # Import the corresponding classifier and define the self.variable
        self.classifier = classifier
        self.k = k
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.binary_search_steps = binary_search_steps
        self.initial_c = initial_c
            
    def is_satisfied_with_k(self, logits, target_class):
        """
        with k value increased, logits will basically increase
        「 k を加味したターゲットクラスの logit が最大」という制約を満たすか確認する。
        ターゲットクラスの logit から k　を引いた値が他のどのクラスの logit よりも大きければ　True を返し、
        そうでなければ、False を返す。
        Args:
            logits (ndarray): logits
            target_class (ndarray): target class
        Returns:
            satisfied (bool): whether k satisfy the objective.
        """
        logits = np.copy(logits)
        logits[target_class] -= self.k
        satisfied = np.argmax(logits) == target_class
        return satisfied    
    
    def generate(self, model, original_image, target_class_ohe):
        """
        Generate adversarial sample
        Args:
            original_image (ndarray): orginal shape -- (28, 28)
            target_class_ohe (ndarray): target class corresponding one-hot encoder (10, )
        Returns:
            o_best_adv_image (ndarray): adversarial class -- (28,28)
        """

        # set of the corresponding target class - argmax is used to find the best trusted class.
        target_class = np.argmax(target_class_ohe)

        # Define the orginal label
        original_label = model.predict(np.expand_dims(original_image, 0))
        original_label = np.argmax(original_label[0])
        
        # calcualte the shape of the original image
        shape = original_image.shape
               
        # constant and its corresponding limit
        c = self.initial_c
        c_lower = 0
        c_upper = 1e10

        #  (tanh(w) + 1) / 2 correspoinding w
        w = tf.Variable(np.zeros(shape, dtype=np.float32))
        
        # cast the corresponding tensor to the original image
        original_image = tf.cast(original_image, dtype=tf.float32)

        # generating adversarial samples
        def build_objective():

            # adversarial image generation
            adv_image = (tf.tanh(w) + 1) / 2

            # calculating objective
            objective1 = tf.reduce_sum(tf.square(adv_image - original_image))

            # calculating adversarial samples.logits
            logits = self.classifier(tf.expand_dims(adv_image, axis=0))[0]

            # calculating target classes logits
            target_logit = tf.reduce_sum(target_class_ohe * logits)
            
            # maximum logits other than target classes; logits
            other_max_logit = tf.reduce_max((1 - target_class_ohe) * logits + (target_class_ohe * np.min(logits)))
          
            # loss function 2
            objective2 = c * tf.maximum(0.0, other_max_logit - target_logit + self.k)

            # total loss function
            objective = objective1 + objective2

            return objective

        # loss function 
        objective = lambda: build_objective()
        
        
        o_best_objective1 = np.inf # initial value guess
        o_best_adv_image = np.zeros(shape) # initialize
        o_best_class = -1 # initialize
        loop_i = 0 # We also need to initialize the loop counter
        
        # for loop 
        for outer_step in range(self.binary_search_steps):
            
    
            best_objective1 = np.inf 
            best_class = -1 
            
            
            prev_objective = np.inf # initialize prev_objective
            
            # define adam optimizer
            opt = keras.optimizers.Adam(self.learning_rate)
            
            # Adam loop
            for iteration in range(self.max_iterations):
                

                opt.minimize(objective, var_list=[w])
                
                # calculate the logits of the adversarial samples
                adv_image = (tf.tanh(w) + 1) / 2
                logits = self.classifier(tf.expand_dims(adv_image, axis=0))[0]

                objective1 = tf.reduce_sum(tf.square(adv_image - original_image))

                # check every 10%
                if iteration % (self.max_iterations // 10) == 0:
                    if objective() > prev_objective * 0.9999:
                        break
                    prev_objective = objective()
                
                
                satisfied = self.is_satisfied_with_k(logits, target_class)

               
                if objective1 < best_objective1 and satisfied:
                    best_objective1 = objective1
                    best_class = target_class
                    
            
                if objective1 < o_best_objective1 and satisfied:
                    o_best_objective1 = objective1
                    o_best_class = target_class
                    o_best_adv_image = adv_image
                    loop_i = iteration # Here, we check the number of iteration to be able to output it

            
            if best_class == target_class:
                c_upper = c
                c = (c_lower + c_upper) / 2

            
            else:
                
                c_lower = c
                if c_upper < 1e9:                   
                    c = (c_lower + c_upper) / 2
                else:
                    c *= 10
            
            # print the output
            print('Binary Search {0}/{1}'.format(outer_step + 1, self.binary_search_steps))
            print('  L2 square: {0:.2f} - c: {1:.2f} - class: {2}'.format(o_best_objective1, c, o_best_class))
            print(f"original_label: {original_label}")
            print(f"adv_label: {np.argmax(self.classifier(tf.expand_dims(o_best_adv_image, axis=0)).numpy().flatten())}")


        # Calculate the total perturbation
        # r_tot = o_best_adv_image - original_image 
        # Generate adversarial label
        # adv_label = np.argmax(self.classifier(tf.expand_dims(o_best_adv_image, axis=0)).numpy().flatten()) 
        # ori_label is the original label
        # ori_label = original_label
        # # pert_image
        # pert_image = o_best_adv_image

        '''
        return value:
        r_tot (total perturbation); loop_i (number of iteration)
        ori_label (original label); adv_label (adversarial label); pert_image (adversarial image)
        '''

        return o_best_adv_image - original_image, loop_i, original_label, np.argmax(self.classifier(tf.expand_dims(o_best_adv_image, axis=0)).numpy().flatten()), o_best_adv_image.numpy()
    

# Check if a GPU is available
gpus = tf.config.list_physical_devices('GPU')

if gpus:
    print(f"GPUs detected: {len(gpus)}")
    for i, gpu in enumerate(gpus):
        print(f"GPU {i}: {gpu}")
else:
    print("No GPUs detected.")
model = keras.models.load_model('/ssd-sata1/mwt/def_project/DiffRobOT/DiffRobOT/MNIST/LeNet5_MNIST.h5')
model_logits = keras.models.load_model('/ssd-sata1/mwt/def_project/DiffRobOT/DiffRobOT/MNIST/LeNet5_MNIST_logits.h5')
_, (X_test, Y_test) = keras.datasets.mnist.load_data()
X_test = X_test/255.0
original_image = X_test[0]
print(original_image.shape)
# plt.axis('off')
# # plt.imshow(original_image, cmap=plt.cm.gray)
# # Save the image to the local directory
# plt.savefig('original_image.png', bbox_inches='tight', pad_inches=0)

Y_hat = model.predict(np.expand_dims(original_image, 0)) # 推論結果
original_class = np.argmax(Y_hat, axis=1)[0] # 分類結果
original_score = np.max(Y_hat, axis=1)[0] # スコア

# 分類結果とスコアを表示
print('Prediction: {0} - score {1:.2f}%'.format(original_class, original_score * 100))

logits = model_logits.predict(np.expand_dims(original_image, 0)) # logits
original_class_logits = np.argmax(logits, axis=1)[0] # 分類結果
original_logit = np.max(logits, axis=1)[0] # オリジナルクラスの logit

# 分類結果と logit を表示
print('Prediction: {0} - logit {1:.2f}'.format(original_class_logits, original_logit))



print("start generating adversarial ")
print("start generating adversarial ")
print("start generating adversarial ")

target_class_ohe = to_categorical(1, num_classes=10) # ターゲットクラスを 0 とする

attack = CW2(model_logits, k=0) # k = 0 で CW2 のインスタンスを生成


adv_image_k0 = attack.generate(model, original_image, target_class_ohe) # 敵対的サンプルを生成
print("original_label")
print(adv_image_k0[2])
print("adv_label")
print(adv_image_k0[3])
print("maximum iter")
print(adv_image_k0[1])








# 推論結果。 model_logits ではなく、model での推論である点に注意
Y_hat_k0  = model.predict(np.expand_dims(adv_image_k0[4], 0))

# スコア
score_k0 = np.max(Y_hat_k0[0])

# 分類結果
class_k0  = np.argmax(Y_hat_k0, axis = 1)[0]
print("class k0 is: ")
print(class_k0)
print("score_k0 is: ")
print(score_k0)

plt.axis('off')
plt.title("no")
plt.imshow(adv_image_k0[4], cmap=plt.cm.gray)
# Save the image to the local directory
plt.savefig('adv_image_k0.png', bbox_inches='tight', pad_inches=0)

# # L2
# L2_k0 = np.linalg.norm(adv_image_k0 - original_image)

# # 敵対的サンプルを表示
# plt.axis('off')
# plt.title('{0} - {1:.2f}% - {2:.2f}'.format(class_k0, score_k0 * 100, L2_k0))
# # plt.imshow(adv_image_k7, cmap=plt.cm.gray)
# # Save the image to the local directory
# plt.savefig('adv_image_k7.png', bbox_inches='tight', pad_inches=0)