**This paper has been accepted by The 40th Annual AAAI Conference on Aitificial Intelligence (AAAI 2026)**

```bibtex
@inproceedings{twinfuzz,
 author = {Dai, Enze and Mo, Wentao and Hu, Kun and Zhu, Xiaogang and Xiao, Xi and Wen, Sheng and Wang, Shaohua and Xiang, Yang}, 
 title = {TwinFuzz: Dual-Model Fuzzing for Robustness Generalization in Deep Learning},
 booktitle = {Proceedings of the 40th AAAI Conference on Artificial Intelligence (AAAI'26)},
 year = {2026},
 numpages = {9},
}
```

# TwinFuzz

TwinFuzz is an entropy-guided differential testing framework enhancing the robust generalization of defensible models.

## 0x01 Prerequisite
The project is developed based on Python 3.7 and Tensorflow 2.6.0, conda is recommended for enviroment isolation.

```bash
conda create -n py37-tf2-gpu python=3.7
conda activate py37-tf2-gpu
pip3 install -r requirements.txt
```

## 0x02 Train Models
Enter the folder of the corresponding dataset to train the models to be tested.

Take training LeNet4 under MNIST as an example.
```bash
cd MNIST
python train_models.py -m lenet4
```
Then the model is saved as *LeNet4_MNIST.h5*, following the naming rule as *{model_name}_{dataset}.h5*.

For PACS dataset, pretrained models are required.
```bash
cd PACS
python models.py -m pretrain_Resnet50
```

Next, the pre adversarial trained model for differential testing should be obtained.
```bash
mkdir checkpoint
python std_adv_train.py
```
*LeNet4_MNIST_Adv_12000.h5* in the checkpoint folder is the pre adversarial trained model, following the naming rule as *{model_name}_{dataset}\_Adv\_{advSample}.h5*.

## 0x03 Start Testing
Before testing, the config.ini should be configured.
```bash
cd diff_fuzz
vim config.ini
```
```ini
[model]
name = LeNet4
dataset = MNIST
advSample = 12000
duration = 300
```
*duration* represents the testing time. Now you are all set to start testing.
```bash
python fuzzing.py
```

## 0x04 Comparison with Previous Works
The following works are used for comparison
1. 📑 *RobOT: Robustness-Oriented Testing for Deep Learning Systems (ICSE)*
   - 🧑‍💻Code: https://github.com/Testing4AI/RobOT

2. 📑 *DLFuzz: differential fuzzing testing of deep learning systems (FSE)*
   - 🧑‍💻Code: https://github.com/turned2670/DLFuzz

3. 📑 *DeepXplore: automated whitebox testing of deep learning systems (SOSP)*
   - 🧑‍💻Code: https://github.com/peikexin9/deepxplore

