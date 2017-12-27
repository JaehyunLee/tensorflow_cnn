# CNN using Tensorflow
This is code for labs covered in CNN TensorFlow using MNIST data

## Version
* Python (3.6.3)
* tensorflow (1.4.1)
* tensorflow-tensorboard (0.4.0rc3)
* numpy (1.13.3)

## Training Data
This code makes two files:
* check point file(.ckpt) for saving train result
* log file for using tensorboard
```bash
python mnist_cnn_train.py
```

## Predict Data
This code restores train result and checks accuracy
```bash
python mnist_cnn_predict.py
```

## Reference Implementations
* https://github.com/hunkim/DeepLearningZeroToAll/
* https://github.com/bwcho75/tensorflowML/
* https://github.com/bwcho75/facerecognition/
