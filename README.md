# eecs 598 project
Explore Image and feature space wiener deconvolution network for non-blind image deblurring  with Richardson-Lucy deconvolution algorithm.

Our experiments mainly follows the codes provided by https://github.com/zhihongz/INFWIDE.
The `srcs` contains the definition of the model, and under the model folder, there's our implemented Richardson-Lucy deconvolution model (rl_filter.py, and the current was only able to deal with input of batch size 1). The `conf` folder contains the configurations for running the experiment, where infwide_test.yaml and infwide_train.yaml describes the basic configurations for testing and training separately. And under its data folder, one can configure where is the dataset for training or testing.
The dataset we used was also from INFWIDE, https://drive.google.com/file/d/1woLLJU1RxsXehXOZpCRTX-1ep8IxIbTP/view?usp=sharing.
The following describes our changes and how to run the codes.

Our trained models can be found at:
https://drive.google.com/drive/folders/1r8pQDsZYVIjWcyDBysUAPGrTw0bJqSUk?usp=sharing

To run an simulation experiment:
- create folder `dataset/`, and put the input image folder under it. Configure the input data folder in `conf/data/infwide_test_data.yaml`
- create folder `model_zoo/`, and put the trained model checkpoint under it. Configure the used chekpoint in `conf/infwide_test.yaml`
- more configures for different types of experiments: in `conf/data/infwide_test_data.yaml` select different noise_type, where the none type is modified by us to run experiment without adding additional noise.
- also in the `srcs/model/infwide_model.py`, one may comment/uncomment the lines for image branch in order to use Richardson-Lucy or Wiener deconvolution module according the checkpoint one is using.
- then run the code `test.py`

To train the model:
- create folder `dataset/` folder, and put the training image folder under it. Configure the input data folder in `conf/data/infwide_train_data.yaml`
- in code `conf/infwide_train.yaml`, one can configure whether to train from scratch or train from some checkpoint.
- in the code `srcs/model/infwide_model.py`, one may comment/uncomment the lines for image branch in order to use Richardson-Lucy or Wiener deconvolution module according the checkpoint one is using.
- then run the code `train.py`

For the more detailed setups and code structure, the reader can referred to the description in INFWIDE.
