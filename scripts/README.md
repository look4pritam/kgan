# AttGAN on GCP or AWS

## Clone kgan repository.

```sh
cd ~

git clone https://github.com/look4pritam/kgan.git

cd /home/ubuntu/kgan
```

## Install NVIDIA driver.

```sh
./scripts/ubuntu-18.04/install_nvidia_driver
```

### Verify NVIDIA driver installation.

```sh
nvidia-smi
```

## Install CUDA-10.1.

```sh
./scripts/ubuntu-18.04/install_cuda-10.1
```

### Verify CUDA-10.1 installation.

```sh
nvcc -V
```

## Install Python3.

```sh
./scripts/ubuntu-18.04/install_python3
```

## Install TensorFlow-2.2.0-GPU.

```sh
./scripts/ubuntu-18.04/install_tensorflow-2.2.0
```

## Download datasets.

```sh
./scripts/datasets/download_dataset
```

## Train the model.

```sh
cd /home/ubuntu/kgan
```

```sh
nohup python3 kgan/train_model.py --model attgan  --dataset celeba --model_shape 128 128 3 --latent_dimension 40 --learning_rate 0.0002 --batch_size 32 --maximum_epochs 200 --start_epoch 0 --discriminator_number 5 --generator_number 1 --save_frequency 1 --loss_scan_frequency 1000 &
```



