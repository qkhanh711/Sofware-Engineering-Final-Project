import torch
from math import log2

START_TRAIN_AT_IMG_SIZE = 4
DATASET = '../../Celeba_dataset/img_align_celeba'
PATH_COLAB = '/content/Celeba_dataset/img_align_celeba'
CHECKPOINT_GEN = "weightTest/generator.pth"
CHECKPOINT_CRITIC = "weightTest/critic.pth"
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SAVE_MODEL = True
LOAD_MODEL = False
LEARNING_RATE = 1e-3
BATCH_SIZES = [32, 32, 32, 16, 16, 16, 16, 8, 4]
CHANNELS_IMG = 3
Z_DIM = 256  # 512 in paper
IN_CHANNELS = 256  # 512 in paper
CRITIC_ITERATIONS = 1
LAMBDA_GP = 10
PROGRESSIVE_EPOCHS = [30] * len(BATCH_SIZES)
FIXED_NOISE = torch.randn(8, Z_DIM, 1, 1).to(DEVICE)
NUM_WORKERS = 4
