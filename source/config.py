from easydict import EasyDict as edict

config = edict()
config.TRAIN = edict()
config.VALID = edict()
config.TEST = edict()

# Training
config.TRAIN.logs_dir = '..\\logs'
config.TRAIN.logs_train = '..\\logs\\logs_train.txt'
config.TRAIN.hdr_dir = '..\\data\\train\\HDR'
config.TRAIN.model_dir = '..\\model\\model.ckpt'

# Validation
config.VALID.logs_valid = '..\\logs\\logs_valid.txt'
config.VALID.ldr_dir = '..\\data\\valid\\LDR'
config.VALID.hdr_dir = '..\\data\\valid\\HDR'
config.VALID.gen_dir = '..\\report\\valid_result'
config.VALID.img_height = 1080
config.VALID.img_width = 1920
config.VALID.gamma = 0.45

# Test
config.TEST.logs_test = '..\\logs\\logs_test.txt'
config.TEST.ldr_dir = '..\\data\\test\\LDR'
config.TEST.gen_dir = '..\\report\\test_result'

# Hyper parameters
config.TRAIN.num_epoches = 1000
config.TRAIN.patch_size = 320
config.TRAIN.batch_size = 4
config.TRAIN.lr_init = 5*1e-5
config.TRAIN.lr_decay = 0.5
config.TRAIN.lr_decay_period = 100

# Parameters
config.TRAIN.eps = 1.0/255.0
config.TRAIN.lambda_ir = 0.5
