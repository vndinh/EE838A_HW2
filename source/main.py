import tensorflow as tf
import os
import shutil
from train import training
from valid import validate
from config import config
from test import testing

# Directories
# Validation
valid_ldr_dir = config.VALID.ldr_dir
valid_hdr_dir = config.VALID.hdr_dir
valid_gen_dir = config.VALID.gen_dir
logs_valid = config.VALID.logs_valid
# Test
test_ldr_dir = config.TEST.ldr_dir
test_gen_dir = config.TEST.gen_dir
logs_test = config.TEST.logs_test

# Parameters
valid_img_height = config.VALID.img_height
valid_img_width = config.VALID.img_width

if __name__ == '__main__':
  import argparse
  parser = argparse.ArgumentParser()
  parser.add_argument('--mode', type=str, default='train', help='Running process')
  parser.add_argument('--img_height', type=int, default=1080, help='Height of images')
  parser.add_argument('--img_width', type=int, default=1920, help='Width of images')
  args = parser.parse_args()
  if args.mode == 'train':
    training()
  elif args.mode == 'valid':
    validate(valid_ldr_dir, valid_hdr_dir, valid_gen_dir, logs_valid, valid_img_height, valid_img_width)
  elif args.mode == 'test':
    testing(test_ldr_dir, test_gen_dir, logs_test, args.img_height, args.img_width, 500)
  else:
    raise Exception("Unknown mode")


