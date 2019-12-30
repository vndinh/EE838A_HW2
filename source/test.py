import tensorflow as tf
import time
import numpy as np
import imageio
import random

from hdr_cnn import hdrcnn, get_final_hdr
from config import config
from utils import get_filepath, write_logs, img_write, alpha_msk, reinhard02

# Directories
model_dir = config.TRAIN.model_dir

# Parameters
lambda_ir = config.TRAIN.lambda_ir
gamma = config.VALID.gamma

def test_parse(ldr_dir):
  ldr_string = tf.read_file(ldr_dir)
  ldr_img = tf.image.decode_png(ldr_string, channels=3)
  ldr_img = tf.image.convert_image_dtype(ldr_img, tf.float32)
  return ldr_img

def testing(ldr_dir, gen_dir, logs_dir, img_height, img_width, Hth):
  X = tf.placeholder(tf.float32, [1, img_height , img_width, 3])
  Y = tf.placeholder(tf.float32, [1, img_height, img_width, 3])
  
  test_ldr_path, test_ldr_name = get_filepath(ldr_dir, '.png')
  num_test = len(test_ldr_path)
  test_ldr_path = tf.constant(test_ldr_path)
  
  # Data loader
  dataset = tf.data.Dataset.from_tensor_slices(test_ldr_path)
  dataset = dataset.map(test_parse, num_parallel_calls=4)
  #dataset = dataset.batch(1)
  iter = dataset.make_one_shot_iterator()
  ldr_img = iter.get_next()

  alpha = alpha_msk(X)

  # Prediction
  with tf.name_scope('HDR_CNN'):
    hdr_nn = hdrcnn(X, is_training=False, reuse=False)
  hdr_final = get_final_hdr(X, hdr_nn)

  saver = tf.train.Saver(tf.global_variables())

  with tf.Session() as sess:
    # Initialize variables
    sess.run(tf.global_variables_initializer())
    
    # Restore weights of model
    saver.restore(sess, model_dir)
    
    # Validation
    log = "\n========== Test Begin ==========\n"
    write_logs(logs_dir, log, True)
    test_start = time.time()
    avg_irloss = 0
    avg_dirloss = 0
    for f in test_ldr_name:
      test_img_start = time.time()
      ldr_image = sess.run([ldr_img])
      alpha_val, hdr_pred = sess.run([alpha, hdr_final], feed_dict={X:ldr_image})

      f1, _ = f.split("_")
      img_write(gen_dir, 'alpha_'+f1+'_HDR.png', alpha_val, 'PNG-FI')

      # Gamma correction
      hdr_pred_save = np.multiply(Hth, np.maximum(hdr_pred, 0.0))
      img_write(gen_dir, 'pred_'+f1+'_HDR.hdr', hdr_pred_save, 'HDR-FI')

      # Tone mapping
      hdr_pred_gamma = np.power(np.maximum(hdr_pred, 0.0), gamma)
      ldr_tone = reinhard02(hdr_pred_gamma, a=0.18)
      img_write(gen_dir, 'tm_'+f1+'_HDR.png', ldr_tone, 'PNG-FI')
    
      log = "Image {}, Time {:2.5f}, Shape = {}".format(f, time.time()-test_img_start, hdr_pred.shape)
      write_logs(logs_dir, log, False)
    log = "\nTest Time: {:2.5f}".format(time.time()-test_start)
    write_logs(logs_dir, log, False)
    log = "\n========== Test End ==========\n"
    write_logs(logs_dir, log, False)
    
    sess.close()
    
