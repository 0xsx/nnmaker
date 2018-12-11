#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Preprocesses and makes TFRecords files from the MNIST dataset.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function



import gzip
import json
import numpy as np

from os import listdir, makedirs, path
from struct import unpack

import tensorflow as tf


_EPSILON = float(1e-8)






def preprocess_images(img_arr):
  """Returns a new array with pixel values centered and scaled to a
  mean of 0 and variance of 1."""
  
  # flatten images and normalize to max value of 1
  flat_img_arr = img_arr.reshape((img_arr.shape[0], np.prod(img_arr.shape[1:])))
  flat_img_arr /= (np.max(flat_img_arr, axis=1, keepdims=True) + _EPSILON)


  # scale and center pixel values to mean of 0 and variance of 1
  flat_img_arr -= np.mean(flat_img_arr, axis=1, keepdims=True)
  flat_img_arr /= np.std(flat_img_arr, axis=1, keepdims=True) + _EPSILON

  out_img_arr = flat_img_arr.reshape((img_arr.shape[0], img_arr.shape[1],
                                      img_arr.shape[2], img_arr.shape[3]))

  return out_img_arr



def quantize_image(img, num_levels):
  """Returns a new numpy array with the image quantized to the specified
  number of levels."""

  bins = np.linspace(-1., 1., num=num_levels)
  return np.clip(np.digitize(img, bins), 0, num_levels-1).astype("int64")







def load_train_images(in_dir):
  """Loads all training images and returns a tuple of numpy arrays
  containing images and labels."""

  with gzip.open(path.realpath(path.join(in_dir, "train-images-idx3-ubyte.gz")), "rb") as f_in:
    magic_num = unpack(">I", f_in.read(4))[0]
    num_items = unpack(">I", f_in.read(4))[0]
    img_height = unpack(">I", f_in.read(4))[0]
    img_width = unpack(">I", f_in.read(4))[0]

    assert magic_num == 2051
    assert num_items == 60000
    assert img_height == 28
    assert img_width == 28

    x_images = np.zeros((num_items, img_height, img_width, 1), dtype="float32")

    for i in range(num_items):
      for j in range(img_height):
        for k in range(img_width):
          cur_pixel = 255 - unpack(">B", f_in.read(1))[0]
          x_images[i, j, k, 0] = cur_pixel


  x_images = preprocess_images(x_images)


  with gzip.open(path.realpath(path.join(in_dir, "train-labels-idx1-ubyte.gz")), "rb") as f_in:
    magic_num = unpack(">I", f_in.read(4))[0]
    num_items = unpack(">I", f_in.read(4))[0]

    assert magic_num == 2049
    assert num_items == 60000

    x_labels = np.zeros((num_items,), dtype="uint8")

    for i in range(num_items):
      x_labels[i] = unpack(">B", f_in.read(1))[0]


  return x_images, x_labels







def load_test_images(in_dir):
  """Loads all testing images and returns a tuple of numpy arrays
  containing images and labels."""

  with gzip.open(path.realpath(path.join(in_dir, "t10k-images-idx3-ubyte.gz")), "rb") as f_in:
    magic_num = unpack(">I", f_in.read(4))[0]
    num_items = unpack(">I", f_in.read(4))[0]
    img_height = unpack(">I", f_in.read(4))[0]
    img_width = unpack(">I", f_in.read(4))[0]

    assert magic_num == 2051
    assert num_items == 10000
    assert img_height == 28
    assert img_width == 28

    x_images = np.zeros((num_items, img_height, img_width, 1), dtype="float32")

    for i in range(num_items):
      for j in range(img_height):
        for k in range(img_width):
          cur_pixel = 255 - unpack(">B", f_in.read(1))[0]
          x_images[i, j, k, 0] = cur_pixel

    x_images = preprocess_images(x_images)



  with gzip.open(path.realpath(path.join(in_dir, "t10k-labels-idx1-ubyte.gz")), "rb") as f_in:
    magic_num = unpack(">I", f_in.read(4))[0]
    num_items = unpack(">I", f_in.read(4))[0]

    assert magic_num == 2049
    assert num_items == 10000

    x_labels = np.zeros((num_items,), dtype="uint8")

    for i in range(num_items):
      x_labels[i] = unpack(">B", f_in.read(1))[0]

  return x_images, x_labels











def write_set(images, labels, out_dir, max_num_per_file):
  """Writes the set of images and labels to tfrecords files in the specified
  output directory."""

  def __int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


  def __float32_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


  try:
    makedirs(out_dir)
  except OSError:
    pass

  writer_opts = tf.python_io.TFRecordOptions(tf.python_io.TFRecordCompressionType.ZLIB)
  
  cur_file = 0
  count = 0
  filename = path.join(out_dir, "0.tfrecords")
  writer = tf.python_io.TFRecordWriter(filename, options=writer_opts)

  for i in range(images.shape[0]):
    if count >= max_num_per_file:
      writer.close()
      count = 0
      cur_file += 1
      filename = path.join(out_dir, "%d.tfrecords" % cur_file)
      writer = tf.python_io.TFRecordWriter(filename, options=writer_opts)

    img = images[i]
    label = labels[i]
    count += 1

    example = tf.train.Example(
      features=tf.train.Features(
        feature={
        "example": __float32_feature(img.flatten()),
        "target": __int64_feature([label])
    }))
    writer.write(example.SerializeToString())

  writer.close()






def write_quantized_set(images, labels, out_dir, max_num_per_file, num_levels):
  """Writes the set of images and labels to tfrecords files in the specified
  output directory after quantizing the images."""

  def __int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


  try:
    makedirs(out_dir)
  except OSError:
    pass

  writer_opts = tf.python_io.TFRecordOptions(tf.python_io.TFRecordCompressionType.ZLIB)
  
  cur_file = 0
  count = 0
  filename = path.join(out_dir, "0.tfrecords")
  writer = tf.python_io.TFRecordWriter(filename, options=writer_opts)

  for i in range(images.shape[0]):
    if count >= max_num_per_file:
      writer.close()
      count = 0
      cur_file += 1
      filename = path.join(out_dir, "%d.tfrecords" % cur_file)
      writer = tf.python_io.TFRecordWriter(filename, options=writer_opts)

    img = quantize_image(images[i], num_levels)
    label = labels[i]
    count += 1

    example = tf.train.Example(
      features=tf.train.Features(
        feature={
        "example": __int64_feature(img.flatten()),
        "target": __int64_feature([label])
    }))
    writer.write(example.SerializeToString())

  writer.close()







def write_manifest(filename):
  """Writes the manifest file for the dataset as a JSON object."""

  manifest = {}
  manifest["compression"] = "zlib"
  manifest["allow_var_len"] = False
  manifest["features"] = [
    {"name": "example",
     "dtype": "float32",
     "shape": [28, 28],
     "var_len": False,
     "deserialize_type": "float"
    },

    {"name": "target",
     "dtype": "int64",
     "shape": [],
     "var_len": False,
     "deserialize_type": "int"
    }
  ]


  with open(filename, "w") as f_out:
    json.dump(manifest, f_out, indent=1)
    f_out.write("\n")




def write_quantized_manifest(filename):
  """Writes the manifest file for the dataset as a JSON object."""

  manifest = {}
  manifest["compression"] = "zlib"
  manifest["allow_var_len"] = False
  manifest["features"] = [
    {"name": "example",
     "dtype": "int64",
     "shape": [28, 28],
     "var_len": False,
     "deserialize_type": "int"
    },

    {"name": "target",
     "dtype": "int64",
     "shape": [],
     "var_len": False,
     "deserialize_type": "int"
    }
  ]


  with open(filename, "w") as f_out:
    json.dump(manifest, f_out, indent=1)
    f_out.write("\n")






def main(in_dir, out_train_dir, out_val_dir, out_test_dir, num_val,
         max_num_per_file, quant_levels):
  
  train_images, train_labels = load_train_images(in_dir)
  test_images, test_labels = load_test_images(in_dir)

  val_images = train_images[:num_val]
  train_images = train_images[num_val:]

  val_labels = train_labels[:num_val]
  train_labels = train_labels[num_val:]


  if args.quant_levels is None:
    write_set(train_images, train_labels, out_train_dir, max_num_per_file)
    write_set(val_images, val_labels, out_val_dir, max_num_per_file)
    write_set(test_images, test_labels, out_test_dir, max_num_per_file)

    write_manifest(path.join(out_train_dir, "__manifest__.json"))
    write_manifest(path.join(out_val_dir, "__manifest__.json"))
    write_manifest(path.join(out_test_dir, "__manifest__.json"))

  else:
    write_quantized_set(train_images, train_labels, out_train_dir,
                        max_num_per_file, quant_levels)
    write_quantized_set(val_images, val_labels, out_val_dir,
                        max_num_per_file, quant_levels)
    write_quantized_set(test_images, test_labels, out_test_dir,
                        max_num_per_file, quant_levels)

    write_quantized_manifest(path.join(out_train_dir, "__manifest__.json"))
    write_quantized_manifest(path.join(out_val_dir, "__manifest__.json"))
    write_quantized_manifest(path.join(out_test_dir, "__manifest__.json"))



if __name__ == "__main__":
  import argparse

  parser = argparse.ArgumentParser(description=__doc__)

  parser.add_argument("in_dir", help="Directory containing gzipped MNIST dataset files")
  
  parser.add_argument("out_train_dir", help="Directory to write training set")
  parser.add_argument("out_val_dir", help="Directory to write validation set")
  parser.add_argument("out_test_dir", help="Directory to write test set")

  parser.add_argument("--num_val", default=5000, type=int, metavar="n",
                      help="Number of training examples to use as validation (default: 5000)")

  parser.add_argument("--max_num_per_file", default=500, type=int, metavar="n",
                      help="Max number of examples per data file (default: 500)")

  parser.add_argument("--quant_levels", default=None, metavar="n",
                      help="Number of levels for quantizing pixels (default: None)")



  args = parser.parse_args()

  if args.quant_levels is None:
    quant_levels = None
  else:
    quant_levels = int(args.quant_levels)

  main(path.realpath(args.in_dir), path.realpath(args.out_train_dir),
       path.realpath(args.out_val_dir), path.realpath(args.out_test_dir),
       args.num_val, args.max_num_per_file, quant_levels)

