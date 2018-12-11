# -*- coding: utf-8 -*-
"""
Defines loader objects that parse config files and return functions that provide
inputs to TensorFlow Estimator objects.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


__all__ = ["ContinuousSequenceBatchLoader", "DiscreteSequenceBatchLoader",
           "IndependentBatchLoader"]

import json
import tensorflow as tf


from os import path, walk



from nnetmaker.loader.parsers import *
from nnetmaker.loader.processors import *
from nnetmaker.loader.secondaries import *
from nnetmaker.util import *




_PARSERS = {"raw": RawParser,
            "int": Int64Parser,
            "float": FloatParser,
            "string": StringParser}

_SECONDARIES = {"ones": OnesFeature}

_PROCESSORS = {"slice": SliceProcessor,
               "onehot": OneHotProcessor,
               "reshape": ReshapeProcessor}



_EPSILON = float(1e-8)



def _parse_and_validate_manifest(manifest_filename):
  """Reads parameters from the specified __manifest__.json file, validates the
  entries, and returns a dictionary of record parser objects for each feature."""

  # Strip comments while keeping line numbers.
  s = ""
  with open(manifest_filename, "r") as f_in:
    for line in f_in:
      comment_pos = line.find("//")
      s += line[:comment_pos] + "\n"


  manifest = json.loads(s)
  manifest_val = ArgumentsValidator(manifest, "Dataset manifest")
  with manifest_val:

    compression_type = manifest_val.get("compression", [ATYPE_NONE, ATYPE_STRING], True)
    if compression_type is not None:
      compression_type = compression_type.upper()
      if compression_type not in ["ZLIB", "GZIP"]:
        raise ValueError("Unsupported compression type: %s" % compression_type)

    allow_var_len = manifest_val.get("allow_var_len", ATYPE_BOOL, True)
    features_list = manifest_val.get("features", ATYPE_DICTS_LIST, True)



  # Validate each feature and create parser objects.
  feat_parsers = {}
  feat_shapes = {}
  feat_dtypes = {}

  for feat in features_list:

    feat_val = ArgumentsValidator(feat, "Dataset feature")
    with feat_val:
      name = feat_val.get("name", ATYPE_STRING, True)
      dtype = tf.as_dtype(feat_val.get("dtype", ATYPE_STRING, True))
      shape = feat_val.get("shape", ATYPE_INTS_LIST, True)
      deserialize_type = feat_val.get("deserialize_type", ATYPE_STRING, True)
      deserialize_args = feat_val.get("deserialize_args", ATYPE_DICT, False, default={})
      var_len = feat_val.get("var_len", ATYPE_BOOL, allow_var_len, default=False)

    if var_len and not allow_var_len:
      raise ValueError("Variable length features not allowed for this dataset.")

    try:
      shape = [int(x) for x in list(shape)]
    except:
      raise ValueError("Invalid shape for feature `%s`: %s" % (name, shape))

    
    try:
      feat_parsers[name] = _PARSERS[deserialize_type](shape, dtype, deserialize_args, var_len)
    except KeyError:
      raise ValueError("Unsupported deserialization type: %s" % deserialize_type)


    if var_len:
      feat_shapes[name] = [-1] + shape
    else:
      feat_shapes[name] = shape

    feat_dtypes[name] = dtype


  return compression_type, allow_var_len, feat_parsers, feat_shapes, feat_dtypes









def _parse_dataset_dict(dataset_dict):
  """Parses the dataset dictionary and loads the list of filenames, parsers, and
  manifest info."""

  dataset_val = ArgumentsValidator(dataset_dict, "Dataset loader")

  with dataset_val:
    dataset_type = dataset_val.get("type", ATYPE_STRING, True)
    dataset_args = dataset_val.get("args", ATYPE_DICT, True)


  if dataset_type == "dir":
    # Recursively get all TFRecords files from the specified data dir into a sorted list.

    dataset_val = ArgumentsValidator(dataset_args, "Dataset dir loader")
    with dataset_val:
      data_dir = path.realpath(dataset_val.get("data_dir", ATYPE_STRING, True))
      manifest_filename = path.join(data_dir, "__manifest__.json")

    all_filenames = []
    for root_dir, _, filenames in walk(data_dir):
      for f in filenames:
        if f.lower().endswith(".tfrecords"):
          all_filenames.append(path.join(root_dir, f))

    all_filenames = list(sorted(all_filenames))

    if len(all_filenames) == 0:
      raise ValueError("No .tfrecords files found in %s" % data_dir)


  elif dataset_type == "list":
    # Use the user-specified file that explicitly lists data source locations.

    dataset_val = ArgumentsValidator(dataset_args, "Dataset list loader")
    with dataset_val:
      list_filename = path.realpath(dataset_val.get("list_file", ATYPE_STRING, True))
      manifest_filename = path.realpath(dataset_val.get("manifest_file", ATYPE_STRING, True))

    with open(list_filename) as f_in:
      all_filenames = [line.strip() for line in f_in if len(line.strip()) > 0] 

    if len(all_filenames) == 0:
      raise ValueError("No filenames found in %s" % list_filename)


  else:
    raise ValueError("Unsupported datatype type: %s" % dataset_type)



  return tuple([all_filenames] + list(_parse_and_validate_manifest(manifest_filename)))







class BaseLoader(object):
  """Base class for data loader objects."""

  @property
  def target_batch_size(self):
    """The target batch size of the loader."""
    return self._target_batch_size




  def __init__(self, in_names, target_names, loader_args_dict, **kwargs):

    loader_val = ArgumentsValidator(loader_args_dict, "Batch loader")


    with loader_val:
      # Load parameters from the specified dataset.
      tup = _parse_dataset_dict(loader_val.get("dataset", ATYPE_DICT, True))
      filenames = tup[0]
      compression_type = tup[1]
      allow_var_len = tup[2]
      parsers = tup[3]
      feat_shapes = tup[4]
      feat_dtypes = tup[5]



      # Read and validate parameters from the rest of the loader config.
      multi_load = loader_val.get("multi_load", ATYPE_BOOL, False, default=False)
      if allow_var_len and multi_load:
        raise ValueError("Multi load not supported for datasets with variable length.")

      loader_name = loader_val.get("name", ATYPE_STRING, True)
      epochs = loader_val.get("epochs", [ATYPE_NONE, ATYPE_INT], True)
      target_batch_size = loader_val.get("target_batch_size", ATYPE_INT, True)
      drop_remainder = loader_val.get("drop_remainder", ATYPE_BOOL, True)
      num_parallel_reads = loader_val.get("num_parallel_reads", ATYPE_INT, False, default=1)
      num_parallel_parses = loader_val.get("num_parallel_parses", ATYPE_INT, False, default=1)
      num_read_buffer_bytes = loader_val.get("num_read_buffer_bytes", ATYPE_INT, True)
      num_prefetch = loader_val.get("num_prefetch", ATYPE_INT, True)
      sloppy_interleave = loader_val.get("sloppy_interleave", ATYPE_BOOL, False, default=False)
      num_interleave_out_buffer_elements = loader_val.get("num_interleave_out_buffer_elements", ATYPE_INT, False, default=1)
      num_interleave_in_buffer_elements = loader_val.get("num_interleave_in_buffer_elements", ATYPE_INT, False, default=1)
      shuffle = loader_val.get("shuffle", ATYPE_BOOL, False, default=False)
      num_shuffle_buffer_elements = loader_val.get("num_shuffle_buffer_elements", ATYPE_INT, shuffle, default=1)
      num_filenames_shuffle_buffer = loader_val.get("num_filenames_shuffle_buffer", ATYPE_INT, shuffle, default=1)
      num_mix_files = loader_val.get("num_mix_files", ATYPE_INT, shuffle, default=1)


      primary_features = loader_val.get("primary_features", ATYPE_DICTS_LIST, True)
      if not primary_features:
        raise ValueError("No primary features specified.")

      secondary_features = loader_val.get("secondary_features", ATYPE_DICTS_LIST, False, default=[])
      if secondary_features and multi_load:
        raise ValueError("Secondary features not supported for multi loaded data.")

      processing_steps = loader_val.get("processing_steps", ATYPE_DICTS_LIST, False, default=[])
      if processing_steps and multi_load:
        raise ValueError("Processing steps not supported for multi loaded data.")

      padding = loader_val.get("padding", [ATYPE_BOOL, ATYPE_DICTS_LIST], False, default=[])
      if isinstance(padding, list):
        padding_required = len(padding) > 0
      else:
        padding_required = padding
        padding = []
      if padding_required and multi_load:
        raise ValueError("Batch padding not supported for multi loaded data.")


      normalization = loader_val.get("normalization", ATYPE_DICTS_LIST, False, default=[])
      normalization_required = len(normalization) > 0
      normalization_dict = {}
      for d in normalization:
        norm_val = ArgumentsValidator(d, "Tensor normalization")
        with norm_val:
          norm_name = norm_val.get("tensor", ATYPE_STRING, True)
          norm_axis = norm_val.get("axis", ATYPE_INT, True)
          if norm_name not in in_names and norm_name not in target_names:
            raise ValueError("Tensor %s not defined." % norm_name)
          normalization_dict[norm_name] = norm_axis

      if normalization_required and multi_load:
        raise ValueError("Normalization not supported for multi loaded data.")


      self._read_extra_params(loader_val)



    # Create lambda to deserialize requested features and return a single
    # dict of tensors, whether the TFRecords specifies sequence data or not.
    fixed_feat_def = {}
    var_feat_def = {}
    feat_name_pairs = []

    for feat_dict in primary_features:

      feat_val = ArgumentsValidator(feat_dict, "Primary feature")
      with feat_val:
        from_name = feat_val.get("from_name", ATYPE_STRING, True)
        to_name = feat_val.get("to_name", ATYPE_STRING, True)

      if to_name not in in_names and to_name not in target_names:
        raise ValueError("Primary feature with to_name `%s` not required." % to_name)

      try:
        feat, var_len = parsers[from_name].get_feat()
      except KeyError:
        raise ValueError("Primary feature `%s` not in dataset." % from_name)

      feat_name_pairs.append((to_name, from_name))

      if var_len:
        var_feat_def[from_name] = feat
      else:
        fixed_feat_def[from_name] = feat


    if allow_var_len:
      read_dict_fn = lambda record: {k: v for d in
                       tf.parse_single_sequence_example(record,
                         context_features=fixed_feat_def,
                         sequence_features=var_feat_def) for k, v in d.items()}
    elif multi_load:
      read_dict_fn = lambda record: tf.parse_example(record, fixed_feat_def)
    else:
      read_dict_fn = lambda record: tf.parse_single_example(record, fixed_feat_def)




    # Create lambda for executing the parser object on each deserialized tensor and
    # renaming it to its name specified in the config.
    if multi_load:
      parse_dict_fn = lambda read_dict, batch_size: {to: parsers[fro].parse_multi(read_dict[fro],
                                                                                  batch_size)
                                                     for to, fro in feat_name_pairs}
    else:
      parse_dict_fn = lambda read_dict, batch_size: {to: parsers[fro].parse(read_dict[fro])
                                                     for to, fro in feat_name_pairs}




    # Create lambdas for separating a single tensor dictionary into input/output.
    in_dict_fn = lambda tensor_dict: {name: tensor_dict[name]
                                      for name in set(in_names) & set(tensor_dict.keys())}
    out_dict_fn = lambda tensor_dict: {name: tensor_dict[name]
                                      for name in set(target_names) & set(tensor_dict.keys())}





    # We need to keep track of static shapes and dtypes for feature tensors to
    # create the padding step near the end of the pipeline. These may change
    # at each step specified in the config file. Shapes here only need to
    # track the number of dimensions; the actual size of each dimension isn't
    # used.
    shapes_dict = {feat_dict["to_name"]: feat_shapes[feat_dict["from_name"]]
                   for feat_dict in primary_features}
    dtypes_dict = {feat_dict["to_name"]: feat_dtypes[feat_dict["from_name"]]
                   for feat_dict in primary_features}




    # Create the secondary feature objects that may depend on primary feature
    # tensors, and keep track of their shapes/dtypes as well.
    secondary_objs = {}

    for sec_dict in secondary_features:

      sec_val = ArgumentsValidator(sec_dict, "Secondary feature")
      with sec_val:
        sec_name = sec_val.get("to_name", ATYPE_STRING, True)
        sec_type = sec_val.get("type", ATYPE_STRING, True)
        sec_args = sec_val.get("args", ATYPE_DICT, False, default={})

      if sec_name not in in_names and sec_name not in target_names:
        raise ValueError("Secondary feature with to_name `%s` not required." % sec_name)

      try:
        secondary_objs[sec_name] = _SECONDARIES[sec_type](sec_args, shapes_dict, dtypes_dict)
      except KeyError:
        raise ValueError("Unsupported secondary feature type: %s" % sec_type)


    for name in secondary_objs:
      shapes_dict[name] = secondary_objs[name].get_static_shape()
      dtypes_dict[name] = secondary_objs[name].get_dtype()





    # Construct Processor objects that encapsulate processing steps to be
    # performed in the pipeline, and adjust the shapes/dtypes for each step.
    processor_tups = []

    for proc_dict in processing_steps:

      proc_val = ArgumentsValidator(proc_dict, "Processing step")
      with proc_val:
        name = proc_val.get("tensor", ATYPE_STRING, True)
        proc_type = proc_val.get("type", ATYPE_STRING, True)
        args_dict = proc_val.get("args", ATYPE_DICT, False, default={})

      try:
        in_shape = shapes_dict[name]
        in_dtype = dtypes_dict[name]
      except KeyError:
        raise ValueError("Unknown tensor: %s" % name)

      try:
        proc = _PROCESSORS[proc_type](args_dict, in_shape, in_dtype)
      except KeyError:
        raise ValueError("Unsupported processor type: %s" % proc_type)

      processor_tups.append((name, proc))
      shapes_dict[name] = proc.get_static_shape()
      dtypes_dict[name] = proc.get_dtype()





    # Prepare padded shapes for input/output tensors if any are specified in
    # config file. Pad all dimensions to the batch maximum unless otherwise
    # specified.
    if not padding_required:
      padded_shapes = None
      pad_values = None

    else:
      pad_shapes_dict = {}
      pad_values_dict = {}

      for pad_dict in padding:
        pad_val = ArgumentsValidator(pad_dict, "Padding")
        with pad_val:
          name = pad_val.get("tensor", True, str)
          shape = pad_val.get("shape", [ATYPE_NONE, ATYPE_INTS_LIST], False)
          value = pad_val.get("value", [ATYPE_NONE, ATYPE_INT,
                                        ATYPE_FLOAT, ATYPE_STRING], False)

        if not name in shapes_dict:
          raise ValueError("Tensor `%s` does not exist for padding." % name)

        pad_shapes_dict[name] = shape
        pad_values_dict[name] = value

      for name in shapes_dict:
        try:
          shape = pad_shapes_dict[name]
        except KeyError:
          shape = None
        try:
          value = pad_values_dict[name]
        except KeyError:
          value = None


        if shape is not None:
          try:
            pad_shapes_dict[name] = [int(x) for x in list(shape)]
          except:
            raise ValueError("Invalid pad shape for `%s`" % name)

          if len(pad_shapes_dict[name]) != len(shapes_dict[name]):
            raise ValueError("Expected %d pad dimensions for tensor %s" % (len(shapes_dict[name]),
                                                                           name))
        else:
          pad_shapes_dict[name] = [-1] * len(shapes_dict[name])


        dtype = dtypes_dict[name]
        if value is not None:
          try:
            pad_values_dict[name] = tf.constant(value, dtype=dtype)
          except TypeError:
            raise ValueError("Invalid pad value for `%s`: %s" % (name, value))
        else:
          if dtype == tf.string:
            pad_values_dict[name] = tf.constant("", dtype=tf.string)
          else:
            pad_values_dict[name] = tf.constant(0, dtype=dtype)



      # Split dictionaries into in/out sides to use for building pipeline later.
      in_pad_shapes_dict = in_dict_fn(pad_shapes_dict)
      out_pad_shapes_dict = out_dict_fn(pad_shapes_dict)

      in_pad_values_dict = in_dict_fn(pad_values_dict)
      out_pad_values_dict = out_dict_fn(pad_values_dict)

      padded_shapes = (in_pad_shapes_dict, out_pad_shapes_dict)
      pad_values = (in_pad_values_dict, out_pad_values_dict)




    # Define function for normalizing tensors at the end of the pipeline, after
    # batching and padding.
    def normalization_fn(in_dict, out_dict):
      
      in_dict_norm = {}
      for name in in_dict:
        tensor = in_dict[name]
        if name in normalization_dict:
          axis = normalization_dict[name] + 1  # +1 for batch dimension.
          mean, var = tf.nn.moments(tensor, [axis], keep_dims=True)
          std = tf.sqrt(var + _EPSILON)
          in_dict_norm[name] = (tensor - mean) / (std + _EPSILON)
        else:
          in_dict_norm[name] = tensor

      out_dict_norm = {}
      for name in out_dict:
        tensor = out_dict[name]
        if name in normalization_dict:
          axis = normalization_dict[name] + 1  # +1 for batch dimension.
          mean, var = tf.nn.moments(tensor, [axis], keep_dims=True)
          std = tf.sqrt(var + _EPSILON)
          out_dict_norm[name] = (tensor - mean) / (std + _EPSILON)
        else:
          out_dict_norm[name] = tensor

      return in_dict_norm, out_dict_norm




    # Store the member variables required for building the pipeline.
    self._read_dict_fn = read_dict_fn
    self._parse_dict_fn = parse_dict_fn
    self._in_dict_fn = in_dict_fn
    self._out_dict_fn = out_dict_fn

    self._secondary_objs = secondary_objs
    self._processor_tups = processor_tups

    self._loader_name = loader_name
    self._filenames = filenames
    self._compression_type = compression_type
    self._shuffle = shuffle
    self._epochs = epochs
    self._drop_remainder = drop_remainder
    self._target_batch_size = target_batch_size
    self._multi_load = multi_load
    self._num_filenames_shuffle_buffer = num_filenames_shuffle_buffer
    self._num_read_buffer_bytes = num_read_buffer_bytes
    self._num_mix_files = num_mix_files
    self._sloppy_interleave = sloppy_interleave
    self._num_parallel_reads = num_parallel_reads
    self._num_interleave_out_buffer_elements = num_interleave_out_buffer_elements
    self._num_interleave_in_buffer_elements = num_interleave_in_buffer_elements
    self._num_shuffle_buffer_elements = num_shuffle_buffer_elements
    self._num_parallel_parses = num_parallel_parses
    self._num_prefetch = num_prefetch
    self._padding_required = padding_required
    self._padded_shapes = padded_shapes
    self._pad_values = pad_values
    self._normalization_required = normalization_required
    self._normalization_fn = normalization_fn



  def _read_extra_params(self, validator):
    pass


  def build_input_fn(self):
    return lambda *args: self._get_next(*args)


  def _get_next(self):
    raise NotImplementedError













class IndependentBatchLoader(BaseLoader):
  """A loader for iterating over batches of independent examples stored as
  TFRecord files in the specified data directory."""


  def _get_next(self, *args):

    # Honor `batch_size` argument if given.
    batch_size = self._target_batch_size
    if len(args) > 0:
      params = args[0]
      try:
        batch_size = params["batch_size"]
      except KeyError: pass



    # Build input pipeline using configuration loaded in constructor.
    with tf.name_scope("Loader%s" % self._loader_name):

      def __parse(record):

        with tf.name_scope("FeatureDeserializer"):
          read_dict = self._read_dict_fn(record)
          tensors_dict = self._parse_dict_fn(read_dict, batch_size)

        with tf.name_scope("SecondaryFeatures"):
          for name in self._secondary_objs:
            tensors_dict[name] = self._secondary_objs[name].get_tensor(tensors_dict)


        with tf.name_scope("FeatureProcessing"):
          for name, proc in self._processor_tups:
            tensors_dict[name] = proc.process(tensors_dict[name])


        in_dict = self._in_dict_fn(tensors_dict)
        out_dict = self._out_dict_fn(tensors_dict)


        return in_dict, out_dict




      load_fn = lambda filenames: tf.data.TFRecordDataset(filenames,
                                    compression_type=self._compression_type,
                                    buffer_size=self._num_read_buffer_bytes,
                                    num_parallel_reads=self._num_parallel_reads)

      filenames_dataset = tf.data.Dataset.from_tensor_slices(self._filenames)

      if self._shuffle:
        filenames_dataset = filenames_dataset.shuffle(self._num_filenames_shuffle_buffer)
        cycle_length = self._num_mix_files
      else:
        cycle_length = 1  # No mixing files if not shuffling.


      interleave_trans = tf.contrib.data.parallel_interleave(load_fn,
                           cycle_length=cycle_length, sloppy=self._sloppy_interleave,
                           buffer_output_elements=self._num_interleave_out_buffer_elements,
                           prefetch_input_elements=self._num_interleave_in_buffer_elements)

      dataset = filenames_dataset.apply(interleave_trans)



      if self._shuffle:
        shuffle_trans = tf.contrib.data.shuffle_and_repeat(self._num_shuffle_buffer_elements,
                                                           self._epochs)
        dataset = dataset.apply(shuffle_trans)
      else:
        dataset = dataset.repeat(self._epochs)


      if self._multi_load:   # Multi-load: batch records before parsing.

        if self._drop_remainder:
          batch_trans = tf.contrib.data.batch_and_drop_remainder(batch_size)
          dataset = dataset.apply(batch_trans)
        else:
          dataset = dataset.batch(batch_size)

        dataset = dataset.map(__parse, num_parallel_calls=self._num_parallel_parses)


      else:   # No multi-load, parse records before batching.

        if not self._padding_required:   # No padding required, fuse map and batch.
          batch_trans = tf.contrib.data.map_and_batch(__parse, batch_size,
                          num_parallel_batches=self._num_parallel_parses,
                          drop_remainder=self._drop_remainder)
          dataset = dataset.apply(batch_trans)

        else:  # Padding required.
          dataset = dataset.map(__parse, num_parallel_calls=self._num_parallel_parses)

          if self._drop_remainder:
            batch_trans = tf.contrib.data.padded_batch_and_drop_remainder(batch_size,
                                                                          self._padded_shapes,
                                                                          padding_values=self._pad_values)
            dataset = dataset.apply(batch_trans)

          else:
            dataset = dataset.padded_batch(batch_size, self._padded_shapes,
                                           padding_values=self._pad_values)



      if self._normalization_required:
        dataset = dataset.map(self._normalization_fn,
                              num_parallel_calls=self._num_parallel_parses)


      dataset = dataset.prefetch(self._num_prefetch)


      return dataset.make_one_shot_iterator().get_next()










class ContinuousSequenceBatchLoader(BaseLoader):
  """A loader for iterating over batches of subsequences selected from large
  continuous sequences of data. Each ``TFRecord`` file stores a sequence of contiguous
  data chunks that together comprise a sequence containing no boundaries within.
  Each file is considered an independent source, and batch subsequences are selected by
  concatenating all chunks along the first dimension then selecting windows of the
  concatenated data. All features for each chunk must have the same length
  along the first dimension."""

  
  def __init__(self, in_names, target_names, loader_args_dict, **kwargs):
    BaseLoader.__init__(self, in_names, target_names, loader_args_dict, **kwargs)

    if self._multi_load:
      raise ValueError("Multi load not supported for continuous sequence loader.")




  def _read_extra_params(self, validator):
    self._min_window = validator.get("min_window", ATYPE_INT, True)
    self._max_window = validator.get("max_window", ATYPE_INT, True)
    self._min_stride = validator.get("min_stride", ATYPE_INT, True)
    self._max_stride = validator.get("max_stride", ATYPE_INT, True)



  def _get_next(self, *args):

    # Honor `batch_size` argument if given.
    batch_size = self._target_batch_size
    if len(args) > 0:
      params = args[0]
      try:
        batch_size = params["batch_size"]
      except KeyError: pass



    # Build input pipeline using configuration loaded in constructor.
    with tf.name_scope("BatchLoader_%s" % self._loader_name):

      def __parse(record):

        with tf.name_scope("FeatureDeserializer"):
          read_dict = self._read_dict_fn(record)
          tensors_dict = self._parse_dict_fn(read_dict, batch_size)
          

        with tf.name_scope("SecondaryFeatures"):
          for name in self._secondary_objs:
            tensors_dict[name] = self._secondary_objs[name].get_tensor(tensors_dict)


        with tf.name_scope("FeatureProcessing"):
          for name, proc in self._processor_tups:
            tensors_dict[name] = proc.process(tensors_dict[name])


        in_dict = self._in_dict_fn(tensors_dict)
        out_dict = self._out_dict_fn(tensors_dict)


        return in_dict, out_dict




      load_fn = lambda filenames: tf.data.TFRecordDataset(filenames,
                                    compression_type=self._compression_type,
                                    buffer_size=self._num_read_buffer_bytes,
                                    num_parallel_reads=self._num_parallel_reads
                                    ).map(__parse, num_parallel_calls=self._num_parallel_parses
                                    ).apply(tf.contrib.data.unbatch()
                                    ).apply(tf.contrib.data.sliding_window_batch(
                                      tf.random_uniform([], minval=self._min_window, maxval=self._max_window+1, dtype="int64"),
                                      stride=tf.random_uniform([], minval=self._min_stride, maxval=self._max_stride+1, dtype="int64")))

      filenames_dataset = tf.data.Dataset.from_tensor_slices(self._filenames)




      if self._shuffle:
        filenames_dataset = filenames_dataset.shuffle(self._num_filenames_shuffle_buffer)
        cycle_length = self._num_mix_files
      else:
        cycle_length = 1  # No mixing files if not shuffling.


      interleave_trans = tf.contrib.data.parallel_interleave(load_fn,
                           cycle_length=cycle_length, sloppy=self._sloppy_interleave,
                           buffer_output_elements=self._num_interleave_out_buffer_elements,
                           prefetch_input_elements=self._num_interleave_in_buffer_elements)

      dataset = filenames_dataset.apply(interleave_trans)



      if self._shuffle:
        shuffle_trans = tf.contrib.data.shuffle_and_repeat(self._num_shuffle_buffer_elements,
                                                           self._epochs)
        dataset = dataset.apply(shuffle_trans)
      else:
        dataset = dataset.repeat(self._epochs)


      if not self._padding_required:
        dataset = dataset.batch(batch_size, drop_remainder=self._drop_remainder)

      else:
        if self._drop_remainder:
          batch_trans = tf.contrib.data.padded_batch_and_drop_remainder(batch_size,
                                                                        self._padded_shapes,
                                                                        padding_values=self._pad_values)
          dataset = dataset.apply(batch_trans)

        else:
          dataset = dataset.padded_batch(batch_size, self._padded_shapes,
                                         padding_values=self._pad_values)



      if self._normalization_required:
        dataset = dataset.map(self._normalization_fn,
                              num_parallel_calls=self._num_parallel_parses)

      dataset = dataset.prefetch(self._num_prefetch)


      return dataset.make_one_shot_iterator().get_next()













class DiscreteSequenceBatchLoader(BaseLoader):
  """A loader for iterating over batches of subsequences selected from large
  discrete sequences of data. Each TFRecord file stores a sequence of contiguous
  subsequences that each represent a single unit and together comprise a longer
  sequence. Each file is considered an independent source, and batch
  subsequences are selected by concatenating all unit subsequences along the
  first dimension then selecting a number of contiguous units, aligned at unit
  subsequence boundaries. Unit subsequence features may have different lengths
  along the first dimension."""


  pass





