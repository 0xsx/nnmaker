Input Pipeline
**************

The input pipeline assumes datasets exist somewhere they can be accessed by TensorFlow
as one or more ``TFRecord`` data files, and that all data files in a dataset
have the same feature names and structure. A manifest file for each dataset describes
these features and structure and must be configured prior to loading the dataset.

Data files are read, parsed, preprocessed, and batched according to the
configuration specified by the user. Features are requested from the
dataset and mapped to model inputs or outputs for training and evaluation.

Batch loaders construct a TensorFlow ``Estimator`` model's ``input_fn``. They
load requested features from records in the dataset, preprocess them, and
construct minibatches to feed into the model.


Dataset Specification
=====================

Datasets are comprised of ``TFRecord`` data files together with a manifest file containing
global information about the contents of those data files. All data files in a dataset
must be consistent with what is described in the manifest.


The manifest file contains a single JSON object with the following keys:

+--------------------------+----------------+-------------+----------------------------------+
| Key                      | Value Type     | Required    | Description                      |
+==========================+================+=============+==================================+
| ``compression``          | string or      | yes         | The type of compression used     |
|                          | ``null``       |             | for the records. May be          |
|                          |                |             | ``"zlib"`` or ``"gzip"``, or     |
|                          |                |             | ``null`` for no compression.     |
+--------------------------+----------------+-------------+----------------------------------+
| ``allow_var_len``        | boolean        | yes         | If ``false``, all records are    |
|                          |                |             | serialized ``Example``           |
|                          |                |             | protobufs with fixed-length      |
|                          |                |             | ``Feature`` protos. If           |
|                          |                |             | ``true``, all records are        |
|                          |                |             | ``SequenceExample`` protobufs    |
|                          |                |             | and may contain variable-length  |
|                          |                |             | ``FeatureList`` protos.          |
+--------------------------+----------------+-------------+----------------------------------+
| ``features``             | list           | yes         | A JSON array of objects          |
|                          |                |             | specifying all features          |
|                          |                |             | available for loading from the   |
|                          |                |             | dataset. Only features requested |
|                          |                |             | by the running pipeline          |
|                          |                |             | configuration will be loaded     |
|                          |                |             | for a model.                     |
+--------------------------+----------------+-------------+----------------------------------+


Dataset Features
----------------

All feature specifiers are JSON objects with the following keys:

+----------------------+----------------+-------------------+-----------------------------------+
| Key                  | Value Type     | Required          | Description                       |
+======================+================+===================+===================================+
| ``name``             | string         | yes               | The name of the feature or        |
|                      |                |                   | feature list in the protobuf.     |
+----------------------+----------------+-------------------+-----------------------------------+
| ``dtype``            | string         | yes               | The datatype to give the tensor   |
|                      |                |                   | once deserialized, cast           |
|                      |                |                   | if different from what is         |
|                      |                |                   | read from the data file.          |
+----------------------+----------------+-------------------+-----------------------------------+
| ``shape``            | list           | yes               | The shape to give the tensor      |
|                      |                |                   | once deserialized. If it is a     |
|                      |                |                   | variable length sequence, this    |
|                      |                |                   | is the shape of each sequence     |
|                      |                |                   | element instead.                  |
+----------------------+----------------+-------------------+-----------------------------------+
| ``var_len``          | boolean        | if                | Must be ``false`` if              |
|                      |                | ``allow_var_len`` | ``allow_var_len`` is ``false``.   |
|                      |                | is ``true``       | Otherwise, specifies that the     |
|                      |                |                   | feature is a fixed-length         |
|                      |                |                   | context feature if ``false``, or  |
|                      |                |                   | a variable-length feature list    |
|                      |                |                   | if ``true``.                      |
+----------------------+----------------+-------------------+-----------------------------------+
| ``deserialize_type`` | string         | yes               | Type of deserialization used to   |
|                      |                |                   | parse the serialized protobuf     |
|                      |                |                   | into a tensor, as defined below.  |
+----------------------+----------------+-------------------+-----------------------------------+
| ``deserialize_args`` | dictionary     | no                | Arguments JSON object for the     |
|                      |                |                   | specific type of deserialization. |
|                      |                |                   | Defaults to empty object, but     |
|                      |                |                   | types may require certain         |
|                      |                |                   | arguments to be present.          |
+----------------------+----------------+-------------------+-----------------------------------+


The types of deserialization available are as follows:

:Type: ``int``
:Description: The feature is stored in the data files as a TensorFlow int64 list.
:Arguments: None

|

:Type: ``float``
:Description: The feature is stored in the data files as a TensorFlow float list.
:Arguments: None

|

:Type: ``string``
:Description: The feature is stored in the data files as a TensorFlow bytes list.
:Arguments: None

|

:Type: ``raw``
:Description: The feature is stored in the data files as one or more tensors encoded
              as a raw bytes string in a TensorFlow bytes list.
:Arguments:

+---------------+--------------+-------------+-------------------------------------------+
| Key           | Value Type   | Required    | Description                               |
+===============+==============+=============+===========================================+
| ``endian``    | string       | yes         | Specifies the endianness of the raw       |
|               |              |             | bytes. Must be ``"little"`` or ``"big"``. |
+---------------+--------------+-------------+-------------------------------------------+
| ``len``       | integer      | no          | The number of raw byte strings in each    |
|               |              |             | record. If greater than ``1``, tensors    |
|               |              |             | are concatenated along a new first axis   |
|               |              |             | (before batching). Ignored if ``var_len`` |
|               |              |             | is ``true``. Defaults to ``1``.           |
+---------------+--------------+-------------+-------------------------------------------+










Types of Batch Loaders
======================

Different batch loader types are defined for different use cases. Each type
reads files containing ``TFRecord`` protobufs from either a specified local
directory or from a predefined list of files, then optionally shuffles the
contents and outputs minibatches that the computation graph consumes. 

The types of batch loaders are:

:Type: ``independent`` (:ref:`args <independent-loader-ref>`)
:Description: Each data file represents a collection of examples that are independent from
              all other data files, and each ``TFRecord`` inside them represents a
              single example independent from all other examples.
:Use Case Example: Each data file is a set of random images, each
                   ``TFRecord`` is a single image, and the task is to label each image.

|

:Type: ``continuous_sequence`` (:ref:`args <continuous-loader-ref>`)
:Description: Each data file represents a contiguous sequence of data, but the
              data inside each file is independent from the data inside other files.
              Each ``TFRecord`` represents a non-overlapping chunk of the full
              sequence, such that the concatenation of all records produces a
              single long sequence with no logical subsequence boundaries.
              Batch examples are selected by choosing random-length sliding windows
              from the concatenated full sequence. All selected features must
              have the same length along the first dimension.
:Use Case Example: Each data file is a song, each ``TFRecord`` is a
                   10 second segment of audio samples, and the task is to predict
                   the next sample from the previous 3 seconds.

|

:Type: ``discrete_sequence`` (:ref:`args <discrete-loader-ref>`)
:Description: Each data file again represents a contiguous sequence of data and
              again the data inside each file is independent from the data inside
              other files, but here each full sequence is separable into logical
              subsequences. Each ``TFRecord`` represents a non-overlapping subsequence,
              such that the concatenation of all records produces a single long
              sequence with logical subsequence boundaries at the beginning and end
              of each record. Batch examples are selected by choosing windows
              containing a random number of subsequences from the concatenated full
              sequence, aligned at record (subsequence) boundaries.
:Use Case Example: Each data file is a text document, each ``TFRecord``
                   is a single sentence, and the task is to predict the next word using
                   an LSTM model that resets the hidden and cell states every 3-5
                   sentences.





Batch Loader Arguments
======================

All batch loaders use the following configuration arguments:


+----------------------------------------+----------------+-------------+------------------------------+
| Key                                    | Value Type     | Required    | Description                  |
+========================================+================+=============+==============================+
| ``dataset``                            | dictionary     | yes         | Specfies the dataset to      |
|                                        |                |             | be loaded. See               |
|                                        |                |             | :ref:`dataset-ref` below.    |
+----------------------------------------+----------------+-------------+------------------------------+
| ``target_batch_size``                  | integer        | yes         | Target size of each batch.   |
+----------------------------------------+----------------+-------------+------------------------------+
| ``drop_remainder``                     | boolean        | yes         | If ``true``, every batch     |
|                                        |                |             | is exactly the same size     |
|                                        |                |             | and unaligned remaining      |
|                                        |                |             | examples are dropped. If     |
|                                        |                |             | ``false``, the final         |
|                                        |                |             | batch may be smaller than    |
|                                        |                |             | others. Only considered      |
|                                        |                |             | if ``epochs`` is not         |
|                                        |                |             | ``null``.                    |
+----------------------------------------+----------------+-------------+------------------------------+
| ``epochs``                             | integer or     | yes         | Number of passes through     |
|                                        | ``null``       |             | the dataset. Setting to      |
|                                        |                |             | ``null`` means infinite      |
|                                        |                |             | repetitions.                 |
+----------------------------------------+----------------+-------------+------------------------------+
| ``num_parallel_reads``                 | integer        | no          | Number of parallel threads   |
|                                        |                |             | for reading data. Defaults   |
|                                        |                |             | to ``1``.                    |
+----------------------------------------+----------------+-------------+------------------------------+
| ``num_parallel_parses``                | integer        | no          | Number of parallel threads   |
|                                        |                |             | for parsing data after       |
|                                        |                |             | reading. Defaults to ``1``.  |
+----------------------------------------+----------------+-------------+------------------------------+
| ``num_read_buffer_bytes``              | integer        | yes         | Number of bytes in the read  |
|                                        |                |             | buffer. Setting to ``0``     |
|                                        |                |             | means no read buffering.     |
+----------------------------------------+----------------+-------------+------------------------------+
| ``num_prefetch``                       | integer        | yes         | Maximum number of batches to |
|                                        |                |             | prefetch at the end of the   |
|                                        |                |             | input pipeline.              |
+----------------------------------------+----------------+-------------+------------------------------+
| ``shuffle``                            | boolean        | no          | Whether to shuffle data      |
|                                        |                |             | records. Defaults to         |
|                                        |                |             | ``false``.                   |
+----------------------------------------+----------------+-------------+------------------------------+
| ``num_shuffle_buffer_elements``        | integer        | if          | Number of data records to    |
|                                        |                | ``shuffle`` | buffer for shuffling.        |
|                                        |                | is ``true`` | Ignored if shuffling         |
|                                        |                |             | disabled.                    |
+----------------------------------------+----------------+-------------+------------------------------+
| ``num_filenames_shuffle_buffer``       | integer        | if          | Number of filenames to       |
|                                        |                | ``shuffle`` | buffer for shuffling         |
|                                        |                | is ``true`` | before loading records.      |
|                                        |                |             | Ignored if shuffling         |
|                                        |                |             | disabled.                    |
+----------------------------------------+----------------+-------------+------------------------------+
| ``num_mix_files``                      | integer        | if          | Number of files loaded       |
|                                        |                | ``shuffle`` | together to shuffle          |
|                                        |                | is ``true`` | records among. Ignored if    |
|                                        |                |             | shuffling disabled.          |
+----------------------------------------+----------------+-------------+------------------------------+
| ``sloppy_interleave``                  | boolean        | no          | Whether to enable            |
|                                        |                |             | nondeterministic loading     |
|                                        |                |             | for potentially faster       |
|                                        |                |             | batches. Defaults to         |
|                                        |                |             | ``false``.                   |
+----------------------------------------+----------------+-------------+------------------------------+
| ``num_interleave_out_buffer_elements`` | boolean        | no          | Number of output elements    |
|                                        |                |             | to buffer when               |
|                                        |                |             | interleaving file loading.   |
|                                        |                |             | Defaults to ``1``.           |
+----------------------------------------+----------------+-------------+------------------------------+
| ``num_interleave_in_buffer_elements``  | boolean        | no          | Number of input elements to  |
|                                        |                |             | prefetch when interleaving   |
|                                        |                |             | file loading. Defaults to    |
|                                        |                |             | ``1``.                       |
+----------------------------------------+----------------+-------------+------------------------------+
| ``primary_features``                   | list           | yes         | List of named features       |
|                                        |                |             | read from dataset.           |
|                                        |                |             | See :ref:`primaries-ref`     |
|                                        |                |             | below.                       |
+----------------------------------------+----------------+-------------+------------------------------+
| ``secondary_features``                 | list           | no          | List of named features       |
|                                        |                |             | constructed after primaries. |
|                                        |                |             | See :ref:`secondaries-ref`.  |
|                                        |                |             | below. Defaults to empty     |
|                                        |                |             | list.                        |
+----------------------------------------+----------------+-------------+------------------------------+
| ``processing_steps``                   | list           | no          | List of processing steps to  |
|                                        |                |             | perform on primary and       |
|                                        |                |             | secondary features. See      |
|                                        |                |             | :ref:`processors-ref` below. |
|                                        |                |             | Defaults to empty list.      |
+----------------------------------------+----------------+-------------+------------------------------+
| ``padding``                            | list or        | no          | Tensor padding               |
|                                        | boolean        |             | specifications. See          |
|                                        |                |             | :ref:`paddings-ref` below.   |
|                                        |                |             | Defaults to ``false``.       |
+----------------------------------------+----------------+-------------+------------------------------+




Arguments Details
-----------------



.. _dataset-ref:

``dataset``
^^^^^^^^^^^

The value of the ``dataset`` argument must be a single JSON object with the following keys:

+-------------------+--------------+-------------+----------------------------------------+
| Key               | Value Type   | Required    | Description                            |
+===================+==============+=============+========================================+
| ``type``          | string       | yes         | The type of dataset specifier, as      |
|                   |              |             | defined below.                         |
+-------------------+--------------+-------------+----------------------------------------+
| ``args``          | dictionary   | yes         | Arguments object for the specified     |
|                   |              |             | ``type``.                              |
+-------------------+--------------+-------------+----------------------------------------+


The type may be either of the following:



:Type: ``dir``
:Description: The dataset is loaded recursively from a local directory.
:Arguments:

+---------------+--------------+-------------+---------------------------------------+
| Key           | Value Type   | Required    | Description                           |
+===============+==============+=============+=======================================+
| ``data_dir``  | string       | yes         | Path to a locally-available data      |
|               |              |             | directory. The directory must contain |
|               |              |             | a dataset manifest file named         |
|               |              |             | ``__manifest__.json`` in its top      |
|               |              |             | level, and at least one file with     |
|               |              |             | a ``.tfrecords`` extension storing    |
|               |              |             | the ``TFRecord`` protobufs. The       |
|               |              |             | directory is read recursively and     |
|               |              |             | all ``.tfrecords`` files are assumed  |
|               |              |             | to be data files.                     |
+---------------+--------------+-------------+---------------------------------------+

|

:Type: ``list``
:Description: The dataset is loaded from an explicit list of files specified by the user.
:Arguments:

+-------------------+--------------+-------------+----------------------------------------+
| Key               | Value Type   | Required    | Description                            |
+===================+==============+=============+========================================+
| ``manifest_file`` | string       | yes         | Path to a locally-available dataset    |
|                   |              |             | manifest used for all data files.      |
+-------------------+--------------+-------------+----------------------------------------+
| ``list_file``     | string       | yes         | Path to a locally-available text file  |
|                   |              |             | listing an absolute path to each data  |
|                   |              |             | file, one per line. Paths in the text  |
|                   |              |             | file may be local, cloud locations,    |
|                   |              |             | URLs, or any file path that can be     |
|                   |              |             | loaded by TensorFlow.                  |
+-------------------+--------------+-------------+----------------------------------------+





.. _primaries-ref:

``primary_features``
^^^^^^^^^^^^^^^^^^^^

The value of the ``primary_features`` argument must be a list of JSON objects
containing the following keys:

+-------------------+--------------+-------------+----------------------------------------+
| Key               | Value Type   | Required    | Description                            |
+===================+==============+=============+========================================+
| ``from_name``     | string       | yes         | Name of a feature tensor in the        |
|                   |              |             | dataset as specified in its manifest.  |
+-------------------+--------------+-------------+----------------------------------------+
| ``to_name``       | string       | yes         | Name of the feature tensor once loaded |
|                   |              |             | into the input pipeline. Must be one   |
|                   |              |             | of the model's input or output tensors |
|                   |              |             | defined in the model configuration.    |
+-------------------+--------------+-------------+----------------------------------------+

.. note:: Every model input must be satisfied by a primary or secondary feature, or an
          error will be raised.

.. note:: Every ``to_name`` must be unique or an error will be raised.

.. note:: Any ``to_name`` that is constructed and not used by the model will raise an error.


.. _secondaries-ref:

``secondary_features``
^^^^^^^^^^^^^^^^^^^^^^

Secondary features are constructed after all primary features are loaded, and may
or may not reference primary features by their ``to_name``. The value of the
``secondary_features`` argument must be a list of JSON objects containing the
following keys:

+-------------------+--------------+-------------+----------------------------------------+
| Key               | Value Type   | Required    | Description                            |
+===================+==============+=============+========================================+
| ``to_name``       | string       | yes         | Name of the feature tensor in the      |
|                   |              |             | input pipeline. Must be one of the     |
|                   |              |             | model's input or output tensors        |
|                   |              |             | defined in the model configuration.    |
+-------------------+--------------+-------------+----------------------------------------+
| ``type``          | string       | yes         | The type of secondary feature, as      |
|                   |              |             | defined below.                         |
+-------------------+--------------+-------------+----------------------------------------+
| ``args``          | dictionary   | no          | Arguments object for the specified     |
|                   |              |             | ``type``. Defaults to an empty object, |
|                   |              |             | but types may require certain          |
|                   |              |             | arguments to be present.               |
+-------------------+--------------+-------------+----------------------------------------+


Secondary features may be any of the following types:

:Type: ``const``
:Description: Constructs a tensor with all entries a constant value.
:Arguments:

+----------------+--------------+-------------+----------------------------------------+
| Key            | Value Type   | Required    | Description                            |
+================+==============+=============+========================================+
| ``shape``      | list or      | yes         | Either a shape specifier as a JSON     |
|                | string       |             | array of integers, or the name of a    |
|                |              |             | primary feature tensor whose shape to  |
|                |              |             | copy at the time of construction. Must |
|                |              |             | not include the batch dimension.       |
+----------------+--------------+-------------+----------------------------------------+
| ``dtype``      | string       | yes         | Either a TensorFlow datatype specifier |
|                |              |             | or the name of a primary feature       |
|                |              |             | tensor whose datatype to copy at the   |
|                |              |             | time of construction.                  |
+----------------+--------------+-------------+----------------------------------------+
| ``value``      | ``dtype``    | no          | The constant value to populate the     |
|                |              |             | tensor with. Defaults to ``0`` or      |
|                |              |             | empty string.                          |
+----------------+--------------+-------------+----------------------------------------+




.. note:: Every model input must be satisfied by a primary or secondary feature, or an
          error will be raised.

.. note:: Every ``to_name`` must be unique or an error will be raised.

.. note:: Any ``to_name`` that is constructed and not used by the model will raise an error.









.. _processors-ref:

``processing_steps``
^^^^^^^^^^^^^^^^^^^^

Processing steps are performed after constructing both primary and secondary
feature tensors. They are performed in exactly the order specified. The value of the
``processing_steps`` argument must be a list of JSON objects containing the
following keys:

+-------------------+--------------+-------------+----------------------------------------+
| Key               | Value Type   | Required    | Description                            |
+===================+==============+=============+========================================+
| ``tensor``        | string       | yes         | Name of the primary or secondary       |
|                   |              |             | feature tensor to operate on.          |
+-------------------+--------------+-------------+----------------------------------------+
| ``type``          | string       | yes         | The type of processesing step, as      |
|                   |              |             | defined below.                         |
+-------------------+--------------+-------------+----------------------------------------+
| ``args``          | dictionary   | no          | Arguments object for the specified     |
|                   |              |             | ``type``. Defaults to an empty object, |
|                   |              |             | but types may require certain          |
|                   |              |             | arguments to be present.               |
+-------------------+--------------+-------------+----------------------------------------+


Processing steps may be any of the following types:

:Type: ``slice``
:Description: Extract a multidimensional slice from the tensor.
:Arguments:

+----------------+--------------+-------------+-----------------------------------------+
| Key            | Value Type   | Required    | Description                             |
+================+==============+=============+=========================================+
| ``slice``      | string       | yes         | The slice specifier as a Pythonic       |
|                |              |             | multidimensional array slice, with      |
|                |              |             | no batch dimension. Only integer        |
|                |              |             | indices, negative indices, and colons   |
|                |              |             | are allowed, e.g. ``[3,:,4:,:-2,1:4]``. |
+----------------+--------------+-------------+-----------------------------------------+




.. note:: All tensors are input and output from each step using the names given at
          construction. Each processing step overwrites the tensor it operates on.




.. _paddings-ref:

``padding``
^^^^^^^^^^^

Either all tensors in the pipeline are padded, or none are. For this reason, the value
of the ``padding`` argument may be ``true`` to zero pad all tensors to the batch maximum
along each dimension, or ``false`` to specify no padding. If more specific padding
requirements are needed, the ``padding`` argument may be a list of JSON objects
with the following keys:


+----------------+--------------+-------------+----------------------------------------+
| Key            | Value Type   | Required    | Description                            |
+================+==============+=============+========================================+
| ``tensor``     | string       | yes         | Name of the feature tensor to pad.     |
+----------------+--------------+-------------+----------------------------------------+
| ``shape``      | list         | no          | Shape specifier as a JSON array of     |
|                |              |             | integers. Set any dimension to ``-1``  |
|                |              |             | to pad to the maximum batch size. Must |
|                |              |             | not include the batch dimension.       |
|                |              |             | Defaults to maximum batch size for     |
|                |              |             | all dimensions.                        |
+----------------+--------------+-------------+----------------------------------------+
| ``value``      | ``dtype`` of | no          | The constant value to pad the          |
|                | padded       |             | tensor with. Defaults to ``0`` or      |
|                | tensor       |             | empty string.                          |
+----------------+--------------+-------------+----------------------------------------+


.. note:: By either setting ``padding`` to ``true`` or by specifying at least
          one padding object, **all** tensors are automatically zero padded to the
          batch maximum along all dimensions unless a different desired padding
          specification is explicitly given. Either all tensors are padded, or none are.








Type-Specific Arguments
-----------------------

Each loader type has special arguments that define the behavior.


.. _independent-loader-ref:

:Type: ``independent``
:Additional Arguments:

+------------------+----------------+-------------+----------------------------------+
| Key              | Value Type     | Required    | Description                      |
+==================+================+=============+==================================+
| ``multi_load``   | boolean        | no          | If ``true``, records are         |
|                  |                |             | loaded and parsed in batches.    |
|                  |                |             | If ``false``, each record is     |
|                  |                |             | loaded and parsed individually   |
|                  |                |             | and then batched later. Enabling |
|                  |                |             | may boost performance for some   |
|                  |                |             | input pipelines, but is only     |
|                  |                |             | available in very restricted     |
|                  |                |             | cases. Must be ``false`` if the  |
|                  |                |             | records are variable length or   |
|                  |                |             | require any secondary features,  |
|                  |                |             | processing steps, or padding.    |
|                  |                |             | May not be available for all     |
|                  |                |             | dataset types. Defaults to       |
|                  |                |             | ``false``.                       |
+------------------+----------------+-------------+----------------------------------+


|


.. _continuous-loader-ref:

:Type: ``continuous_sequence``
:Additional Arguments:

+------------------+----------------+-------------+-----------------------------------------+
| Key              | Value Type     | Required    | Description                             |
+==================+================+=============+=========================================+
| ``min_window``   | integer        | yes         | Minimum random size of the sliding      |
|                  |                |             | window to select.                       |
+------------------+----------------+-------------+-----------------------------------------+
| ``max_window``   | integer        | yes         | Maximum random size of the sliding      |
|                  |                |             | window to select.                       |
+------------------+----------------+-------------+-----------------------------------------+
| ``stride``       | integer or     | no          | Sliding window stride, or ``null`` for  |
|                  |                |             | striding with the exact window length   |
|                  |                |             | and no overlap. Defaults to ``null``.   |
+------------------+----------------+-------------+-----------------------------------------+


.. note:: All selected features must have the same length along the first dimension.


|


.. _discrete-loader-ref:

:Type: ``discrete_sequence``
:Additional Arguments:

+------------------+----------------+-------------+-----------------------------------------+
| Key              | Value Type     | Required    | Description                             |
+==================+================+=============+=========================================+
| ``min_window``   | integer        | yes         | Minimum random number of subsequences   |
|                  |                |             | to include in a selected window.        |
+------------------+----------------+-------------+-----------------------------------------+
| ``max_window``   | integer        | yes         | Maximum random number of subsequences   |
|                  |                |             | to include in a selected window.        |
+------------------+----------------+-------------+-----------------------------------------+


.. note:: Unlike with the ``continuous_sequence`` loader, selected features may
          have different lengths along their first dimension. They will still
          be aligned at subsequence boundaries.


