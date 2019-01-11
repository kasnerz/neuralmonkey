from typing import cast, Iterable, List

import numpy as np
import tensorflow as tf
import warpctc_tensorflow as warpctc
from typeguard import check_argument_types

from neuralmonkey.dataset import Dataset
from neuralmonkey.decorators import tensor
from neuralmonkey.model.feedable import FeedDict
from neuralmonkey.model.parameterized import InitializerSpecs
from neuralmonkey.model.model_part import ModelPart
from neuralmonkey.model.stateful import TemporalStateful
from neuralmonkey.model.sequence import EmbeddedSequence
from neuralmonkey.tf_utils import get_variable
from neuralmonkey.vocabulary import Vocabulary, END_TOKEN


class CTCDecoder(ModelPart):
    """Connectionist Temporal Classification.

    See `tf.nn.ctc_loss`, `tf.nn.ctc_greedy_decoder` etc.
    """

    # pylint: disable=too-many-arguments
    def __init__(self,
                 name: str,
                 encoder: TemporalStateful,
                 vocabulary: Vocabulary,
                 data_id: str,
                 decode_layer_index: int = 4,
                 input_sequence: EmbeddedSequence = None,
                 max_length: int = None,
                 merge_repeated_outputs: bool = True,
                 beam_width: int = 1,
                 reuse: ModelPart = None,
                 save_checkpoint: str = None,
                 load_checkpoint: str = None,
                 initializers: InitializerSpecs = None) -> None:
        check_argument_types()
        ModelPart.__init__(self, name, reuse, save_checkpoint, load_checkpoint,
                           initializers)

        self.encoder = encoder
        self.vocabulary = vocabulary
        self.data_id = data_id
        self.max_length = max_length
        self.decode_layer_index = decode_layer_index
        self.merge_repeated_outputs = merge_repeated_outputs
        self.beam_width = beam_width
        self.input_sequence = input_sequence

    # pylint: disable=no-self-use
    @tensor
    def flat_labels(self) -> tf.Tensor:
        return tf.placeholder(tf.int32, name="flat_labels")

    @tensor
    def label_lengths(self) -> tf.Tensor:
        return tf.placeholder(tf.int32, name="label_lengths")

    @tensor
    def decoded(self) -> tf.Tensor:
        if self.beam_width == 1:
            decoded, _ = tf.nn.ctc_greedy_decoder(
                inputs=self.logits, sequence_length=self.encoder.lengths,
                merge_repeated=self.merge_repeated_outputs)
        else:
            decoded, _ = tf.nn.ctc_beam_search_decoder(
                inputs=self.logits, sequence_length=self.encoder.lengths,
                beam_width=self.beam_width,
                merge_repeated=self.merge_repeated_outputs)

        return tf.sparse_tensor_to_dense(
            tf.sparse_transpose(decoded[0]),
            default_value=self.vocabulary.get_word_index(END_TOKEN))

    @property
    def train_loss(self) -> tf.Tensor:
        return self.cost

    @property
    def runtime_loss(self) -> tf.Tensor:
        return self.cost

    @tensor
    def cost(self) -> tf.Tensor:
        loss = warpctc.ctc(
            activations=self.logits,
            flat_labels=self.flat_labels,
            label_lengths=self.label_lengths,
            input_lengths=self.encoder.lengths,
            blank_label=len(self.vocabulary))

        return tf.reduce_sum(loss)
    
    @tensor
    def logits(self) -> tf.Tensor:
        if self.input_sequence is None:
            return self.logits_with_weight_matrix
        
        return self.logits_with_input_sequence

    @tensor
    def logits_with_weight_matrix(self) -> tf.Tensor:
        vocabulary_size = len(self.vocabulary)

        encoder_states = self.encoder.temporal_states

        weights = get_variable(
            name="state_to_word_W",
            shape=[encoder_states.shape[2], vocabulary_size + 1],
            initializer=tf.random_uniform_initializer(-0.5, 0.5))

        biases = get_variable(
            name="state_to_word_b",
            shape=[vocabulary_size + 1],
            initializer=tf.zeros_initializer())

        # To multiply 3-D matrix (encoder hidden states) by a 2-D matrix
        # (weights), we use 1-by-1 convolution (similar trick can be found in
        # attention computation)

        encoder_states = tf.expand_dims(encoder_states, 2)
        weights_4d = tf.expand_dims(tf.expand_dims(weights, 0), 0)

        multiplication = tf.nn.conv2d(
            encoder_states, weights_4d, [1, 1, 1, 1], "SAME")
        multiplication_3d = tf.squeeze(multiplication, axis=[2])

        biases_3d = tf.expand_dims(tf.expand_dims(biases, 0), 0)

        logits = multiplication_3d + biases_3d
        return tf.transpose(logits, perm=[1, 0, 2])  # time major

    @tensor
    def logits_with_input_sequence(self) -> tf.Tensor:
        vocabulary_size = len(self.vocabulary)

        encoder_states = self.encoder.temporal_states
        embedding_matrix = tf.transpose(self.input_sequence.embedding_matrix)

        # there is no row for the blank symbol in the embedding matrix
        embedding_blank = tf.get_variable(
            name="state_to_word_blank",
            shape=[encoder_states.shape[2], 1],
            dtype=tf.float32)

        weights = tf.concat([embedding_matrix, embedding_blank],
                             axis=1)

        # To multiply 3-D matrix (encoder hidden states) by a 2-D matrix
        # (weights), we use 1-by-1 convolution (similar trick can be found in
        # attention computation)

        encoder_states = tf.expand_dims(encoder_states, 2)
        weights_4d = tf.expand_dims(tf.expand_dims(weights, 0), 0)

        multiplication = tf.nn.conv2d(
            encoder_states, weights_4d, [1, 1, 1, 1], "SAME")
        multiplication_3d = tf.squeeze(multiplication, axis=[2])

        logits = tf.transpose(multiplication_3d, perm=[1, 0, 2]) # time major
        return logits

    def feed_dict(self, dataset: Dataset, train: bool = False) -> FeedDict:
        fd = ModelPart.feed_dict(self, dataset, train)

        sentences = cast(Iterable[List[str]],
                         dataset.maybe_get_series(self.data_id))

        if sentences is None and train:
            raise ValueError("When training, you must feed "
                             "reference sentences")

        if sentences is not None:
            vectors, paddings = self.vocabulary.sentences_to_tensor(
                list(sentences), train_mode=train, max_len=self.max_length)

            # sentences_to_tensor returns time-major tensors, targets need to
            # be batch-major
            vectors = vectors.T
            paddings = paddings.T

            bool_mask = (paddings > 0.5)
            flat_labels = vectors[bool_mask]
            label_lengths = bool_mask.sum(axis=1)

            fd[self.label_lengths] = label_lengths
            fd[self.flat_labels] = flat_labels

        return fd
