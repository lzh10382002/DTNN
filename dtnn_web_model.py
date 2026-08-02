"""TensorFlow checkpoint loader used by the Streamlit deployment.

This module intentionally contains inference-only TensorFlow code.  It rebuilds
the original DTNN dense network and restores the original TensorFlow 1.x
checkpoints through TensorFlow's supported ``compat.v1`` API.
"""

from __future__ import annotations

import os
from pathlib import Path
from threading import RLock

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import tensorflow.compat.v1 as tf


tf.disable_v2_behavior()


class ModelLoadError(RuntimeError):
    """Raised when a DTNN checkpoint or its metadata cannot be loaded."""


class DTNNModel:
    """Load and run one of the original 1%/2% DTNN checkpoints."""

    def __init__(self, model_dir: str | Path):
        self.model_dir = Path(model_dir).resolve()
        self._lock = RLock()

        metadata = self._load_metadata()
        self.layers = [int(value) for value in metadata["layers"]]
        self.lb_input = np.asarray(metadata["lb_input"], dtype=np.float64)
        self.ub_input = np.asarray(metadata["ub_input"], dtype=np.float64)
        self.lb_output = np.asarray(metadata["lb_output"], dtype=np.float64)
        self.ub_output = np.asarray(metadata["ub_output"], dtype=np.float64)
        self._validate_metadata()

        self.checkpoint_prefix = self.model_dir / "model.ckpt"
        self._validate_checkpoint_files()

        self.graph = tf.Graph()
        with self.graph.as_default():
            self.input_tf = tf.placeholder(
                tf.float64, shape=[None, 4], name="dtnn_input"
            )
            self.weights, self.biases = self._create_checkpoint_variables()
            self.rf_pred = self._build_forward_pass(self.input_tf)

            checkpoint_variables = {
                variable.op.name: variable
                for variable in [*self.weights, *self.biases]
            }
            self.saver = tf.train.Saver(var_list=checkpoint_variables)
            self.init_op = tf.global_variables_initializer()

        config = tf.ConfigProto(
            allow_soft_placement=True,
            intra_op_parallelism_threads=1,
            inter_op_parallelism_threads=1,
        )
        config.gpu_options.allow_growth = True
        self.session = tf.Session(graph=self.graph, config=config)

        try:
            with self.graph.as_default():
                self.session.run(self.init_op)
                self._validate_checkpoint_variable_names()
                self.saver.restore(self.session, str(self.checkpoint_prefix))
        except Exception as exc:
            self.session.close()
            raise ModelLoadError(
                f"Unable to restore TensorFlow checkpoint in {self.model_dir}: {exc}"
            ) from exc

    def _load_metadata(self) -> dict:
        metadata_path = self.model_dir / "metadata.npy"
        if not metadata_path.is_file():
            raise ModelLoadError(f"Missing model metadata: {metadata_path}")

        try:
            metadata = np.load(metadata_path, allow_pickle=True).item()
        except Exception as exc:
            raise ModelLoadError(f"Unable to read {metadata_path}: {exc}") from exc

        required_keys = {
            "layers",
            "lb_input",
            "ub_input",
            "lb_output",
            "ub_output",
        }
        missing = required_keys.difference(metadata)
        if missing:
            raise ModelLoadError(
                f"Metadata in {metadata_path} is missing: {sorted(missing)}"
            )
        return metadata

    def _validate_metadata(self) -> None:
        if len(self.layers) < 2 or self.layers[0] != 4 or self.layers[-1] != 1:
            raise ModelLoadError(
                f"Unexpected DTNN architecture {self.layers}; expected 4 inputs and 1 output"
            )
        if self.lb_input.shape != (4,) or self.ub_input.shape != (4,):
            raise ModelLoadError("Input bounds in metadata must each contain four values")
        if self.lb_output.shape != (1,) or self.ub_output.shape != (1,):
            raise ModelLoadError("Output bounds in metadata must each contain one value")
        if np.any(self.ub_input <= self.lb_input):
            raise ModelLoadError("Invalid input bounds in model metadata")

    def _validate_checkpoint_files(self) -> None:
        required = [
            self.model_dir / "model.ckpt.index",
            self.model_dir / "model.ckpt.data-00000-of-00001",
        ]
        missing = [str(path) for path in required if not path.is_file()]
        if missing:
            raise ModelLoadError(f"Missing checkpoint files: {missing}")

    @staticmethod
    def _checkpoint_variable_name(index: int) -> str:
        return "Variable" if index == 0 else f"Variable_{index}"

    def _create_checkpoint_variables(self):
        """Recreate variables in the exact order used by the original DTNN."""

        weights = []
        biases = []
        variable_index = 0

        for input_width, output_width in zip(self.layers[:-1], self.layers[1:]):
            weight = tf.Variable(
                tf.zeros([input_width, output_width], dtype=tf.float64),
                dtype=tf.float64,
                name=self._checkpoint_variable_name(variable_index),
            )
            variable_index += 1
            bias = tf.Variable(
                tf.zeros([1, output_width], dtype=tf.float64),
                dtype=tf.float64,
                name=self._checkpoint_variable_name(variable_index),
            )
            variable_index += 1
            weights.append(weight)
            biases.append(bias)

        return weights, biases

    def _build_forward_pass(self, inputs):
        lower = tf.constant(self.lb_input.reshape(1, 4), dtype=tf.float64)
        upper = tf.constant(self.ub_input.reshape(1, 4), dtype=tf.float64)
        normalized = 2.0 * (inputs - lower) / (upper - lower) - 1.0

        hidden = normalized
        for weight, bias in zip(self.weights[:-1], self.biases[:-1]):
            hidden = tf.tanh(tf.matmul(hidden, weight) + bias)

        raw_rf = tf.matmul(hidden, self.weights[-1]) + self.biases[-1]
        capped_rf = tf.clip_by_value(raw_rf, 0.0, 1.0)

        depth = inputs[:, 0:1]
        deep_rf = tf.where(
            tf.greater(depth, 300.0),
            tf.ones_like(capped_rf, dtype=tf.float64),
            capped_rf,
        )
        return tf.where(
            tf.equal(depth, 0.0),
            tf.zeros_like(capped_rf, dtype=tf.float64),
            deep_rf,
            name="rf_prediction",
        )

    def _validate_checkpoint_variable_names(self) -> None:
        checkpoint_names = {
            name for name, _shape in tf.train.list_variables(str(self.checkpoint_prefix))
        }
        expected_names = {
            variable.op.name for variable in [*self.weights, *self.biases]
        }
        missing = expected_names.difference(checkpoint_names)
        if missing:
            raise ModelLoadError(
                f"Checkpoint is missing expected variables: {sorted(missing)}"
            )

    def predict(self, inputs: np.ndarray) -> np.ndarray:
        """Predict RF for an ``N x 4`` or ``N x 5`` input array."""

        values = np.asarray(inputs, dtype=np.float64)
        if values.ndim == 1:
            values = values.reshape(1, -1)
        if values.ndim != 2 or values.shape[1] not in (4, 5):
            raise ValueError("DTNN input must have shape (N, 4) or (N, 5)")
        if not np.all(np.isfinite(values[:, :4])):
            raise ValueError("DTNN input contains a non-finite value")

        with self._lock, self.graph.as_default():
            return self.session.run(
                self.rf_pred, feed_dict={self.input_tf: values[:, :4]}
            )

    def predict_one(self, depth: float, magnitude: float, span: float, log_q: float) -> float:
        result = self.predict([[depth, magnitude, span, log_q]])
        return float(result[0, 0])

    def close(self) -> None:
        with self._lock:
            self.session.close()

