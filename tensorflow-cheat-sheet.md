# TensorFlow Cheat Sheet

## Table of Contents

1. [Installation](#installation)
2. [Basic Setup](#basic-setup)
3. [Tensors](#tensors)
4. [Operations](#operations)
5. [Variables](#variables)
6. [Neural Networks](#neural-networks)
7. [Layers](#layers)
8. [Models](#models)
9. [Training](#training)
10. [Data Processing](#data-processing)
11. [Saving & Loading](#saving--loading)
12. [Debugging](#debugging)
13. [Common Patterns](#common-patterns)

## Installation

```bash
# Install TensorFlow
pip install tensorflow

# Install with GPU support (if CUDA is available)
pip install tensorflow-gpu

# Install specific version
pip install tensorflow==2.15.0

# Check installation
python -c "import tensorflow as tf; print(tf.__version__)"
```

## Basic Setup

```python
import tensorflow as tf
import numpy as np

# Check TensorFlow version
print(tf.__version__)

# Check if GPU is available
print("GPU Available:", tf.config.list_physical_devices('GPU'))

# Set memory growth for GPU (prevents memory allocation issues)
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)
```

## Tensors

### Creating Tensors

Multi-dimensional arrays that hold numbers, strings, or booleans. The fundamental data structure for all TensorFlow operations.

```python
# From Python lists
tensor = tf.constant([1, 2, 3, 4])
matrix = tf.constant([[1, 2], [3, 4]])

# From NumPy arrays
np_array = np.array([1, 2, 3, 4])
tensor = tf.constant(np_array)

# Special tensors
zeros = tf.zeros([2, 3])             # 2x3 matrix of zeros
ones = tf.ones([2, 3])               # 2x3 matrix of ones
identity = tf.eye(3)                 # 3x3 identity matrix
random = tf.random.normal([2, 3])    # Random normal distribution

# Range tensors
range_tensor = tf.range(10)          # [0, 1, 2, ..., 9]
linspace = tf.linspace(0.0, 1.0, 5)  # [0.0, 0.25, 0.5, 0.75, 1.0]
```

### Constant vs Variable

Constants are immutable, variables are mutable. Constants store fixed values, variables store learnable parameters.
1. Mutability: Constants cannot be changed, variables can
2. Training: Variables are trainable by default, constants are not
3. Memory: Constants are stored in the graph, variables in memory
4. Use cases: Constants for fixed values, variables for learnable parameters

```python
# tf.constant - Immutable values
constant = tf.constant([1, 2, 3])
constant.assign([4, 5, 6])  # This would cause an error!

# tf.Variable - Mutable values
variable = tf.Variable([1, 2, 3])
variable.assign([4, 5, 6])  # This works!
```

### Variables

```python
# Create variables
var = tf.Variable([1, 2, 3], name="my_variable")
var_zeros = tf.Variable(tf.zeros([2, 3]), name="weights")

# Assign new values
var.assign([4, 5, 6])
var.assign_add([1, 1, 1])  # Add to existing value
var.assign_sub([1, 1, 1])  # Subtract from existing value

# Properties
print(var.name)        # Variable name
print(var.shape)       # Variable shape
print(var.dtype)       # Variable data type
print(var.trainable)   # Whether variable is trainable

# Convert to tensor
tensor_value = var.value()
```

### Tensor Properties

Helps you debug shape mismatches and data type errors. Shows dimensions, size, and data type of tensors.

```python
tensor = tf.constant([[1, 2, 3], [4, 5, 6]])

# Shape and dimensions
print(tensor.shape)      # (2, 3)
print(tensor.ndim)       # 2
print(tf.rank(tensor))   # 2
print(tf.size(tensor))   # 6

# Data type
print(tensor.dtype)      # <dtype: 'int32'>

# Convert to NumPy
numpy_array = tensor.numpy()
```

### Reshaping Tensors

Changes tensor dimensions to match layer input requirements. Essential for preparing data for different operations.

```python
tensor = tf.constant([1, 2, 3, 4, 5, 6])

# Reshape
reshaped = tf.reshape(tensor, [2, 3])      # [[1, 2, 3], [4, 5, 6]]
reshaped = tf.reshape(tensor, [3, -1])     # Auto-calculate dimension

# Transpose
matrix = tf.constant([[1, 2, 3], [4, 5, 6]])
transposed = tf.transpose(matrix)          # [[1, 4], [2, 5], [3, 6]]

# Expand/squeeze dimensions
expanded = tf.expand_dims(tensor, axis=0)  # Add dimension at axis 0
squeezed = tf.squeeze(expanded)            # Remove dimensions of size 1
```

## Operations

### Mathematical Operations

Building blocks of neural networks. Performs computations on tensors from basic arithmetic to matrix multiplication.

```python
a = tf.constant([1, 2, 3])
b = tf.constant([4, 5, 6])

# Element-wise operations
addition = tf.add(a, b)      # or a + b
subtraction = tf.subtract(a, b)  # or a - b
multiplication = tf.multiply(a, b)  # or a * b
division = tf.divide(a, b)   # or a / b

# Matrix operations
matrix_a = tf.constant([[1, 2], [3, 4]])
matrix_b = tf.constant([[5, 6], [7, 8]])
matmul = tf.matmul(matrix_a, matrix_b)  # Matrix multiplication

# Reduction operations
sum_all = tf.reduce_sum(a)           # Sum all elements
sum_axis = tf.reduce_sum(matrix_a, axis=0)  # Sum along axis 0
mean = tf.reduce_mean(a)             # Mean of all elements
max_val = tf.reduce_max(a)           # Maximum value
min_val = tf.reduce_min(a)           # Minimum value

# Other useful operations
absolute = tf.abs(a)                 # Absolute value
square = tf.square(a)                # Element-wise square
sqrt = tf.sqrt(tf.cast(a, tf.float32))  # Square root
exp = tf.exp(tf.cast(a, tf.float32))    # Exponential
log = tf.math.log(tf.cast(a, tf.float32))  # Natural logarithm
```

### Logical Operations

Filters data, creates masks, and performs conditional computations. Essential for data preprocessing and model logic.

```python
a = tf.constant([True, False, True])
b = tf.constant([False, False, True])

# Logical operations
logical_and = tf.logical_and(a, b)
logical_or = tf.logical_or(a, b)
logical_not = tf.logical_not(a)

# Comparison operations
x = tf.constant([1, 2, 3])
y = tf.constant([1, 4, 2])

equal = tf.equal(x, y)           # Element-wise equality
greater = tf.greater(x, y)       # Element-wise greater than
less = tf.less(x, y)            # Element-wise less than
```

## Neural Networks

### Activation Functions

Introduces non-linearity into neural networks. Allows models to learn complex patterns beyond linear relationships.

```python
x = tf.constant([-2.0, -1.0, 0.0, 1.0, 2.0])

# Common activation functions
relu = tf.nn.relu(x)                    # ReLU
leaky_relu = tf.nn.leaky_relu(x)        # Leaky ReLU
sigmoid = tf.nn.sigmoid(x)              # Sigmoid
tanh = tf.nn.tanh(x)                    # Hyperbolic tangent
softmax = tf.nn.softmax(x)              # Softmax
gelu = tf.nn.gelu(x)                    # GELU
swish = tf.nn.swish(x)                  # Swish
```

### Loss Functions

Measures how far predictions are from actual values. Guides the training process by quantifying model errors.

```python
# Classification losses
y_true = tf.constant([0, 1, 1, 0])
y_pred = tf.constant([0.1, 0.9, 0.8, 0.2])

# Binary crossentropy
bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)

# Categorical crossentropy
y_true_cat = tf.constant([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
y_pred_cat = tf.constant([[0.9, 0.05, 0.05], [0.1, 0.8, 0.1], [0.2, 0.2, 0.6]])
cce = tf.keras.losses.categorical_crossentropy(y_true_cat, y_pred_cat)

# Sparse categorical crossentropy
y_true_sparse = tf.constant([0, 1, 2])
scce = tf.keras.losses.sparse_categorical_crossentropy(y_true_sparse, y_pred_cat)

# Regression losses
y_true_reg = tf.constant([1.0, 2.0, 3.0])
y_pred_reg = tf.constant([1.1, 1.9, 3.2])

mse = tf.keras.losses.mean_squared_error(y_true_reg, y_pred_reg)
mae = tf.keras.losses.mean_absolute_error(y_true_reg, y_pred_reg)
huber = tf.keras.losses.huber(y_true_reg, y_pred_reg)
```

## Layers

### Dense Layers

Fully connected layers where each neuron connects to every neuron in the previous layer. Foundation of most neural networks.

```python
# Dense (fully connected) layer
dense = tf.keras.layers.Dense(units=64, activation='relu', name='dense_1')

# With specific initialization
dense = tf.keras.layers.Dense(
    units=64,
    activation='relu',
    kernel_initializer='glorot_uniform',
    bias_initializer='zeros',
    kernel_regularizer=tf.keras.regularizers.l2(0.01)
)
```

### Convolutional Layers

Detects features like edges and patterns by sliding filters across input data. Essential for image processing tasks.

```python
# 2D Convolution
conv2d = tf.keras.layers.Conv2D(
    filters=32,
    kernel_size=(3, 3),
    strides=(1, 1),
    padding='same',
    activation='relu'
)

# 1D Convolution
conv1d = tf.keras.layers.Conv1D(
    filters=64,
    kernel_size=3,
    strides=1,
    padding='same',
    activation='relu'
)

# Transposed convolution (deconvolution)
conv2d_transpose = tf.keras.layers.Conv2DTranspose(
    filters=32,
    kernel_size=(3, 3),
    strides=(2, 2),
    padding='same'
)
```

### Pooling Layers

Reduces spatial dimensions while keeping important information. Prevents overfitting and reduces computational load.

```python
# Max pooling
max_pool = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))

# Average pooling
avg_pool = tf.keras.layers.AveragePooling2D(pool_size=(2, 2))

# Global pooling
global_max_pool = tf.keras.layers.GlobalMaxPooling2D()
global_avg_pool = tf.keras.layers.GlobalAveragePooling2D()
```

### Recurrent Layers

Processes sequential data by maintaining memory of previous inputs. Perfect for text, time series, and other ordered data.

```python
# LSTM
lstm = tf.keras.layers.LSTM(units=128, return_sequences=True, return_state=True)

# GRU
gru = tf.keras.layers.GRU(units=64, return_sequences=False)

# Simple RNN
simple_rnn = tf.keras.layers.SimpleRNN(units=32)

# Bidirectional wrapper
bidirectional_lstm = tf.keras.layers.Bidirectional(
    tf.keras.layers.LSTM(64, return_sequences=True)
)
```

### Regularization Layers

Prevents overfitting by adding constraints or noise during training. Helps models generalize to new data.

```python
# Dropout
dropout = tf.keras.layers.Dropout(rate=0.5)

# Batch normalization
batch_norm = tf.keras.layers.BatchNormalization()

# Layer normalization
layer_norm = tf.keras.layers.LayerNormalization()
```

## Models

### Sequential Model

Simplest way to build neural networks with layers stacked linearly. Perfect for straightforward architectures.

```python
# Sequential model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation='relu', input_shape=(784,)),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

# Alternative way to build sequential model
model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(128, activation='relu', input_shape=(784,)))
model.add(tf.keras.layers.Dropout(0.2))
model.add(tf.keras.layers.Dense(64, activation='relu'))
model.add(tf.keras.layers.Dense(10, activation='softmax'))
```

### Functional API

Builds complex models with multiple inputs, outputs, or non-linear connections. More flexible than Sequential models.

```python
# Functional API for more complex architectures
inputs = tf.keras.Input(shape=(784,))
x = tf.keras.layers.Dense(128, activation='relu')(inputs)
x = tf.keras.layers.Dropout(0.2)(x)
x = tf.keras.layers.Dense(64, activation='relu')(x)
outputs = tf.keras.layers.Dense(10, activation='softmax')(x)

model = tf.keras.Model(inputs=inputs, outputs=outputs)

# Multi-input model
input1 = tf.keras.Input(shape=(64,))
input2 = tf.keras.Input(shape=(32,))
x1 = tf.keras.layers.Dense(64, activation='relu')(input1)
x2 = tf.keras.layers.Dense(32, activation='relu')(input2)
merged = tf.keras.layers.Concatenate()([x1, x2])
outputs = tf.keras.layers.Dense(1, activation='sigmoid')(merged)

model = tf.keras.Model(inputs=[input1, input2], outputs=outputs)
```

### Model Subclassing

Complete control over model architecture by defining custom forward passes. Most flexible approach for unique behaviors.

```python
class MyModel(tf.keras.Model):
    def __init__(self, num_classes=10):
        super(MyModel, self).__init__()
        self.dense1 = tf.keras.layers.Dense(128, activation='relu')
        self.dropout = tf.keras.layers.Dropout(0.2)
        self.dense2 = tf.keras.layers.Dense(num_classes, activation='softmax')

    def call(self, inputs, training=False):
        x = self.dense1(inputs)
        x = self.dropout(x, training=training)
        return self.dense2(x)

# Create and use the model
model = MyModel(num_classes=10)
```

## Training

### Compile Model

Configures model for training by specifying optimizer, loss function, and metrics. Prepares model to learn from data.

```python
model.compile(
    optimizer='adam',                    # Optimizer
    loss='sparse_categorical_crossentropy',  # Loss function
    metrics=['accuracy']                 # Metrics to track
)

# Custom optimizer configuration
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=[tf.keras.metrics.SparseCategoricalAccuracy()]
)
```

### Train Model

Model learns patterns from data. Handles forward passes, loss calculation, and weight updates automatically.

```python
# Train the model
history = model.fit(
    x_train, y_train,
    batch_size=32,
    epochs=10,
    validation_data=(x_val, y_val),
    verbose=1,
    shuffle=True
)

# Train with callbacks
callbacks = [
    tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
    tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=3),
    tf.keras.callbacks.ModelCheckpoint('best_model.h5', save_best_only=True)
]

history = model.fit(
    x_train, y_train,
    batch_size=32,
    epochs=100,
    validation_data=(x_val, y_val),
    callbacks=callbacks
)
```

### Custom Training Loop

Fine-grained control over the training process. Use for custom loss calculations or complex training procedures.

```python
# Custom training loop with GradientTape
optimizer = tf.keras.optimizers.Adam()
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

@tf.function
def train_step(x, y):
    with tf.GradientTape() as tape:
        predictions = model(x, training=True)
        loss = loss_fn(y, predictions)

    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss

# Training loop
for epoch in range(epochs):
    for step, (x_batch, y_batch) in enumerate(train_dataset):
        loss = train_step(x_batch, y_batch)
        if step % 100 == 0:
            print(f"Epoch {epoch}, Step {step}, Loss: {loss:.4f}")
```

## Data Processing

### tf.data API

Efficient data pipelines for feeding data to models. Handles batching, shuffling, and preprocessing with automatic optimization.

```python
# Create dataset from tensors
dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))

# Create dataset from generator
def data_generator():
    for i in range(1000):
        yield (tf.random.normal([28, 28]), tf.random.uniform([1], maxval=10, dtype=tf.int32))

dataset = tf.data.Dataset.from_generator(
    data_generator,
    output_signature=(
        tf.TensorSpec(shape=[28, 28], dtype=tf.float32),
        tf.TensorSpec(shape=[1], dtype=tf.int32)
    )
)

# Dataset operations
dataset = dataset.batch(32)                    # Batch data
dataset = dataset.shuffle(buffer_size=1000)    # Shuffle data
dataset = dataset.prefetch(tf.data.AUTOTUNE)   # Prefetch for performance
dataset = dataset.repeat()                     # Repeat dataset

# Map transformations
def preprocess(image, label):
    image = tf.cast(image, tf.float32) / 255.0
    return image, label

dataset = dataset.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)

# Filter data
dataset = dataset.filter(lambda x, y: tf.reduce_sum(x) > 0)
```

### Image Processing

Loads, resizes, and augments images for computer vision tasks. Prepares image data in the right format and size.

```python
# Load and preprocess images
image = tf.io.read_file('image.jpg')
image = tf.image.decode_jpeg(image, channels=3)
image = tf.image.resize(image, [224, 224])
image = tf.cast(image, tf.float32) / 255.0

# Data augmentation
def augment_image(image, label):
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_brightness(image, max_delta=0.2)
    image = tf.image.random_contrast(image, lower=0.8, upper=1.2)
    image = tf.image.random_saturation(image, lower=0.8, upper=1.2)
    return image, label

# Apply augmentation to dataset
augmented_dataset = dataset.map(augment_image)
```

### Text Processing

Converts raw text into numerical format for neural networks. Includes tokenization, padding, and vectorization for NLP.

```python
# Tokenization
tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=10000)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)

# Padding sequences
padded_sequences = tf.keras.preprocessing.sequence.pad_sequences(
    sequences, maxlen=100, padding='post', truncating='post'
)

# Using TextVectorization layer
vectorize_layer = tf.keras.layers.TextVectorization(
    max_tokens=10000,
    output_sequence_length=100
)
vectorize_layer.adapt(text_dataset)
```

## Saving & Loading

### Save/Load Entire Model

Preserves architecture, weights, and training configuration. Easiest way to deploy models or continue training.

```python
# Save entire model
model.save('my_model.h5')                    # HDF5 format
model.save('my_model')                       # SavedModel format

# Load model
loaded_model = tf.keras.models.load_model('my_model.h5')
loaded_model = tf.keras.models.load_model('my_model')
```

### Save/Load Weights Only

Preserves learned parameters while recreating architecture programmatically. More memory efficient than saving entire models.

```python
# Save weights
model.save_weights('model_weights.h5')

# Load weights
model.load_weights('model_weights.h5')

# Create model with same architecture, then load weights
new_model = create_model()  # Function that creates the same architecture
new_model.load_weights('model_weights.h5')
```

### Checkpoints

Automatically saves model during training. Allows recovery from interruptions or reverting to best performance.

```python
# Create checkpoint callback
checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath='checkpoint_epoch_{epoch:02d}_val_loss_{val_loss:.2f}.h5',
    save_weights_only=True,
    save_best_only=True,
    monitor='val_loss',
    mode='min'
)

# Use in training
model.fit(x_train, y_train, callbacks=[checkpoint_callback])

# Manual checkpointing
checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)
checkpoint_manager = tf.train.CheckpointManager(
    checkpoint, directory='./checkpoints', max_to_keep=3
)

# Save checkpoint
checkpoint_manager.save()

# Restore checkpoint
checkpoint.restore(tf.train.latest_checkpoint('./checkpoints'))
```

## Debugging

### Debugging Tools

Identifies and fixes model issues. Provides insights into tensor values, training progress, and performance bottlenecks.

```python
# Enable eager execution (default in TF 2.x)
tf.config.run_functions_eagerly(True)

# Print tensor values
tf.print("Tensor value:", tensor)

# Debug callback
class DebugCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        print(f"Epoch {epoch}: {logs}")

# TensorBoard
tensorboard_callback = tf.keras.callbacks.TensorBoard(
    log_dir='./logs',
    histogram_freq=1,
    write_graph=True,
    write_images=True
)

# Profiler
tf.profiler.experimental.start('logdir')
# Your training code here
tf.profiler.experimental.stop()
```

### Common Issues and Solutions

Diagnoses and fixes frequent TensorFlow problems like NaN values, shape mismatches, and memory issues.

```python
# Check for NaN values
tf.debugging.check_numerics(tensor, "Tensor contains NaN or Inf")

# Assert tensor shapes
tf.debugging.assert_equal(tf.shape(tensor), [batch_size, features])

# Memory growth for GPU
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    tf.config.experimental.set_memory_growth(gpus[0], True)

# Mixed precision training
policy = tf.keras.mixed_precision.Policy('mixed_float16')
tf.keras.mixed_precision.set_global_policy(policy)
```

## Common Patterns

### Transfer Learning

Leverages pre-trained models to solve new problems faster with less data. Powerful for computer vision and NLP tasks.

```python
# Load pre-trained model
base_model = tf.keras.applications.VGG16(
    weights='imagenet',
    include_top=False,
    input_shape=(224, 224, 3)
)

# Freeze base model
base_model.trainable = False

# Add custom layers
model = tf.keras.Sequential([
    base_model,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(num_classes, activation='softmax')
])

# Fine-tuning: unfreeze some layers
base_model.trainable = True
for layer in base_model.layers[:-10]:
    layer.trainable = False
```

### Custom Metrics

Tracks performance measures not built into TensorFlow. Useful for domain-specific evaluations and research experiments.

```python
class F1Score(tf.keras.metrics.Metric):
    def __init__(self, name='f1_score', **kwargs):
        super(F1Score, self).__init__(name=name, **kwargs)
        self.precision = tf.keras.metrics.Precision()
        self.recall = tf.keras.metrics.Recall()

    def update_state(self, y_true, y_pred, sample_weight=None):
        self.precision.update_state(y_true, y_pred, sample_weight)
        self.recall.update_state(y_true, y_pred, sample_weight)

    def result(self):
        p = self.precision.result()
        r = self.recall.result()
        return 2 * ((p * r) / (p + r + tf.keras.backend.epsilon()))

    def reset_state(self):
        self.precision.reset_state()
        self.recall.reset_state()

# Use custom metric
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=[F1Score()]
)
```

### Learning Rate Scheduling

Adjusts learning rate during training to improve convergence and final performance. Different schedules for different problems.

```python
# Step decay
def step_decay(epoch):
    initial_lr = 0.01
    drop = 0.5
    epochs_drop = 10.0
    lr = initial_lr * (drop ** (epoch // epochs_drop))
    return lr

lr_scheduler = tf.keras.callbacks.LearningRateScheduler(step_decay)

# Exponential decay
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=0.01,
    decay_steps=1000,
    decay_rate=0.9
)

optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

# Cosine decay
lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
    initial_learning_rate=0.01,
    decay_steps=1000
)
```

### Model Ensembling

Combines predictions from multiple models for better performance. Powerful technique for competitions and production systems.

```python
# Simple averaging ensemble
def ensemble_predictions(models, x):
    predictions = []
    for model in models:
        pred = model.predict(x)
        predictions.append(pred)

    # Average predictions
    ensemble_pred = tf.reduce_mean(predictions, axis=0)
    return ensemble_pred

# Weighted ensemble
def weighted_ensemble(models, weights, x):
    predictions = []
    for model, weight in zip(models, weights):
        pred = model.predict(x) * weight
        predictions.append(pred)

    ensemble_pred = tf.reduce_sum(predictions, axis=0)
    return ensemble_pred
```

---

## Quick Reference

### Common Imports

```python
import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
```

### Model Building Template

```python
# 1. Create model
model = tf.keras.Sequential([
    layers.Dense(64, activation='relu', input_shape=(features,)),
    layers.Dropout(0.2),
    layers.Dense(num_classes, activation='softmax')
])

# 2. Compile
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# 3. Train
history = model.fit(
    x_train, y_train,
    batch_size=32,
    epochs=10,
    validation_data=(x_val, y_val)
)

# 4. Evaluate
test_loss, test_accuracy = model.evaluate(x_test, y_test)

# 5. Predict
predictions = model.predict(x_new)
```

### Essential Callbacks

```python
callbacks = [
    tf.keras.callbacks.EarlyStopping(patience=5),
    tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=3),
    tf.keras.callbacks.ModelCheckpoint('best_model.h5', save_best_only=True),
    tf.keras.callbacks.TensorBoard(log_dir='./logs')
]
```

This cheat sheet covers the most important aspects of TensorFlow for deep learning. Keep it handy for quick reference while developing your models!
