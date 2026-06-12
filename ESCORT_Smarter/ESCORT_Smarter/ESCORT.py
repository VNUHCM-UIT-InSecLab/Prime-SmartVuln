import tensorflow as tf
from tensorflow.keras.layers import Input, Embedding, GRU, Dense, Dropout, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.callbacks import ModelCheckpoint
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences
from sklearn.metrics import classification_report, precision_score, recall_score, f1_score
import glob
import os
import re

def get_latest_checkpoint(pattern):
    checkpoints = glob.glob(pattern)
    if not checkpoints:
        return None
    return max(checkpoints, key=os.path.getctime)

def get_epoch_from_checkpoint(checkpoint_path):
    match = re.search(r'epoch_(\d+)', checkpoint_path)
    if match:
        return int(match.group(1))
    return 0

def get_needed_label(num_vuln):
    label_columns = ['Other', 'access_control', 'arithmetic', 'denial_service', 'front_running', 'reentrancy', 'time_manipulation', 'unchecked_low_calls']
    needed_labels = []
    for i in range(num_vuln):
        needed_labels.append(label_columns[i])
    return needed_labels

def ESCORT(num_vuln):
    max_sequence_length = 17500
    vocab_size = 256  # Example vocabulary size
    embedding_dim = 20
    gru_units = 128
    dropout_rate = 0.5

    # Input layer
    inputs = Input(shape=(max_sequence_length,), dtype='int64')

    # Embedding layer
    embedding = Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_sequence_length)(inputs)

    # GRU layer
    gru = GRU(units=gru_units)(embedding)

    outputs = []
    # Common feature extractor
    feature_extractor = BatchNormalization()(gru)
    feature_extractor = Dropout(rate=dropout_rate)(feature_extractor)

    for _ in range(num_vuln):
        branch = Dense(units=128, activation='relu')(feature_extractor)
        branch = BatchNormalization()(branch)
        branch = Dropout(rate=dropout_rate)(branch)
        output = Dense(units=1, activation='sigmoid')(branch)
        outputs.append(output)

    # Multi-output model
    model = Model(inputs=inputs, outputs=outputs)

    METRICS = [
        [
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall')
        ]
        for _ in range(num_vuln)
    ]
    # Compile the model
    adam = tf.keras.optimizers.Adam(learning_rate=0.001)
    bce = BinaryCrossentropy()
    model.compile(optimizer=adam, loss=bce, metrics=METRICS)

    return model

def train_initial_model(data):
    num_vuln_initial = 8
    labels = get_needed_label(num_vuln_initial)

    X = data['opcode'].values
    y = data[labels].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    tokenizer = Tokenizer(num_words=256)
    tokenizer.fit_on_texts(X_train)
    sequences = tokenizer.texts_to_sequences(X_train)
    sequences_test = tokenizer.texts_to_sequences(X_test)
    X_train = pad_sequences(sequences, maxlen=17500, padding='post')
    X_test = pad_sequences(sequences_test, maxlen=17500, padding='post')
    y_train_list = [y_train[:, i].reshape((-1, 1)) for i in range(num_vuln_initial)]
    y_test_list = [y_test[:, i].reshape((-1, 1)) for i in range(num_vuln_initial)]

    # Check for existing checkpoint
    latest_checkpoint = get_latest_checkpoint('ESCORT_8_labels_epoch_*.h5')
    initial_epoch = 0

    if latest_checkpoint:
        print(f"Found checkpoint: {latest_checkpoint}")
        model_initial = tf.keras.models.load_model(latest_checkpoint)
        completed_epoch = get_epoch_from_checkpoint(latest_checkpoint)
        initial_epoch = completed_epoch
        print(f"Resuming training from epoch {initial_epoch}")
    else:
        model_initial = ESCORT(num_vuln_initial)
        print("Starting training from epoch 0")

    # Train the model with 8 labels
    checkpoint = ModelCheckpoint('ESCORT_8_labels_epoch_{epoch:02d}.h5', save_freq='epoch', verbose=1)
    model_initial.fit(X_train, y_train_list, epochs=10, batch_size=32, validation_data=(X_test, y_test_list),
                      callbacks=[checkpoint], initial_epoch=initial_epoch)
    # Save the weights
    model_initial.save_weights('model_common_weights.h5')

    # Save the model
    model_initial.save("ESCORT_Smarter_8_labels.h5")

    # Evaluate the model
    initial_results = model_initial.evaluate(X_test, y_test_list)
    print("Initial Model Evaluation Results:", initial_results)

    # Classification report
    y_pred_initial = model_initial.predict(X_test)
    for i, label in enumerate(labels):
        y_pred_binary = (y_pred_initial[i] > 0.5).astype(int)
        y_true = y_test_list[i].flatten()
        print(f"Classification Report for {label}:")
        print(classification_report(y_true, y_pred_binary))
        print(f"Weighted Avg Precision: {precision_score(y_true, y_pred_binary, average='weighted', zero_division=0):.4f}")
        print(f"Weighted Avg Recall: {recall_score(y_true, y_pred_binary, average='weighted', zero_division=0):.4f}")
        print(f"Weighted Avg F1: {f1_score(y_true, y_pred_binary, average='weighted', zero_division=0):.4f}")
        print(f"Micro Avg Precision: {precision_score(y_true, y_pred_binary, average='micro', zero_division=0):.4f}")
        print(f"Micro Avg Recall: {recall_score(y_true, y_pred_binary, average='micro', zero_division=0):.4f}")
        print(f"Micro Avg F1: {f1_score(y_true, y_pred_binary, average='micro', zero_division=0):.4f}")
        print(f"Macro Avg Precision: {precision_score(y_true, y_pred_binary, average='macro', zero_division=0):.4f}")
        print(f"Macro Avg Recall: {recall_score(y_true, y_pred_binary, average='macro', zero_division=0):.4f}")
        print(f"Macro Avg F1: {f1_score(y_true, y_pred_binary, average='macro', zero_division=0):.4f}")
        print("-" * 50)

    return model_initial

def ESCORT_transfer_learning(num_vuln):
    labels = get_needed_label(num_vuln)

    max_sequence_length = 17500
    vocab_size = 256  # Example vocabulary size
    embedding_dim = 20
    gru_units = 128
    dropout_rate = 0.5

    X = data['opcode'].values
    y = data[labels].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    print("TOKENIZING DATA......")
    tokenizer = Tokenizer(num_words=256)
    tokenizer.fit_on_texts(X_train)
    sequences = tokenizer.texts_to_sequences(X_train)
    sequences_test = tokenizer.texts_to_sequences(X_test)

    X_train = pad_sequences(sequences, maxlen=17500, padding='post')
    X_test = pad_sequences(sequences_test, maxlen=17500, padding='post')

    y_train_list = [y_train[:, i].reshape((-1, 1)) for i in range(num_vuln)]
    y_test_list = [y_test[:, i].reshape((-1, 1)) for i in range(num_vuln)]

    # Input layer
    inputs = Input(shape=(max_sequence_length,), dtype='int64')

    # Embedding layer
    embedding = Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_sequence_length)(inputs)

    # GRU layer
    gru = GRU(units=gru_units)(embedding)

    outputs = []
    # Common feature extractor
    feature_extractor = BatchNormalization()(gru)
    feature_extractor = Dropout(rate=dropout_rate)(feature_extractor)

    for _ in range(num_vuln):
        branch = Dense(units=128, activation='relu')(feature_extractor)
        branch = BatchNormalization()(branch)
        branch = Dropout(rate=dropout_rate)(branch)
        output = Dense(units=1, activation='sigmoid')(branch)
        outputs.append(output)

    # Multi-output model
    model = Model(inputs=inputs, outputs=outputs)

    METRICS = [
        tf.keras.metrics.BinaryAccuracy(name='accuracy'),
        tf.keras.metrics.Precision(name='precision'),
        tf.keras.metrics.Recall(name='recall')
    ]
    # Compile the model
    adam = tf.keras.optimizers.Adam(learning_rate=0.001)
    bce = BinaryCrossentropy()
    model.compile(optimizer=adam, loss=bce, metrics=METRICS)

    return model

def train_transfer_learning_model(data):
    num_vuln_final = 8
    labels = get_needed_label(num_vuln_final)

    X = data['opcode'].values
    y = data[labels].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("TOKENIZING DATA......")
    tokenizer = Tokenizer(num_words=256)
    tokenizer.fit_on_texts(X_train)
    sequences = tokenizer.texts_to_sequences(X_train)
    sequences_test = tokenizer.texts_to_sequences(X_test)
    X_train = pad_sequences(sequences, maxlen=17500, padding='post')
    X_test = pad_sequences(sequences_test, maxlen=17500, padding='post')

    y_train_list = [y_train[:, i].reshape((-1, 1)) for i in range(num_vuln_final)]
    y_test_list = [y_test[:, i].reshape((-1, 1)) for i in range(num_vuln_final)]

    # Check for existing checkpoint
    latest_checkpoint = get_latest_checkpoint('ESCORT_transfer_epoch_*.h5')
    initial_epoch = 0

    if latest_checkpoint:
        print(f"Found checkpoint: {latest_checkpoint}")
        model_final = tf.keras.models.load_model(latest_checkpoint)
        completed_epoch = get_epoch_from_checkpoint(latest_checkpoint)
        initial_epoch = completed_epoch
        print(f"Resuming training from epoch {initial_epoch}")
    else:
        model_final = ESCORT_transfer_learning(num_vuln_final)
        # Load the common layers' weights from the initial model
        model_final.load_weights('model_common_weights.h5', by_name=True, skip_mismatch=True)
        print("Starting training from epoch 0")

    checkpoint = ModelCheckpoint('ESCORT_transfer_epoch_{epoch:02d}.h5', save_freq='epoch', verbose=1)
    model_final.fit(X_train, y_train_list, epochs=10, batch_size=32, validation_data=(X_test, y_test_list),
                    callbacks=[checkpoint], initial_epoch=initial_epoch)
    
    # Save the final model
    model_final.save("ESCORT_Smarter_all_labels.h5")

    # Evaluate the model
    final_results = model_final.evaluate(X_test, y_test_list)
    print("Final Model Evaluation Results:", final_results)

    # Classification report
    y_pred_final = model_final.predict(X_test)
    for i, label in enumerate(labels):
        print(f"Classification Report for {label}:")
        print(classification_report(y_test_list[i], (y_pred_final[i] > 0.5).astype(int)))

    return model_final

# LOAD DATA
print("LOADING DATA.....")
data = pd.read_csv("Dataset/multilabel_BILSTM_BERT.csv")
data = data.drop(columns='Unnamed: 0')
label_columns = ['Other', 'access_control', 'arithmetic', 'denial_service', 'front_running', 'reentrancy', 'time_manipulation', 'unchecked_low_calls']
for column in label_columns:
    data[column] = data[column].apply(lambda x: 1 if x != 0 else 0)

# Train initial model with 4 labels
model_4_labels = train_initial_model(data)

# Expand to 8 labels and continue training
#model_8_labels = train_transfer_learning_model(data)


