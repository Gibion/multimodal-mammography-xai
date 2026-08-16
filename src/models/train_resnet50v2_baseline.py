from pathlib import Path
import json
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

PROJECT_ROOT = Path.cwd()
MANIFEST_FILE = Path('data/processed/cbis_ddsm/manifests/cbis_ddsm_full_image_manifest_preprocessed_v4.csv')
MODEL_DIR = Path('models/resnet50v2_baseline')
RESULTS_DIR = Path('results/resnet50v2_baseline')
FIGURES_DIR = RESULTS_DIR / 'figures'
LOGS_DIR = RESULTS_DIR / 'logs'

IMAGE_SIZE = 512
BATCH_SIZE = 4
EPOCHS = 15
RANDOM_SEED = 42
INITIAL_LEARNING_RATE = 1e-4
EARLY_STOPPING_PATIENCE = 4
LR_PATIENCE = 2
LR_FACTOR = 0.5
MIN_LEARNING_RATE = 1e-7
AUTOTUNE = tf.data.AUTOTUNE

tf.keras.utils.set_random_seed(RANDOM_SEED)

for d in [MODEL_DIR, RESULTS_DIR, FIGURES_DIR, LOGS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ---------------- Manifest ----------------
df = pd.read_csv(MANIFEST_FILE)
required = ['preprocessed_path', 'preprocessing_status', 'binary_pathology', 'split', 'patient_id', 'image_id']
missing = [c for c in required if c not in df.columns]
if missing:
    raise ValueError('Manifest missing columns: ' + ', '.join(missing))

df = df[df['preprocessing_status'] == 'success'].copy()

def resolve_path(value):
    p = Path(str(value))
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    return str(p.resolve())

df['resolved_path'] = df['preprocessed_path'].apply(resolve_path)
exists = df['resolved_path'].apply(os.path.exists)

print('=' * 72)
print('RESNET50V2 BASELINE TRAINING')
print('=' * 72)
print(f'\nManifest rows after filtering: {len(df):,}')
print(f'Source images found: {exists.sum():,} / {len(exists):,}')
if not exists.all():
    raise FileNotFoundError(f'{(~exists).sum():,} source images missing.')

label_map = {'BENIGN': 0, 'MALIGNANT': 1}
df['label'] = df['binary_pathology'].astype(str).str.upper().map(label_map)
if df['label'].isna().any():
    raise ValueError('Unexpected pathology labels present.')
df['label'] = df['label'].astype(np.float32)

train_df = df[df['split'] == 'train'].copy()
validation_df = df[df['split'] == 'validation'].copy()
test_df = df[df['split'] == 'test'].copy()

print('\nSplit counts:')
print(df['split'].value_counts())
print('\nClass counts by split:')
print(pd.crosstab(df['split'], df['binary_pathology']))

# ---------------- Input pipeline ----------------
def load_image(path, label):
    image = tf.io.decode_jpeg(tf.io.read_file(path), channels=1)
    image = tf.image.convert_image_dtype(image, tf.float32) * 255.0
    image = tf.ensure_shape(image, [IMAGE_SIZE, IMAGE_SIZE, 1])
    image = tf.image.grayscale_to_rgb(image)
    return image, label

augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomRotation(7.0/360.0, fill_mode='constant', fill_value=0.0, seed=RANDOM_SEED),
    tf.keras.layers.RandomTranslation(0.05, 0.05, fill_mode='constant', fill_value=0.0, seed=RANDOM_SEED+1),
    tf.keras.layers.RandomZoom(height_factor=(-0.08, 0.08), width_factor=(-0.08, 0.08), fill_mode='constant', fill_value=0.0, seed=RANDOM_SEED+2),
    tf.keras.layers.RandomContrast(0.08, seed=RANDOM_SEED+3),
], name='training_augmentation')
augmentation.build((None, IMAGE_SIZE, IMAGE_SIZE, 3))

def augment_image(image, label):
    image = tf.expand_dims(image, 0)
    image = augmentation(image, training=True)
    image = tf.squeeze(image, 0)
    image = tf.clip_by_value(image, 0.0, 255.0)
    return image, label

def resnet_preprocess(image, label):
    return tf.keras.applications.resnet_v2.preprocess_input(image), label

def build_dataset(frame, training=False):
    ds = tf.data.Dataset.from_tensor_slices((frame['resolved_path'].to_numpy(), frame['label'].to_numpy(dtype=np.float32)))
    if training:
        ds = ds.shuffle(len(frame), seed=RANDOM_SEED, reshuffle_each_iteration=True)
    ds = ds.map(load_image, num_parallel_calls=AUTOTUNE)
    if training:
        ds = ds.map(augment_image, num_parallel_calls=AUTOTUNE)
    ds = ds.map(resnet_preprocess, num_parallel_calls=AUTOTUNE)
    ds = ds.batch(BATCH_SIZE).prefetch(AUTOTUNE)
    return ds

train_dataset = build_dataset(train_df, training=True)
validation_dataset = build_dataset(validation_df)
test_dataset = build_dataset(test_df)

# ---------------- Model ----------------
print('\nBuilding ResNet50V2 baseline...')
base_model = tf.keras.applications.ResNet50V2(
    include_top=False,
    weights='imagenet',
    input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3)
)
base_model.trainable = False

inputs = tf.keras.Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 3), name='mammogram')
x = base_model(inputs, training=False)
x = tf.keras.layers.GlobalAveragePooling2D(name='global_average_pooling')(x)
x = tf.keras.layers.Dense(256, activation='relu', name='dense_256')(x)
x = tf.keras.layers.Dropout(0.3, name='dropout')(x)
outputs = tf.keras.layers.Dense(1, activation='sigmoid', name='prediction')(x)
model = tf.keras.Model(inputs, outputs, name='resnet50v2_baseline')

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=INITIAL_LEARNING_RATE),
    loss=tf.keras.losses.BinaryCrossentropy(),
    metrics=[
        tf.keras.metrics.BinaryAccuracy(name='accuracy'),
        tf.keras.metrics.AUC(name='auc'),
        tf.keras.metrics.Precision(name='precision'),
        tf.keras.metrics.Recall(name='recall'),
    ],
)
model.summary()

with (RESULTS_DIR/'model_summary.txt').open('w', encoding='utf-8') as f:
    model.summary(print_fn=lambda line: f.write(line + '\n'))

best_model_path = MODEL_DIR/'best_model.keras'
callbacks = [
    tf.keras.callbacks.ModelCheckpoint(best_model_path, monitor='val_auc', mode='max', save_best_only=True, verbose=1),
    tf.keras.callbacks.EarlyStopping(monitor='val_auc', mode='max', patience=EARLY_STOPPING_PATIENCE, restore_best_weights=True, verbose=1),
    tf.keras.callbacks.ReduceLROnPlateau(monitor='val_auc', mode='max', factor=LR_FACTOR, patience=LR_PATIENCE, min_lr=MIN_LEARNING_RATE, verbose=1),
    tf.keras.callbacks.CSVLogger(LOGS_DIR/'training_log.csv', append=False),
]

# ---------------- Training ----------------
print('\nStarting training...')
history = model.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=EPOCHS,
    callbacks=callbacks,
    verbose=1,
)

final_model_path = MODEL_DIR/'final_model.keras'
model.save(final_model_path)

history_df = pd.DataFrame(history.history)
history_df.to_csv(RESULTS_DIR/'training_history.csv', index=False)

# ---------------- Curves ----------------
def save_metric_plot(metric, title, filename):
    val_metric = f'val_{metric}'
    if metric not in history_df.columns:
        return
    fig = plt.figure(figsize=(7, 5))
    plt.plot(history_df.index + 1, history_df[metric], label='Training')
    if val_metric in history_df.columns:
        plt.plot(history_df.index + 1, history_df[val_metric], label='Validation')
    plt.xlabel('Epoch')
    plt.ylabel(metric.replace('_', ' ').title())
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    fig.savefig(FIGURES_DIR/filename, dpi=180, bbox_inches='tight')
    plt.close(fig)

save_metric_plot('loss', 'Training and Validation Loss', 'loss_curve.png')
save_metric_plot('auc', 'Training and Validation AUC', 'auc_curve.png')
save_metric_plot('accuracy', 'Training and Validation Accuracy', 'accuracy_curve.png')

# ---------------- Evaluation ----------------
print('\nEvaluating on validation set...')
validation_results = model.evaluate(validation_dataset, verbose=1, return_dict=True)
print('\nValidation results:')
for k, v in validation_results.items():
    print(f'{k:12s}: {v:.4f}')

print('\nEvaluating on held-out test set...')
test_results = model.evaluate(test_dataset, verbose=1, return_dict=True)
print('\nTest results:')
for k, v in test_results.items():
    print(f'{k:12s}: {v:.4f}')

metrics = {
    'configuration': {
        'image_size': IMAGE_SIZE,
        'batch_size': BATCH_SIZE,
        'epochs_requested': EPOCHS,
        'epochs_completed': len(history_df),
        'initial_learning_rate': INITIAL_LEARNING_RATE,
        'random_seed': RANDOM_SEED,
        'backbone': 'ResNet50V2',
        'backbone_trainable': False,
        'augmentation': {
            'rotation_degrees': 7,
            'translation_fraction': 0.05,
            'zoom_fraction': 0.08,
            'contrast_fraction': 0.08,
            'horizontal_flip': False,
            'vertical_flip': False,
        },
    },
    'validation': {k: float(v) for k, v in validation_results.items()},
    'test': {k: float(v) for k, v in test_results.items()},
}
with (RESULTS_DIR/'metrics.json').open('w', encoding='utf-8') as f:
    json.dump(metrics, f, indent=2)

best_epoch = int(history_df['val_auc'].idxmax() + 1)
best_val_auc = float(history_df['val_auc'].max())

print('\n' + '=' * 72)
print('BASELINE TRAINING COMPLETE')
print('=' * 72)
print(f'\nEpochs completed: {len(history_df)}')
print(f'Best validation AUC: {best_val_auc:.4f}')
print(f'Best validation epoch: {best_epoch}')
print('\nSaved best model:')
print(best_model_path.resolve())
print('\nSaved final model:')
print(final_model_path.resolve())
print('\nSaved results:')
print(RESULTS_DIR.resolve())
