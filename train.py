import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import precision_recall_curve, roc_curve, auc, average_precision_score
from sklearn.model_selection import train_test_split
from modeling import create_model, get_custom_objects
from data_preprocessing import get_sequences_from_excel, seq_to_embeds
from tensorflow.keras.metrics import AUC

# Constants
BATCH_SIZE = 256
SEQ_MIN_SIZE = 149
EPOCHS = 150  # Increased for potential longer training
MODEL_PATH = "canya_model.h5"
LOG_FILE = "training_log.csv"

def plot_training_metrics(history):
    """Enhanced training metrics visualization"""
    metrics = ['loss', 'accuracy', 'auprc', 'auc']
    
    for metric in metrics:
        plt.figure(figsize=(10, 6))
        
        # Plot training metric
        if metric in history.history:
            plt.plot(history.history[metric], label=f'Training {metric}')
        
        # Plot validation metric if exists
        val_metric = f'val_{metric}'
        if val_metric in history.history:
            plt.plot(history.history[val_metric], label=f'Validation {metric}')
        
        plt.title(f'{metric.upper()} vs Epochs')
        plt.ylabel(metric.upper())
        plt.xlabel('Epoch')
        plt.legend()
        plt.grid(True)
        
        # Save and show
        plt.savefig(f'training_{metric}.png', bbox_inches='tight', dpi=300)
        plt.close()

def plot_roc_pr_curves(model, x_val, y_val):
    """Enhanced ROC and Precision-Recall curves with confidence intervals"""
    y_pred = model.predict(x_val).flatten()
    
    # Calculate metrics
    fpr, tpr, _ = roc_curve(y_val, y_pred)
    roc_auc = auc(fpr, tpr)
    precision, recall, _ = precision_recall_curve(y_val, y_pred)
    pr_auc = average_precision_score(y_val, y_pred)
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # ROC Curve
    ax1.plot(fpr, tpr, color='darkorange', lw=2, 
             label=f'ROC curve (AUC = {roc_auc:.3f})')
    ax1.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    ax1.set_xlim([0.0, 1.0])
    ax1.set_ylim([0.0, 1.05])
    ax1.set_xlabel('False Positive Rate')
    ax1.set_ylabel('True Positive Rate')
    ax1.set_title('Receiver Operating Characteristic')
    ax1.legend(loc="lower right")
    ax1.grid(True)
    
    # Precision-Recall Curve
    ax2.plot(recall, precision, color='blue', lw=2, 
             label=f'PR curve (AP = {pr_auc:.3f})')
    ax2.set_xlabel('Recall')
    ax2.set_ylabel('Precision')
    ax2.set_title('Precision-Recall Curve')
    ax2.legend(loc="lower left")
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig('performance_curves.png', bbox_inches='tight', dpi=300)
    plt.close()

def train_model(x_train, y_train, x_val, y_val, model_path=MODEL_PATH):
    """Enhanced training function with more options"""
    # Get custom objects from modeling.py
    custom_objects = get_custom_objects()
    
    # Create model with updated parameters
    model = create_model(
        indim=20,
        input_length=SEQ_MIN_SIZE,
        dense_number=64,  # Increased for potentially better performance 128 will put in the next time
        num_filters=100,
        filter_width=3,
        dropout_rate=0.4,
        num_heads=1,
        key_len=6,
        l2_regularization=0.01,
        pool_size=None,  # Changed to None for clarity
        act_reg=0
    )
    
    # Enhanced callbacks
    callbacks = [
        ModelCheckpoint(
            filepath=model_path,
            monitor='val_auprc',
            save_best_only=True,
            mode='max',
            save_weights_only=False,
            verbose=1
        ),
        EarlyStopping(
            monitor='val_auprc',
            patience=15,  # Increased patience
            mode='max',
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_auprc',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1,
            mode='max'
        ),
        CSVLogger(LOG_FILE, append=True)
    ]
    
    # Compile with enhanced metrics
    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=0.0005,  # Lower initial learning rate
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-07
        ),
        loss=custom_objects['binary_KL'],
        metrics=[
            'accuracy',
            AUC(name='auprc', curve='PR'),
            AUC(name='auc', curve='ROC'),
            AUC(name='auprc_macro', curve='PR', multi_label=True)
        ]
    )
    # Enhanced training with optional class weights
    class_counts = np.bincount(y_train.astype(int))
    class_weights = {i: 1./count for i, count in enumerate(class_counts)}
    
    history = model.fit(
        x_train, y_train,
        validation_data=(x_val, y_val),
        batch_size=32,
        epochs=EPOCHS,
        callbacks=callbacks,
        class_weight=class_weights,
        verbose=2  # More concise output
    )
    
    return model, history, custom_objects

def load_dataset(input_file, sheets, test_size=0.2, random_state=42):
    """Enhanced data loading with train/val split option"""
    sequences, labels = get_sequences_from_excel(input_file, sheets)
    embedded_seqs = np.array([seq_to_embeds(seq) for seq in sequences])
    labels = np.array(labels, dtype=np.float32)
    
    # If only one sheet provided, split into train/val
    if len(sheets) == 1:
        return train_test_split(
            embedded_seqs, labels,
            test_size=test_size,
            random_state=random_state,
            stratify=labels
        )
    return embedded_seqs, labels

if __name__ == "__main__":
    print("Starting training process...")
    print(f"TensorFlow version: {tf.__version__}")
    
    try:
        # Load data with enhanced options
        input_file = "GSE268261_MT_MM_TSM_BB_BL_processed_data_for_CANYA_Model.xlsx"
        x_train, y_train = load_dataset(input_file, ["NNK1", "NNK2", "NNK3"])
        x_val, y_val = load_dataset(input_file, ["NNK4(ValidationSet)"])
        
        print(f"\nData loaded successfully:")
        print(f"Training samples: {len(x_train)}")
        print(f"Validation samples: {len(x_val)}")
        print(f"Positive ratio (train): {y_train.mean():.2f}")
        print(f"Positive ratio (val): {y_val.mean():.2f}\n")
        
        # Train model
        trained_model, training_history, custom_objects = train_model(x_train, y_train, x_val, y_val)
        
        # Generate plots
        plot_training_metrics(training_history)
        plot_roc_pr_curves(trained_model, x_val, y_val)
        
        # Evaluate best model
        best_model = tf.keras.models.load_model(MODEL_PATH, custom_objects=custom_objects)
        results = best_model.evaluate(x_val, y_val, verbose=0)
        
        # Save model in multiple formats
        best_model.save(MODEL_PATH, save_format='h5')
        best_model.save('saved_model')  # SavedModel format
        print("\nModel successfully saved in both formats")
        
        # Print final metrics
        print("\nFinal Evaluation Metrics:")
        metrics = ['Loss', 'Accuracy', 'AUPRC', 'AUC-ROC']
        for name, value in zip(metrics, results):
            print(f"Validation {name}: {value:.4f}")
            
    except Exception as e:
        print(f"\nError during training: {str(e)}")
        raise