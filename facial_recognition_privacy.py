"""
Facial Recognition with Privacy Protection
Assignment: AT&T Face Dataset with CNN and Privacy Methods
Author: Chloe Partrick

This script implements:
1. CNN-based facial recognition for AT&T dataset
2. Privacy protection methods: Basic Pixelization and DP-Pix
3. Training and testing with different privacy levels
"""

import numpy as np
import os
import glob
from PIL import Image
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)


class ATTFaceDataset:
    """
    Load and preprocess the AT&T face dataset.
    Dataset contains 40 subjects with 10 images each (92x112 pixels).
    """
    
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        self.images = []
        self.labels = []
        self.img_height = 112
        self.img_width = 92
        
    def load_data(self):
        """Load all images from the AT&T face dataset."""
        print("Loading AT&T face dataset...")
        
        # Iterate through all 40 subjects (s1 to s40)
        for subject_id in range(1, 41):
            subject_dir = os.path.join(self.dataset_path, f's{subject_id}')
            
            # Load all 10 images for each subject
            for img_num in range(1, 11):
                img_path = os.path.join(subject_dir, f'{img_num}.pgm')
                
                if os.path.exists(img_path):
                    # Read image as grayscale
                    img = Image.open(img_path)
                    img_array = np.array(img, dtype=np.float32)
                    
                    # Normalize to [0, 1]
                    img_array = img_array / 255.0
                    
                    self.images.append(img_array)
                    self.labels.append(subject_id - 1)  # Labels from 0 to 39
        
        self.images = np.array(self.images)
        self.labels = np.array(self.labels)
        
        print(f"Loaded {len(self.images)} images from {len(np.unique(self.labels))} subjects")
        print(f"Image shape: {self.images[0].shape}")
        
        return self.images, self.labels
    
    def get_train_test_split(self, test_size=0.2, random_state=42):
        """
        Split data into train and test sets.
        Following the DBSec paper approach with stratified split.
        """
        X_train, X_test, y_train, y_test = train_test_split(
            self.images, self.labels, 
            test_size=test_size, 
            random_state=random_state,
            stratify=self.labels
        )
        
        print(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}")
        
        return X_train, X_test, y_train, y_test


class PrivacyProtection:
    """
    Implement privacy protection methods for facial images.
    Methods: Basic Pixelization and DP-Pix (Differential Privacy Pixelization)
    """
    
    @staticmethod
    def basic_pixelization(image, block_size=16):
        """
        Apply basic pixelization (mosaicing) to an image.
        
        Args:
            image: Input image (2D numpy array)
            block_size: Size of pixelization blocks (b parameter)
            
        Returns:
            Pixelized image
        """
        h, w = image.shape
        pixelized = np.copy(image)
        
        # Process each block
        for i in range(0, h, block_size):
            for j in range(0, w, block_size):
                # Get the block boundaries
                block_h = min(block_size, h - i)
                block_w = min(block_size, w - j)
                
                # Extract block
                block = image[i:i+block_h, j:j+block_w]
                
                # Replace with mean value of the block
                mean_val = np.mean(block)
                pixelized[i:i+block_h, j:j+block_w] = mean_val
        
        return pixelized
    
    @staticmethod
    def dp_pix(image, block_size=16, mosaic_size=16, epsilon=1.0):
        """
        Apply DP-Pix (Differential Privacy Pixelization) to an image.
        
        Args:
            image: Input image (2D numpy array, values in [0, 1])
            block_size: Size of blocks for initial averaging (b parameter)
            mosaic_size: Size of mosaic blocks (m parameter)
            epsilon: Privacy parameter (smaller = more privacy, more noise)
            
        Returns:
            DP-Pix protected image
        """
        h, w = image.shape
        dp_image = np.copy(image)
        
        # Sensitivity for pixel values normalized to [0, 1]
        # Assuming sensitivity = 1 (max difference in pixel value)
        sensitivity = 1.0
        
        # Laplace scale parameter
        scale = sensitivity / epsilon
        
        # Process each mosaic block
        for i in range(0, h, mosaic_size):
            for j in range(0, w, mosaic_size):
                # Get the mosaic block boundaries
                mosaic_h = min(mosaic_size, h - i)
                mosaic_w = min(mosaic_size, w - j)
                
                # Extract mosaic block
                mosaic_block = image[i:i+mosaic_h, j:j+mosaic_w]
                
                # Calculate mean of the mosaic block
                mean_val = np.mean(mosaic_block)
                
                # Add Laplace noise for differential privacy
                noise = np.random.laplace(0, scale)
                noisy_mean = mean_val + noise
                
                # Clip to valid range [0, 1]
                noisy_mean = np.clip(noisy_mean, 0, 1)
                
                # Apply to all pixels in the mosaic block
                dp_image[i:i+mosaic_h, j:j+mosaic_w] = noisy_mean
        
        return dp_image
    
    @staticmethod
    def apply_protection_to_dataset(X, method='none', **kwargs):
        """
        Apply privacy protection to entire dataset.
        
        Args:
            X: Dataset of images
            method: 'none', 'pixelization', or 'dp_pix'
            **kwargs: Additional parameters for the protection method
            
        Returns:
            Protected dataset
        """
        if method == 'none':
            return X
        
        protected = np.zeros_like(X)
        
        for i, img in enumerate(X):
            if method == 'pixelization':
                protected[i] = PrivacyProtection.basic_pixelization(
                    img, block_size=kwargs.get('block_size', 16)
                )
            elif method == 'dp_pix':
                protected[i] = PrivacyProtection.dp_pix(
                    img, 
                    block_size=kwargs.get('block_size', 16),
                    mosaic_size=kwargs.get('mosaic_size', 16),
                    epsilon=kwargs.get('epsilon', 1.0)
                )
        
        return protected


class FacialRecognitionCNN:
    """
    CNN architecture for facial recognition on AT&T dataset.
    Based on the architecture from the archive paper.
    """
    
    def __init__(self, input_shape=(112, 92, 1), num_classes=40):
        """
        Initialize CNN model.
        
        Args:
            input_shape: Shape of input images (height, width, channels)
            num_classes: Number of subjects to classify (40 for AT&T)
        """
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = None
        
    def build_model(self):
        """
        Build CNN architecture for facial recognition.
        Architecture based on typical face recognition CNNs from literature.
        """
        model = models.Sequential([
            # First convolutional block
            layers.Conv2D(32, (3, 3), activation='relu', padding='same', 
                         input_shape=self.input_shape),
            layers.MaxPooling2D((2, 2)),
            layers.BatchNormalization(),
            
            # Second convolutional block
            layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            layers.MaxPooling2D((2, 2)),
            layers.BatchNormalization(),
            
            # Third convolutional block
            layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            layers.MaxPooling2D((2, 2)),
            layers.BatchNormalization(),
            
            # Fourth convolutional block
            layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
            layers.MaxPooling2D((2, 2)),
            layers.BatchNormalization(),
            
            # Flatten and fully connected layers
            layers.Flatten(),
            layers.Dense(512, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(256, activation='relu'),
            layers.Dropout(0.5),
            
            # Output layer
            layers.Dense(self.num_classes, activation='softmax')
        ])
        
        # Compile model
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        self.model = model
        print("CNN model built successfully")
        print(f"Total parameters: {model.count_params():,}")
        
        return model
    
    def train(self, X_train, y_train, X_val=None, y_val=None, 
              epochs=50, batch_size=32, verbose=1):
        """
        Train the CNN model.
        
        Args:
            X_train: Training images
            y_train: Training labels
            X_val: Validation images (optional)
            y_val: Validation labels (optional)
            epochs: Number of training epochs
            batch_size: Batch size for training
            verbose: Verbosity level
            
        Returns:
            Training history
        """
        # Add channel dimension if not present
        if len(X_train.shape) == 3:
            X_train = np.expand_dims(X_train, axis=-1)
        if X_val is not None and len(X_val.shape) == 3:
            X_val = np.expand_dims(X_val, axis=-1)
        
        # Early stopping callback
        early_stopping = keras.callbacks.EarlyStopping(
            monitor='val_accuracy' if X_val is not None else 'accuracy',
            patience=10,
            restore_best_weights=True
        )
        
        # Train model
        validation_data = (X_val, y_val) if X_val is not None else None
        
        history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=validation_data,
            callbacks=[early_stopping],
            verbose=verbose
        )
        
        return history
    
    def evaluate(self, X_test, y_test):
        """
        Evaluate model on test data.
        
        Args:
            X_test: Test images
            y_test: Test labels
            
        Returns:
            Test loss and accuracy
        """
        # Add channel dimension if not present
        if len(X_test.shape) == 3:
            X_test = np.expand_dims(X_test, axis=-1)
        
        loss, accuracy = self.model.evaluate(X_test, y_test, verbose=0)
        
        return loss, accuracy


def generate_sample_images(original_image, output_dir='sample_images'):
    """
    Generate sample images showing different privacy protection methods.
    
    Args:
        original_image: Original clean image
        output_dir: Directory to save sample images
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Privacy Protection Methods Comparison', fontsize=16)
    
    # Original image
    axes[0, 0].imshow(original_image, cmap='gray')
    axes[0, 0].set_title('Original (Clean)', fontsize=12)
    axes[0, 0].axis('off')
    
    # Basic pixelization
    pixelized = PrivacyProtection.basic_pixelization(original_image, block_size=16)
    axes[0, 1].imshow(pixelized, cmap='gray')
    axes[0, 1].set_title('Basic Pixelization (b=16)', fontsize=12)
    axes[0, 1].axis('off')
    
    # DP-Pix with epsilon=0.1
    dp_01 = PrivacyProtection.dp_pix(original_image, block_size=16, mosaic_size=16, epsilon=0.1)
    axes[0, 2].imshow(dp_01, cmap='gray')
    axes[0, 2].set_title('DP-Pix (ε=0.1)', fontsize=12)
    axes[0, 2].axis('off')
    
    # DP-Pix with epsilon=1
    dp_1 = PrivacyProtection.dp_pix(original_image, block_size=16, mosaic_size=16, epsilon=1)
    axes[1, 0].imshow(dp_1, cmap='gray')
    axes[1, 0].set_title('DP-Pix (ε=1)', fontsize=12)
    axes[1, 0].axis('off')
    
    # DP-Pix with epsilon=2
    dp_2 = PrivacyProtection.dp_pix(original_image, block_size=16, mosaic_size=16, epsilon=2)
    axes[1, 1].imshow(dp_2, cmap='gray')
    axes[1, 1].set_title('DP-Pix (ε=2)', fontsize=12)
    axes[1, 1].axis('off')
    
    # DP-Pix with epsilon=4
    dp_4 = PrivacyProtection.dp_pix(original_image, block_size=16, mosaic_size=16, epsilon=4)
    axes[1, 2].imshow(dp_4, cmap='gray')
    axes[1, 2].set_title('DP-Pix (ε=4)', fontsize=12)
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'privacy_comparison.png'), dpi=300, bbox_inches='tight')
    print(f"Sample images saved to {output_dir}/privacy_comparison.png")
    
    # Save individual images as well
    plt.figure(figsize=(5, 6))
    plt.imshow(original_image, cmap='gray')
    plt.title('Original Image')
    plt.axis('off')
    plt.savefig(os.path.join(output_dir, 'original.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    plt.figure(figsize=(5, 6))
    plt.imshow(pixelized, cmap='gray')
    plt.title('Basic Pixelization (b=16)')
    plt.axis('off')
    plt.savefig(os.path.join(output_dir, 'pixelization.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    for eps in [0.1, 1, 2, 4]:
        dp_img = PrivacyProtection.dp_pix(original_image, block_size=16, mosaic_size=16, epsilon=eps)
        plt.figure(figsize=(5, 6))
        plt.imshow(dp_img, cmap='gray')
        plt.title(f'DP-Pix (ε={eps})')
        plt.axis('off')
        plt.savefig(os.path.join(output_dir, f'dp_pix_eps_{eps}.png'), dpi=150, bbox_inches='tight')
        plt.close()
    
    return


def main():
    """
    Main function to run the complete facial recognition experiment
    with different privacy protection methods.
    """
    
    # Configuration
    DATASET_PATH = os.path.expanduser('~/Downloads/att_faces')
    OUTPUT_DIR = 'results'
    SAMPLE_DIR = 'sample_images'
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(SAMPLE_DIR, exist_ok=True)
    
    print("=" * 80)
    print("Facial Recognition with Privacy Protection - AT&T Dataset")
    print("=" * 80)
    
    # Step 1: Load dataset
    print("\n[Step 1] Loading AT&T Face Dataset...")
    dataset = ATTFaceDataset(DATASET_PATH)
    images, labels = dataset.load_data()
    
    # Step 2: Train-test split (following DBSec paper approach)
    print("\n[Step 2] Creating train-test split...")
    X_train_clean, X_test_clean, y_train, y_test = dataset.get_train_test_split(
        test_size=0.2, random_state=42
    )
    
    # Step 3: Generate sample images
    print("\n[Step 3] Generating sample images...")
    sample_image = X_test_clean[0]  # Use first test image as sample
    generate_sample_images(sample_image, output_dir=SAMPLE_DIR)
    
    # Step 4: Define privacy protection configurations
    print("\n[Step 4] Setting up privacy protection experiments...")
    
    privacy_configs = [
        {'name': 'Baseline (No Protection)', 'method': 'none', 'params': {}},
        {'name': 'Basic Pixelization (b=16)', 'method': 'pixelization', 
         'params': {'block_size': 16}},
        {'name': 'DP-Pix (ε=0.1)', 'method': 'dp_pix', 
         'params': {'block_size': 16, 'mosaic_size': 16, 'epsilon': 0.1}},
        {'name': 'DP-Pix (ε=1)', 'method': 'dp_pix', 
         'params': {'block_size': 16, 'mosaic_size': 16, 'epsilon': 1}},
        {'name': 'DP-Pix (ε=2)', 'method': 'dp_pix', 
         'params': {'block_size': 16, 'mosaic_size': 16, 'epsilon': 2}},
        {'name': 'DP-Pix (ε=4)', 'method': 'dp_pix', 
         'params': {'block_size': 16, 'mosaic_size': 16, 'epsilon': 4}},
    ]
    
    # Storage for results
    results = []
    
    # Step 5: Train and test models for each privacy configuration
    print("\n[Step 5] Training and testing models...")
    print("=" * 80)
    
    for i, config in enumerate(privacy_configs, 1):
        print(f"\n[Experiment {i}/{len(privacy_configs)}] {config['name']}")
        print("-" * 80)
        
        # Apply privacy protection to both train and test sets
        print(f"Applying privacy protection: {config['method']}...")
        X_train_protected = PrivacyProtection.apply_protection_to_dataset(
            X_train_clean, method=config['method'], **config['params']
        )
        X_test_protected = PrivacyProtection.apply_protection_to_dataset(
            X_test_clean, method=config['method'], **config['params']
        )
        
        # Build and compile model
        print("Building CNN model...")
        cnn = FacialRecognitionCNN(input_shape=(112, 92, 1), num_classes=40)
        cnn.build_model()
        
        # Train model
        print("Training model...")
        history = cnn.train(
            X_train_protected, y_train,
            X_val=None,  # Can add validation if needed
            y_val=None,
            epochs=50,
            batch_size=32,
            verbose=1
        )
        
        # Evaluate on training set
        train_loss, train_accuracy = cnn.evaluate(X_train_protected, y_train)
        print(f"Training Accuracy: {train_accuracy*100:.2f}%")
        
        # Evaluate on test set
        test_loss, test_accuracy = cnn.evaluate(X_test_protected, y_test)
        print(f"Test Accuracy: {test_accuracy*100:.2f}%")
        
        # Store results
        results.append({
            'config': config['name'],
            'method': config['method'],
            'params': config['params'],
            'train_accuracy': train_accuracy,
            'test_accuracy': test_accuracy,
            'train_loss': train_loss,
            'test_loss': test_loss
        })
        
        print("-" * 80)
    
    # Step 6: Generate and save report
    print("\n[Step 6] Generating final report...")
    print("=" * 80)
    print("\nFINAL RESULTS - Facial Recognition Accuracy")
    print("=" * 80)
    print(f"{'Privacy Protection Method':<40} {'Train Acc':<12} {'Test Acc':<12}")
    print("-" * 80)
    
    for result in results:
        print(f"{result['config']:<40} {result['train_accuracy']*100:>10.2f}% {result['test_accuracy']*100:>10.2f}%")
    
    print("=" * 80)
    
    # Save results to file
    report_path = os.path.join(OUTPUT_DIR, 'results_report.txt')
    with open(report_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("Facial Recognition with Privacy Protection - Results Report\n")
        f.write("AT&T Face Dataset\n")
        f.write("=" * 80 + "\n\n")
        
        f.write("EXPERIMENT CONFIGURATION\n")
        f.write("-" * 80 + "\n")
        f.write(f"Dataset: AT&T Face Database (40 subjects, 10 images each)\n")
        f.write(f"Total images: {len(images)}\n")
        f.write(f"Train samples: {len(X_train_clean)}\n")
        f.write(f"Test samples: {len(X_test_clean)}\n")
        f.write(f"Image size: 112 x 92 pixels\n")
        f.write(f"Privacy methods tested: {len(privacy_configs)}\n\n")
        
        f.write("RESULTS - FACIAL RECOGNITION ACCURACY\n")
        f.write("=" * 80 + "\n")
        f.write(f"{'Privacy Protection Method':<40} {'Train Acc':<12} {'Test Acc':<12}\n")
        f.write("-" * 80 + "\n")
        
        for result in results:
            f.write(f"{result['config']:<40} {result['train_accuracy']*100:>10.2f}% {result['test_accuracy']*100:>10.2f}%\n")
        
        f.write("=" * 80 + "\n\n")
        
        f.write("DETAILED RESULTS\n")
        f.write("-" * 80 + "\n")
        for result in results:
            f.write(f"\n{result['config']}\n")
            f.write(f"  Method: {result['method']}\n")
            f.write(f"  Parameters: {result['params']}\n")
            f.write(f"  Training Accuracy: {result['train_accuracy']*100:.4f}%\n")
            f.write(f"  Training Loss: {result['train_loss']:.4f}\n")
            f.write(f"  Test Accuracy: {result['test_accuracy']*100:.4f}%\n")
            f.write(f"  Test Loss: {result['test_loss']:.4f}\n")
        
        f.write("\n" + "=" * 80 + "\n")
        f.write("SAMPLE IMAGES\n")
        f.write("-" * 80 + "\n")
        f.write(f"Sample images have been saved to the '{SAMPLE_DIR}' directory:\n")
        f.write(f"  - original.png: Clean sample image\n")
        f.write(f"  - pixelization.png: Basic pixelization (b=16)\n")
        f.write(f"  - dp_pix_eps_0.1.png: DP-Pix with ε=0.1\n")
        f.write(f"  - dp_pix_eps_1.png: DP-Pix with ε=1\n")
        f.write(f"  - dp_pix_eps_2.png: DP-Pix with ε=2\n")
        f.write(f"  - dp_pix_eps_4.png: DP-Pix with ε=4\n")
        f.write(f"  - privacy_comparison.png: All methods side-by-side\n")
        
    print(f"\nReport saved to: {report_path}")
    print(f"Sample images saved to: {SAMPLE_DIR}/")
    print("\n" + "=" * 80)
    print("Experiment completed successfully!")
    print("=" * 80)


if __name__ == "__main__":
    main()
