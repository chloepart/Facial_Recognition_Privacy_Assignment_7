"""
Facial Recognition with Privacy Protection - Simplified Version
Assignment: AT&T Face Dataset with Neural Network and Privacy Methods
Author: Chloe Partrick

This script implements:
1. Neural network-based facial recognition for AT&T dataset
2. Privacy protection methods: Basic Pixelization and DP-Pix
3. Training and testing with different privacy levels
"""

import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)


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


class FacialRecognitionNN:
    """
    Neural Network for facial recognition on AT&T dataset.
    Uses Multi-Layer Perceptron (MLP) architecture.
    """
    
    def __init__(self, hidden_layers=(512, 256, 128)):
        """
        Initialize Neural Network model.
        
        Args:
            hidden_layers: Tuple of hidden layer sizes
        """
        self.hidden_layers = hidden_layers
        self.model = None
        self.scaler = StandardScaler()
        
    def build_model(self):
        """
        Build Neural Network architecture for facial recognition.
        Uses MLPClassifier with multiple hidden layers.
        """
        self.model = MLPClassifier(
            hidden_layer_sizes=self.hidden_layers,
            activation='relu',
            solver='adam',
            alpha=0.0001,
            batch_size=32,
            learning_rate='adaptive',
            learning_rate_init=0.001,
            max_iter=200,
            shuffle=True,
            random_state=42,
            verbose=True,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=15
        )
        
        print(f"Neural Network model created with architecture: {self.hidden_layers}")
        return self.model
    
    def train(self, X_train, y_train):
        """
        Train the Neural Network model.
        
        Args:
            X_train: Training images (flattened)
            y_train: Training labels
            
        Returns:
            Trained model
        """
        # Flatten images
        X_train_flat = X_train.reshape(len(X_train), -1)
        
        # Standardize features
        X_train_scaled = self.scaler.fit_transform(X_train_flat)
        
        # Train model
        print("Training model...")
        self.model.fit(X_train_scaled, y_train)
        
        return self.model
    
    def evaluate(self, X_test, y_test):
        """
        Evaluate model on test data.
        
        Args:
            X_test: Test images
            y_test: Test labels
            
        Returns:
            Test accuracy
        """
        # Flatten images
        X_test_flat = X_test.reshape(len(X_test), -1)
        
        # Standardize features
        X_test_scaled = self.scaler.transform(X_test_flat)
        
        # Predict
        y_pred = self.model.predict(X_test_scaled)
        
        # Calculate accuracy
        accuracy = accuracy_score(y_test, y_pred)
        
        return accuracy
    
    def get_train_accuracy(self, X_train, y_train):
        """Get training accuracy."""
        X_train_flat = X_train.reshape(len(X_train), -1)
        X_train_scaled = self.scaler.transform(X_train_flat)
        y_pred = self.model.predict(X_train_scaled)
        return accuracy_score(y_train, y_pred)


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
    fig.suptitle('Privacy Protection Methods Comparison', fontsize=16, fontweight='bold')
    
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
        print("Building Neural Network model...")
        nn = FacialRecognitionNN(hidden_layers=(512, 256, 128))
        nn.build_model()
        
        # Train model
        nn.train(X_train_protected, y_train)
        
        # Evaluate on training set
        train_accuracy = nn.get_train_accuracy(X_train_protected, y_train)
        print(f"\nTraining Accuracy: {train_accuracy*100:.2f}%")
        
        # Evaluate on test set
        test_accuracy = nn.evaluate(X_test_protected, y_test)
        print(f"Test Accuracy: {test_accuracy*100:.2f}%")
        
        # Store results
        results.append({
            'config': config['name'],
            'method': config['method'],
            'params': config['params'],
            'train_accuracy': train_accuracy,
            'test_accuracy': test_accuracy
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
        f.write("Author: Chloe Partrick\n")
        f.write("=" * 80 + "\n\n")
        
        f.write("EXPERIMENT CONFIGURATION\n")
        f.write("-" * 80 + "\n")
        f.write(f"Dataset: AT&T Face Database (40 subjects, 10 images each)\n")
        f.write(f"Total images: {len(images)}\n")
        f.write(f"Train samples: {len(X_train_clean)}\n")
        f.write(f"Test samples: {len(X_test_clean)}\n")
        f.write(f"Image size: 112 x 92 pixels\n")
        f.write(f"Privacy methods tested: {len(privacy_configs)}\n")
        f.write(f"Model: Multi-Layer Perceptron (MLP) Neural Network\n")
        f.write(f"Architecture: Input -> 512 -> 256 -> 128 -> 40 classes\n\n")
        
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
            f.write(f"  Test Accuracy: {result['test_accuracy']*100:.4f}%\n")
        
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
        f.write("\n" + "=" * 80 + "\n")
        f.write("ANALYSIS\n")
        f.write("-" * 80 + "\n")
        f.write("The results show the trade-off between privacy and accuracy:\n\n")
        f.write("1. Baseline (No Protection): Highest accuracy as expected\n")
        f.write("2. Basic Pixelization: Reduces accuracy due to loss of fine details\n")
        f.write("3. DP-Pix with varying epsilon:\n")
        f.write("   - Lower epsilon (ε=0.1): More privacy but lower accuracy\n")
        f.write("   - Higher epsilon (ε=4): Less privacy but higher accuracy\n\n")
        f.write("The epsilon parameter controls the privacy-utility trade-off in\n")
        f.write("differential privacy. Smaller epsilon values provide stronger privacy\n")
        f.write("guarantees but add more noise, reducing model accuracy.\n")
        
    print(f"\nReport saved to: {report_path}")
    print(f"Sample images saved to: {SAMPLE_DIR}/")
    print("\n" + "=" * 80)
    print("Experiment completed successfully!")
    print("=" * 80)


if __name__ == "__main__":
    main()
