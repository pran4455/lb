"""
Brain Tumor Classification using Trained Captioning Model
========================================================

This script uses the trained DenseNet201 captioning model to classify brain tumor images
and generate captions for the predictions.
"""

import os
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications import DenseNet201
from tensorflow.keras.applications.densenet import preprocess_input
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

class BrainTumorClassifier:
    def __init__(self, model_path="output/densenet_brain_tumor_caption.h5", 
                 tokenizer_path="output/tokenizer.pkl"):
        """
        Initialize the Brain Tumor Classifier
        
        Args:
            model_path: Path to the trained captioning model
            tokenizer_path: Path to the tokenizer file
        """
        self.model_path = model_path
        self.tokenizer_path = tokenizer_path
        
        # Brain tumor classes
        self.class_names = ['glioma', 'meningioma', 'notumor', 'pituitary']
        
        # Initialize DenseNet201 for feature extraction
        self.conv_base = DenseNet201(weights='imagenet', include_top=False, pooling='avg')
        
        # Model components
        self.model = None
        self.tokenizer = None
        self.max_length = None
        
        # Load model and tokenizer
        self.load_model_and_tokenizer()
        
    def load_model_and_tokenizer(self):
        """Load the trained model and tokenizer"""
        try:
            # Load model
            self.model = load_model(self.model_path)
            print(f"✅ Model loaded from {self.model_path}")
            
            # Load tokenizer
            with open(self.tokenizer_path, 'rb') as f:
                self.tokenizer = pickle.load(f)
            print(f"✅ Tokenizer loaded from {self.tokenizer_path}")
            
            # Set max_length (you might need to adjust this based on your training)
            self.max_length = 20  # Default value, adjust if needed
            
        except Exception as e:
            print(f"❌ Error loading model/tokenizer: {e}")
            raise
            
    def extract_features(self, image_path):
        """Extract DenseNet201 features from a single image"""
        try:
            # Load and preprocess image
            image = load_img(image_path, target_size=(224, 224))
            image = img_to_array(image)
            image = np.expand_dims(image, axis=0)
            image = preprocess_input(image)
            
            # Extract features
            features = self.conv_base.predict(image, verbose=0)
            return features
            
        except Exception as e:
            print(f"❌ Error extracting features from {image_path}: {e}")
            return None
            
    def generate_caption(self, photo_feature):
        """Generate caption for a single image"""
        if self.model is None or self.tokenizer is None:
            raise ValueError("Model or tokenizer not loaded")
            
        in_text = 'startseq'
        for _ in range(self.max_length):
            sequence = self.tokenizer.texts_to_sequences([in_text])[0]
            sequence = pad_sequences([sequence], maxlen=self.max_length, padding='post')
            yhat = self.model.predict([photo_feature, sequence], verbose=0)
            yhat_idx = np.argmax(yhat)
            word = self.tokenizer.index_word.get(yhat_idx)
            if word is None:
                break
            in_text += ' ' + word
            if word == 'endseq':
                break
        caption = in_text.replace('startseq', '').replace('endseq', '').strip()
        return caption
        
    def classify_from_caption(self, caption):
        """Classify brain tumor type based on generated caption"""
        caption_lower = caption.lower()
        
        # Check for each class in the caption
        for class_name in self.class_names:
            if class_name in caption_lower:
                return class_name
                
        # If no class found, return unknown
        return 'unknown'
        
    def predict_single_image(self, image_path):
        """Predict class and generate caption for a single image"""
        # Extract features
        features = self.extract_features(image_path)
        if features is None:
            return None, None
            
        # Generate caption
        caption = self.generate_caption(features)
        
        # Classify from caption
        predicted_class = self.classify_from_caption(caption)
        
        return predicted_class, caption
        
    def predict_batch(self, image_paths):
        """Predict classes and captions for a batch of images"""
        predictions = []
        captions = []
        
        for image_path in tqdm(image_paths, desc="Processing images"):
            pred_class, caption = self.predict_single_image(image_path)
            predictions.append(pred_class)
            captions.append(caption)
            
        return predictions, captions
        
    def evaluate_on_dataset(self, dataset_dir, save_results=True):
        """Evaluate the model on a complete dataset"""
        print(f"🔍 Evaluating on dataset: {dataset_dir}")
        
        true_labels = []
        predicted_labels = []
        image_paths = []
        captions = []
        
        # Process each class
        for class_name in self.class_names:
            class_dir = os.path.join(dataset_dir, class_name)
            if not os.path.exists(class_dir):
                print(f"⚠️  Class directory not found: {class_dir}")
                continue
                
            print(f"📁 Processing {class_name} images...")
            
            # Get all images in the class directory
            image_files = [f for f in os.listdir(class_dir) 
                          if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            
            for image_file in tqdm(image_files, desc=f"Processing {class_name}"):
                image_path = os.path.join(class_dir, image_file)
                
                # Predict
                pred_class, caption = self.predict_single_image(image_path)
                
                if pred_class is not None:
                    true_labels.append(class_name)
                    predicted_labels.append(pred_class)
                    image_paths.append(image_path)
                    captions.append(caption)
        
        # Calculate metrics
        accuracy = np.mean([true == pred for true, pred in zip(true_labels, predicted_labels)])
        
        print(f"\n📊 Evaluation Results:")
        print(f"Accuracy: {accuracy:.4f}")
        
        # Classification report
        print("\n📋 Classification Report:")
        print(classification_report(true_labels, predicted_labels, target_names=self.class_names))
        
        # Confusion matrix
        cm = confusion_matrix(true_labels, predicted_labels, labels=self.class_names)
        
        # Plot confusion matrix
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=self.class_names, yticklabels=self.class_names)
        plt.title('Brain Tumor Classification Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.tight_layout()
        
        if save_results:
            plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
            print("📊 Confusion matrix saved as 'confusion_matrix.png'")
        
        plt.show()
        
        # Save detailed results
        if save_results:
            self.save_results(true_labels, predicted_labels, image_paths, captions)
            
        return {
            'accuracy': accuracy,
            'true_labels': true_labels,
            'predicted_labels': predicted_labels,
            'image_paths': image_paths,
            'captions': captions,
            'confusion_matrix': cm
        }
        
    def save_results(self, true_labels, predicted_labels, image_paths, captions):
        """Save detailed results to files"""
        # Save predictions
        with open('classification_results.txt', 'w') as f:
            f.write("Image Path,True Label,Predicted Label,Caption\n")
            for img_path, true_label, pred_label, caption in zip(image_paths, true_labels, predicted_labels, captions):
                f.write(f"{img_path},{true_label},{pred_label},{caption}\n")
        
        print("💾 Detailed results saved to 'classification_results.txt'")
        
        # Save sample predictions
        with open('sample_predictions.txt', 'w') as f:
            f.write("Sample Predictions and Captions\n")
            f.write("=" * 50 + "\n\n")
            
            for i, (img_path, true_label, pred_label, caption) in enumerate(zip(image_paths, true_labels, predicted_labels, captions)):
                if i >= 20:  # Show first 20 samples
                    break
                f.write(f"Image: {os.path.basename(img_path)}\n")
                f.write(f"True Label: {true_label}\n")
                f.write(f"Predicted Label: {pred_label}\n")
                f.write(f"Caption: {caption}\n")
                f.write(f"Correct: {'✅' if true_label == pred_label else '❌'}\n")
                f.write("-" * 30 + "\n\n")
        
        print("💾 Sample predictions saved to 'sample_predictions.txt'")
        
    def interactive_classification(self):
        """Interactive mode for single image classification"""
        print("🧠 Interactive Brain Tumor Classification")
        print("=" * 45)
        
        while True:
            image_path = input("\nEnter image path (or 'quit' to exit): ").strip()
            
            if image_path.lower() == 'quit':
                break
                
            if not os.path.exists(image_path):
                print("❌ Image file not found!")
                continue
                
            print("🔍 Processing image...")
            pred_class, caption = self.predict_single_image(image_path)
            
            if pred_class is not None:
                print(f"\n📊 Results:")
                print(f"Predicted Class: {pred_class}")
                print(f"Generated Caption: {caption}")
            else:
                print("❌ Error processing image")

def main():
    """Main function to run the classification system"""
    print("🧠 Brain Tumor Classification using Captioning Model")
    print("=" * 55)
    
    # Initialize classifier
    try:
        classifier = BrainTumorClassifier()
    except Exception as e:
        print(f"❌ Failed to initialize classifier: {e}")
        return
    
    print("\nChoose an option:")
    print("1. Evaluate on Testing dataset")
    print("2. Interactive single image classification")
    print("3. Evaluate on Training dataset")
    
    choice = input("Enter your choice (1-3): ").strip()
    
    if choice == "1":
        # Evaluate on testing dataset
        results = classifier.evaluate_on_dataset("Testing", save_results=True)
        print(f"\n✅ Evaluation completed with {results['accuracy']:.4f} accuracy")
        
    elif choice == "2":
        # Interactive mode
        classifier.interactive_classification()
        
    elif choice == "3":
        # Evaluate on training dataset
        results = classifier.evaluate_on_dataset("Training", save_results=True)
        print(f"\n✅ Evaluation completed with {results['accuracy']:.4f} accuracy")
        
    else:
        print("❌ Invalid choice. Running evaluation on Testing dataset...")
        results = classifier.evaluate_on_dataset("Testing", save_results=True)
        print(f"\n✅ Evaluation completed with {results['accuracy']:.4f} accuracy")

if __name__ == "__main__":
    main()
