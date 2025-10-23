"""
Brain Tumor Image Captioning using DenseNet201
==============================================

This script implements image captioning for brain tumor classification dataset
using DenseNet201 for feature extraction and LSTM for caption generation.

Dataset classes: glioma, meningioma, notumor, pituitary
"""

import os
import numpy as np
import pickle
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.applications import DenseNet201
from tensorflow.keras.applications.densenet import preprocess_input
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Dense, LSTM, Embedding, Dropout, add
from tensorflow.keras.callbacks import ModelCheckpoint
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')
from sklearn.metrics import accuracy_score
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import nltk

class BrainTumorCaptioning:
    def __init__(self, train_dir="Training", test_dir="Testing", output_dir="output"):
        """
        Initialize the Brain Tumor Captioning system
        
        Args:
            train_dir: Path to training images directory
            test_dir: Path to testing images directory  
            output_dir: Path to save outputs
        """
        self.train_dir = train_dir
        self.test_dir = test_dir
        self.output_dir = output_dir
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Brain tumor classes
        self.class_names = ['glioma', 'meningioma', 'notumor', 'pituitary']
        
        # Medical caption templates for brain tumor images
        self.caption_templates = [
            "A brain MRI scan showing {} tumor characteristics",
            "This medical image displays {} tumor features",
            "Brain scan image with {} tumor indicators",
            "Medical imaging showing {} tumor patterns",
            "Brain MRI with {} tumor visualization",
            "Radiological image displaying {} tumor signs",
            "Brain scan with {} tumor evidence",
            "Medical image showing {} tumor morphology"
        ]
        
        # Initialize DenseNet201 for feature extraction
        self.conv_base = DenseNet201(weights='imagenet', include_top=False, pooling='avg')
        
        # Model components
        self.tokenizer = None
        self.model = None
        self.max_length = None
        self.vocab_size = None
        
    def generate_image_lists(self):
        """Generate text files with image lists for training and testing"""
        print("📝 Generating image lists...")
        
        # Training images
        train_file = os.path.join(self.output_dir, "train_images.txt")
        self._save_image_list(self.train_dir, train_file)
        
        # Testing images  
        test_file = os.path.join(self.output_dir, "test_images.txt")
        self._save_image_list(self.test_dir, test_file)
        
        print("✅ Image lists generated successfully")
        
    def _save_image_list(self, directory, output_file):
        """Save list of all images in directory to text file"""
        image_list = []
        for root, dirs, files in os.walk(directory):
            for file in files:
                if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    image_list.append(file)
        
        with open(output_file, 'w') as f:
            for filename in image_list:
                f.write(filename + "\n")
        
        print(f"📁 Saved {len(image_list)} images to {output_file}")
        
    def generate_captions(self):
        """Generate caption files for training and testing datasets"""
        print("📝 Generating caption files...")
        
        # Training captions
        train_captions_file = os.path.join(self.output_dir, "train_captions.txt")
        self._generate_caption_file(self.train_dir, train_captions_file)
        
        # Testing captions
        test_captions_file = os.path.join(self.output_dir, "test_captions.txt")
        self._generate_caption_file(self.test_dir, test_captions_file)
        
        print("✅ Caption files generated successfully")
        
    def _generate_caption_file(self, directory, output_file):
        """Generate caption file for given directory"""
        with open(output_file, "w") as f_out:
            for class_name in self.class_names:
                class_path = os.path.join(directory, class_name)
                if not os.path.exists(class_path):
                    print(f"⚠️  Folder not found: {class_path}")
                    continue
                    
                for img_file in os.listdir(class_path):
                    if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                        for idx, template in enumerate(self.caption_templates):
                            caption = template.format(class_name)
                            f_out.write(f"{img_file}#{idx} {caption}\n")
        
        print(f"📝 Captions saved to {output_file}")
        
    def extract_features(self, directory):
        """Extract DenseNet201 features from images in directory"""
        print(f"🔍 Extracting features from {directory}...")
        
        features = {}
        for root, dirs, files in os.walk(directory):
            for file in tqdm(files, desc=f"Processing {os.path.basename(directory)}"):
                if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    filepath = os.path.join(root, file)
                    try:
                        # Load and preprocess image
                        image = load_img(filepath, target_size=(224, 224))
                        image = img_to_array(image)
                        image = np.expand_dims(image, axis=0)
                        image = preprocess_input(image)
                        
                        # Extract features using DenseNet201
                        feature = self.conv_base.predict(image, verbose=0)
                        
                        # Save with filename (without extension) as key
                        key = os.path.splitext(file)[0]
                        features[key] = feature
                        
                    except Exception as e:
                        print(f"⚠️  Error processing {filepath}: {e}")
                        continue
        
        print(f"✅ Extracted features for {len(features)} images")
        return features
        
    def save_features(self, train_features, test_features):
        """Save extracted features to pickle files"""
        train_features_file = os.path.join(self.output_dir, "train_features.pkl")
        test_features_file = os.path.join(self.output_dir, "test_features.pkl")
        
        with open(train_features_file, 'wb') as f:
            pickle.dump(train_features, f)
        print(f"💾 Saved train features to {train_features_file}")
        
        with open(test_features_file, 'wb') as f:
            pickle.dump(test_features, f)
        print(f"💾 Saved test features to {test_features_file}")
        
    def load_doc(self, filename):
        """Load document from file"""
        with open(filename, 'r') as file:
            text = file.read()
        return text
        
    def load_set(self, filename):
        """Load set of image identifiers from file"""
        doc = self.load_doc(filename)
        dataset = [line.split('.')[0] for line in doc.split('\n') if len(line) > 0]
        return set(dataset)
        
    def load_clean_descriptions(self, filename, dataset):
        """Load clean descriptions from file"""
        doc = self.load_doc(filename)
        descriptions = dict()
        for line in doc.split('\n'):
            tokens = line.split('#')
            if len(tokens) < 2:
                continue
            image_id = tokens[0].split(".")[0]
            # tokens[1] like "{idx} caption..."; strip the leading index char
            image_desc = tokens[1][1:].strip()
            if image_id in dataset:
                if image_id not in descriptions:
                    descriptions[image_id] = list()
                # Limit to one caption per image to keep dataset compact
                if len(descriptions[image_id]) == 0:
                    desc = 'startseq ' + image_desc + ' endseq'
                    descriptions[image_id].append(desc)
        return descriptions
        
    def load_photo_features(self, filename, dataset):
        """Load photo features from pickle file"""
        all_features = pickle.load(open(filename, 'rb'))
        features = {k: all_features[k] for k in dataset}
        return features
        
    def to_lines(self, descriptions):
        """Convert descriptions dict to list of lines"""
        all_desc = []
        for key in descriptions.keys():
            [all_desc.append(d) for d in descriptions[key]]
        return all_desc
        
    def create_tokenizer(self, descriptions):
        """Create tokenizer from descriptions"""
        lines = self.to_lines(descriptions)
        tokenizer = Tokenizer()
        tokenizer.fit_on_texts(lines)
        return tokenizer
        
    def calculate_max_length(self, descriptions):
        """Calculate maximum length of descriptions"""
        lines = self.to_lines(descriptions)
        return max(len(d.split()) for d in lines)
        
    def create_sequences(self, tokenizer, max_length, descriptions, photos, vocab_size):
        """Create sequences for training"""
        X1, X2, y = list(), list(), list()
        for key, desc_list in descriptions.items():
            for desc in desc_list:
                seq = tokenizer.texts_to_sequences([desc])[0]
                for i in range(1, len(seq)):
                    in_seq, out_seq = seq[:i], seq[i]
                    in_seq = pad_sequences([in_seq], maxlen=max_length, padding="post")[0]
                    X1.append(photos[key][0])  # DenseNet feature vector
                    X2.append(in_seq)
                    y.append(out_seq)
        return np.array(X1), np.array(X2), np.array(y, dtype=np.int32)
        
    def define_model(self, vocab_size, max_length):
        """Define the captioning model architecture"""
        print("🏗️  Building model architecture...")
        
        # Feature extractor
        inputs1 = Input(shape=(1920,))  # DenseNet201 output size
        fe1 = Dropout(0.5)(inputs1)
        fe2 = Dense(256, activation='relu')(fe1)
        
        # Sequence model
        inputs2 = Input(shape=(max_length,))
        se1 = Embedding(vocab_size, 256, mask_zero=True)(inputs2)
        se2 = Dropout(0.5)(se1)
        se3 = LSTM(256)(se2)
        
        # Decoder model
        decoder1 = add([fe2, se3])
        decoder2 = Dense(256, activation='relu')(decoder1)
        outputs = Dense(vocab_size, activation='softmax')(decoder2)
        
        model = Model(inputs=[inputs1, inputs2], outputs=outputs)
        # Use sparse targets for efficiency
        model.compile(loss='sparse_categorical_crossentropy', optimizer='adam')
        
        print("✅ Model architecture created")
        return model
        
    def train_model(self, epochs=10):
        """Train the captioning model"""
        print("🚀 Starting model training...")
        
        # Load datasets
        train_images_file = os.path.join(self.output_dir, "train_images.txt")
        test_images_file = os.path.join(self.output_dir, "test_images.txt")
        train_captions_file = os.path.join(self.output_dir, "train_captions.txt")
        train_features_file = os.path.join(self.output_dir, "train_features.pkl")
        test_features_file = os.path.join(self.output_dir, "test_features.pkl")
        
        # Load training data
        train_set = self.load_set(train_images_file)
        train_descriptions = self.load_clean_descriptions(train_captions_file, train_set)
        train_features = self.load_photo_features(train_features_file, train_set)
        
        # Load test data
        test_set = self.load_set(test_images_file)
        test_descriptions = self.load_clean_descriptions(
            os.path.join(self.output_dir, "test_captions.txt"), test_set)
        test_features = self.load_photo_features(test_features_file, test_set)
        
        # Create tokenizer
        self.tokenizer = self.create_tokenizer(train_descriptions)
        self.vocab_size = len(self.tokenizer.word_index) + 1
        self.max_length = self.calculate_max_length(train_descriptions)
        
        # Save tokenizer
        tokenizer_file = os.path.join(self.output_dir, "tokenizer.pkl")
        pickle.dump(self.tokenizer, open(tokenizer_file, 'wb'))
        print(f"💾 Tokenizer saved to {tokenizer_file}")
        
        # Create sequences
        X1train, X2train, ytrain = self.create_sequences(
            self.tokenizer, self.max_length, train_descriptions, train_features, self.vocab_size)
        X1test, X2test, ytest = self.create_sequences(
            self.tokenizer, self.max_length, test_descriptions, test_features, self.vocab_size)
        
        # Define model
        self.model = self.define_model(self.vocab_size, self.max_length)
        
        # Setup checkpoint
        checkpoint_file = os.path.join(self.output_dir, 
            'brain_tumor_model-ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5')
        checkpoint = ModelCheckpoint(checkpoint_file, monitor='val_loss', 
                                   verbose=1, save_best_only=True, mode='min')
        
        # Train model
        print(f"🏋️  Training for {epochs} epochs...")
        history = self.model.fit([X1train, X2train], ytrain,
                                epochs=epochs,
                                verbose=2,
                                callbacks=[checkpoint],
                                validation_data=([X1test, X2test], ytest))
        
        # Calculate and print accuracy metrics
        self._evaluate_model_accuracy(X1test, X2test, ytest, test_descriptions)
        
        # Save final model
        final_model_file = os.path.join(self.output_dir, "densenet_brain_tumor_caption.h5")
        self.model.save(final_model_file)
        print(f"💾 Final model saved to {final_model_file}")
        
        return history
        
    def _evaluate_model_accuracy(self, X1test, X2test, ytest, test_descriptions):
        """Evaluate model accuracy and BLEU score"""
        print("\n📊 Evaluating model performance...")
        
        # Get predictions
        predictions = self.model.predict([X1test, X2test], verbose=0)
        predicted_classes = np.argmax(predictions, axis=1)
        
        # Calculate accuracy
        accuracy = accuracy_score(ytest, predicted_classes)
        print(f"🎯 Token-level Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        
        # Calculate BLEU score on a sample of captions
        self._calculate_bleu_score(test_descriptions, num_samples=50)
        
    def _calculate_bleu_score(self, test_descriptions, num_samples=50):
        """Calculate BLEU score for generated captions"""
        print(f"🔍 Calculating BLEU score on {num_samples} samples...")
        
        # Get sample images
        sample_images = list(test_descriptions.keys())[:num_samples]
        bleu_scores = []
        
        for image_id in sample_images:
            # Get reference caption (ground truth)
            ref_caption = test_descriptions[image_id][0].replace('startseq ', '').replace(' endseq', '')
            ref_tokens = ref_caption.split()
            
            # Generate caption
            try:
                # Load test features
                test_features_file = os.path.join(self.output_dir, "test_features.pkl")
                test_features = pickle.load(open(test_features_file, 'rb'))
                
                if image_id in test_features:
                    generated_caption = self.generate_caption(test_features[image_id])
                    gen_tokens = generated_caption.split()
                    
                    # Calculate BLEU score
                    smoothie = SmoothingFunction().method4
                    bleu = sentence_bleu([ref_tokens], gen_tokens, smoothing_function=smoothie)
                    bleu_scores.append(bleu)
                    
            except Exception as e:
                print(f"⚠️  Error calculating BLEU for {image_id}: {e}")
                continue
        
        if bleu_scores:
            avg_bleu = np.mean(bleu_scores)
            print(f"📈 Average BLEU Score: {avg_bleu:.4f}")
        else:
            print("⚠️  Could not calculate BLEU score")
        
    def generate_caption(self, photo_feature):
        """Generate caption for a single image"""
        if self.model is None or self.tokenizer is None:
            raise ValueError("Model not trained. Please train the model first.")
            
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
        
    def test_captioning(self, num_samples=10):
        """Test captioning on sample images"""
        print("🧪 Testing captioning on sample images...")
        
        # Load test features
        test_features_file = os.path.join(self.output_dir, "test_features.pkl")
        test_features = pickle.load(open(test_features_file, 'rb'))
        
        # Load tokenizer
        tokenizer_file = os.path.join(self.output_dir, "tokenizer.pkl")
        self.tokenizer = pickle.load(open(tokenizer_file, 'rb'))
        
        # Load model
        model_file = os.path.join(self.output_dir, "densenet_brain_tumor_caption.h5")
        self.model = load_model(model_file)
        
        # Test on sample images
        sample_count = 0
        for image_id, feature in test_features.items():
            if sample_count >= num_samples:
                break
            caption = self.generate_caption(feature)
            print(f"🖼️  {image_id} --> Caption: {caption}")
            sample_count += 1
    
    def check_accuracy(self):
        """Check model accuracy after training"""
        print("📊 Checking model accuracy...")
        
        # Load test data
        test_images_file = os.path.join(self.output_dir, "test_images.txt")
        test_captions_file = os.path.join(self.output_dir, "test_captions.txt")
        test_features_file = os.path.join(self.output_dir, "test_features.pkl")
        
        test_set = self.load_set(test_images_file)
        test_descriptions = self.load_clean_descriptions(test_captions_file, test_set)
        test_features = self.load_photo_features(test_features_file, test_set)
        
        # Load tokenizer and model if not already loaded
        if self.tokenizer is None:
            tokenizer_file = os.path.join(self.output_dir, "tokenizer.pkl")
            self.tokenizer = pickle.load(open(tokenizer_file, 'rb'))
            self.vocab_size = len(self.tokenizer.word_index) + 1
            self.max_length = self.calculate_max_length(test_descriptions)
        
        if self.model is None:
            model_file = os.path.join(self.output_dir, "densenet_brain_tumor_caption.h5")
            self.model = load_model(model_file)
        
        # Create test sequences
        X1test, X2test, ytest = self.create_sequences(
            self.tokenizer, self.max_length, test_descriptions, test_features, self.vocab_size)
        
        # Evaluate accuracy
        self._evaluate_model_accuracy(X1test, X2test, ytest, test_descriptions)
            
    def run_full_pipeline(self, epochs=10):
        # """Run the complete captioning pipeline"""
        # print("🚀 Starting Brain Tumor Image Captioning Pipeline")
        # print("=" * 50)
        
        # # Step 1: Generate image lists
        # self.generate_image_lists()
        
        # # Step 2: Generate captions
        # self.generate_captions()
        
        # # Step 3: Extract features
        # print("\n🔍 Extracting features...")
        # train_features = self.extract_features(self.train_dir)
        # test_features = self.extract_features(self.test_dir)
        
        # # Step 4: Save features
        # self.save_features(train_features, test_features)
        
        # Step 5: Train model
        print("\n🏋️  Training model...")
        history = self.train_model(epochs)
        
        # Step 6: Test captioning
        print("\n🧪 Testing captioning...")
        self.test_captioning()
        
        print("\n✅ Pipeline completed successfully!")
        return history

def main():
    """Main function to run the brain tumor captioning system"""
    # Initialize the captioning system
    captioning_system = BrainTumorCaptioning(
        train_dir="Training",
        test_dir="Testing", 
        output_dir="output"
    )
    
    # Run the complete pipeline
    history = captioning_system.run_full_pipeline(epochs=2)
    
    print("\n🎉 Brain Tumor Image Captioning completed!")
    print("📁 Check the 'output' directory for all generated files:")
    print("   - train_images.txt, test_images.txt")
    print("   - train_captions.txt, test_captions.txt") 
    print("   - train_features.pkl, test_features.pkl")
    print("   - tokenizer.pkl")
    print("   - densenet_brain_tumor_caption.h5")

if __name__ == "__main__":
    main()
