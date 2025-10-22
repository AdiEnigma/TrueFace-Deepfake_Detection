#!/usr/bin/env python3
"""
Model Comparison and Evaluation Tool
Compare different versions of your MesoNet models
"""

import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import time
from datetime import datetime
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ModelComparator:
    def __init__(self, dataset_path="Dataset", models_path="models"):
        self.dataset_path = Path(dataset_path)
        self.models_path = Path(models_path)
        
        # Test data generator
        self.test_datagen = ImageDataGenerator(rescale=1./255)
        
        logger.info("🔍 Model Comparator initialized")

    def load_models(self):
        """Load all available models for comparison"""
        
        model_files = list(self.models_path.glob("*.h5"))
        models = {}
        
        for model_file in model_files:
            try:
                model_name = model_file.stem
                logger.info(f"📥 Loading model: {model_name}")
                
                model = keras.models.load_model(str(model_file))
                models[model_name] = {
                    'model': model,
                    'path': model_file,
                    'params': model.count_params()
                }
                
                logger.info(f"✅ Loaded {model_name} ({model.count_params():,} parameters)")
                
            except Exception as e:
                logger.warning(f"⚠️ Could not load {model_file}: {e}")
        
        return models

    def prepare_test_data(self, image_size=(256, 256)):
        """Prepare test data generator"""
        
        test_generator = self.test_datagen.flow_from_directory(
            self.dataset_path / "Test",
            target_size=image_size,
            batch_size=32,
            class_mode='binary',
            classes=['Real', 'Fake'],
            shuffle=False,
            seed=42
        )
        
        logger.info(f"🧪 Test data prepared: {test_generator.samples} samples")
        return test_generator

    def evaluate_model(self, model, model_name, test_generator):
        """Evaluate a single model"""
        
        logger.info(f"🔍 Evaluating {model_name}...")
        
        # Reset generator
        test_generator.reset()
        
        # Time the evaluation
        start_time = time.time()
        
        # Get predictions
        predictions = model.predict(test_generator, verbose=0)
        predicted_classes = (predictions > 0.5).astype(int).flatten()
        
        evaluation_time = time.time() - start_time
        
        # Get true labels
        true_classes = test_generator.classes
        
        # Calculate metrics
        test_loss, test_accuracy = model.evaluate(test_generator, verbose=0)
        
        # Classification report
        report = classification_report(true_classes, predicted_classes, 
                                     target_names=['Real', 'Fake'], output_dict=True)
        
        # ROC curve
        fpr, tpr, _ = roc_curve(true_classes, predictions)
        roc_auc = auc(fpr, tpr)
        
        # Confusion matrix
        cm = confusion_matrix(true_classes, predicted_classes)
        
        results = {
            'model_name': model_name,
            'test_loss': float(test_loss),
            'test_accuracy': float(test_accuracy),
            'precision_real': float(report['Real']['precision']),
            'recall_real': float(report['Real']['recall']),
            'f1_real': float(report['Real']['f1-score']),
            'precision_fake': float(report['Fake']['precision']),
            'recall_fake': float(report['Fake']['recall']),
            'f1_fake': float(report['Fake']['f1-score']),
            'roc_auc': float(roc_auc),
            'confusion_matrix': cm.tolist(),
            'evaluation_time': float(evaluation_time),
            'predictions': predictions.flatten().tolist(),
            'fpr': fpr.tolist(),
            'tpr': tpr.tolist()
        }
        
        logger.info(f"✅ {model_name} - Accuracy: {test_accuracy:.4f}, AUC: {roc_auc:.4f}")
        
        return results

    def compare_models(self):
        """Compare all available models"""
        
        logger.info("🏁 Starting model comparison...")
        
        # Load models
        models = self.load_models()
        
        if not models:
            logger.error("❌ No models found to compare!")
            return None
        
        # Prepare test data (use largest image size among models)
        test_generator = self.prepare_test_data()
        
        # Evaluate each model
        results = {}
        for model_name, model_info in models.items():
            try:
                result = self.evaluate_model(model_info['model'], model_name, test_generator)
                result['parameters'] = model_info['params']
                results[model_name] = result
            except Exception as e:
                logger.error(f"❌ Error evaluating {model_name}: {e}")
        
        # Generate comparison report
        self.generate_comparison_report(results)
        
        # Create visualizations
        self.create_comparison_plots(results)
        
        return results

    def generate_comparison_report(self, results):
        """Generate a detailed comparison report"""
        
        logger.info("📊 Generating comparison report...")
        
        # Create comparison table
        comparison_data = []
        for model_name, result in results.items():
            comparison_data.append({
                'Model': model_name,
                'Parameters': f"{result['parameters']:,}",
                'Accuracy': f"{result['test_accuracy']:.4f}",
                'Precision (Real)': f"{result['precision_real']:.4f}",
                'Recall (Real)': f"{result['recall_real']:.4f}",
                'Precision (Fake)': f"{result['precision_fake']:.4f}",
                'Recall (Fake)': f"{result['recall_fake']:.4f}",
                'ROC AUC': f"{result['roc_auc']:.4f}",
                'Eval Time (s)': f"{result['evaluation_time']:.2f}"
            })
        
        # Sort by accuracy
        comparison_data.sort(key=lambda x: float(x['Accuracy']), reverse=True)
        
        # Print comparison table
        print("\n" + "="*120)
        print("🏆 MODEL COMPARISON RESULTS")
        print("="*120)
        
        # Print header
        headers = list(comparison_data[0].keys())
        header_line = " | ".join(f"{h:>15}" for h in headers)
        print(header_line)
        print("-" * len(header_line))
        
        # Print data
        for row in comparison_data:
            data_line = " | ".join(f"{row[h]:>15}" for h in headers)
            print(data_line)
        
        print("="*120)
        
        # Find best model
        best_model = comparison_data[0]
        print(f"\n🥇 BEST MODEL: {best_model['Model']}")
        print(f"   Accuracy: {best_model['Accuracy']}")
        print(f"   ROC AUC: {best_model['ROC AUC']}")
        print(f"   Parameters: {best_model['Parameters']}")
        
        # Save detailed results
        with open(self.models_path / 'model_comparison.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info("💾 Comparison report saved")

    def create_comparison_plots(self, results):
        """Create comparison visualizations"""
        
        logger.info("📈 Creating comparison plots...")
        
        # Set up the plot
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        model_names = list(results.keys())
        
        # 1. Accuracy comparison
        accuracies = [results[name]['test_accuracy'] for name in model_names]
        axes[0, 0].bar(model_names, accuracies, color='skyblue')
        axes[0, 0].set_title('Model Accuracy Comparison')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].set_ylim(0, 1)
        for i, v in enumerate(accuracies):
            axes[0, 0].text(i, v + 0.01, f'{v:.3f}', ha='center')
        
        # 2. ROC curves
        for name in model_names:
            fpr = results[name]['fpr']
            tpr = results[name]['tpr']
            auc_score = results[name]['roc_auc']
            axes[0, 1].plot(fpr, tpr, label=f'{name} (AUC = {auc_score:.3f})')
        
        axes[0, 1].plot([0, 1], [0, 1], 'k--', label='Random')
        axes[0, 1].set_xlabel('False Positive Rate')
        axes[0, 1].set_ylabel('True Positive Rate')
        axes[0, 1].set_title('ROC Curves Comparison')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # 3. Precision-Recall comparison
        precisions_real = [results[name]['precision_real'] for name in model_names]
        recalls_real = [results[name]['recall_real'] for name in model_names]
        precisions_fake = [results[name]['precision_fake'] for name in model_names]
        recalls_fake = [results[name]['recall_fake'] for name in model_names]
        
        x = np.arange(len(model_names))
        width = 0.35
        
        axes[1, 0].bar(x - width/2, precisions_real, width, label='Real Precision', alpha=0.8)
        axes[1, 0].bar(x + width/2, precisions_fake, width, label='Fake Precision', alpha=0.8)
        axes[1, 0].set_xlabel('Models')
        axes[1, 0].set_ylabel('Precision')
        axes[1, 0].set_title('Precision Comparison')
        axes[1, 0].set_xticks(x)
        axes[1, 0].set_xticklabels(model_names, rotation=45)
        axes[1, 0].legend()
        
        # 4. Parameters vs Accuracy
        parameters = [results[name]['parameters'] for name in model_names]
        axes[1, 1].scatter(parameters, accuracies, s=100, alpha=0.7)
        for i, name in enumerate(model_names):
            axes[1, 1].annotate(name, (parameters[i], accuracies[i]), 
                              xytext=(5, 5), textcoords='offset points')
        axes[1, 1].set_xlabel('Number of Parameters')
        axes[1, 1].set_ylabel('Accuracy')
        axes[1, 1].set_title('Model Complexity vs Performance')
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.savefig(self.models_path / 'model_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Create individual confusion matrices
        fig, axes = plt.subplots(1, len(model_names), figsize=(5*len(model_names), 4))
        if len(model_names) == 1:
            axes = [axes]
        
        for i, name in enumerate(model_names):
            cm = np.array(results[name]['confusion_matrix'])
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                       xticklabels=['Real', 'Fake'], yticklabels=['Real', 'Fake'],
                       ax=axes[i])
            axes[i].set_title(f'{name}\nAccuracy: {results[name]["test_accuracy"]:.3f}')
        
        plt.tight_layout()
        plt.savefig(self.models_path / 'confusion_matrices_comparison.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info("📊 Comparison plots saved")

def main():
    """Main comparison function"""
    
    print("""
    🔍 MesoNet Model Comparison Tool
    ===============================
    
    This tool will:
    ✅ Load all .h5 models from the models/ directory
    ✅ Evaluate each model on the test dataset
    ✅ Generate detailed comparison metrics
    ✅ Create visualization plots
    ✅ Identify the best performing model
    
    """)
    
    try:
        comparator = ModelComparator()
        results = comparator.compare_models()
        
        if results:
            print("""
            🎉 Model comparison completed!
            
            📁 Generated files:
            - models/model_comparison.json (Detailed results)
            - models/model_comparison.png (Performance plots)
            - models/confusion_matrices_comparison.png (Confusion matrices)
            
            📊 Check the console output above for the comparison table.
            """)
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Comparison failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    
    if success:
        print("\n✅ Model comparison completed!")
    else:
        print("\n❌ Model comparison failed.")
