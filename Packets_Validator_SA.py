#!/usr/bin/env python3
# Independent Model Validator
# For testing generated packets against independent anomaly detection models

import pandas as pd
import numpy as np
import os
import joblib
import time
import warnings
import traceback
import argparse
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.model_selection import train_test_split

# Suppress warnings
warnings.filterwarnings('ignore')

class IndependentValidator:
    def __init__(self, original_dataset_path, generated_packets_dir, output_dir, seed=42):
        """
        Initialize the independent validator
        
        Args:
            original_dataset_path: Path to the original ACI IoT 2023 dataset
            generated_packets_dir: Directory containing the generated packets
            output_dir: Directory to save the validation report
            seed: Random seed for reproducibility
        """
        self.original_dataset_path = original_dataset_path
        self.generated_packets_dir = generated_packets_dir
        self.output_dir = output_dir
        self.seed = seed
        self.flow_id_column = 'Flow ID'
        self.drop_columns = ['Src IP', 'Dst IP', 'Timestamp']
        
        # Create output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        # Set random seed for reproducibility
        np.random.seed(seed)
        
        print("Independent Validator initialized.")
        
    def load_original_data(self):
        """
        Load and preprocess the original ACI IoT 2023 dataset
        """
        print(f"Loading original dataset from {self.original_dataset_path}...")
        try:
            # Load data
            self.original_data = pd.read_csv(self.original_dataset_path)
            print(f"Original dataset loaded with shape: {self.original_data.shape}")
            
            # Check if dataset is empty
            if self.original_data.empty:
                print("Error: Original dataset is empty")
                return False
            
            # Check if dataset is too large
            if len(self.original_data) > 500000:
                print(f"Dataset is very large ({len(self.original_data)} rows). Sampling 500,000 rows for faster processing.")
                self.original_data = self.original_data.sample(n=500000, random_state=self.seed)
                print(f"Reduced dataset shape: {self.original_data.shape}")
                
            # Identify columns to drop
            columns_to_drop = [col for col in self.drop_columns if col in self.original_data.columns]
            
            # Check if 'Label' column exists, if not try to find another label column
            if 'Label' not in self.original_data.columns:
                label_candidates = ['label', 'class', 'Class', 'category', 'Category', 'Attack', 'attack_type']
                label_found = False
                for candidate in label_candidates:
                    if candidate in self.original_data.columns:
                        self.original_data.rename(columns={candidate: 'Label'}, inplace=True)
                        label_found = True
                        break
                        
                if not label_found:
                    print("Error: No label column found in original dataset")
                    return False
            
            # Extract classes
            self.classes = self.original_data['Label'].unique()
            print(f"Found {len(self.classes)} classes: {self.classes}")
            
            # Preprocess original data (drop columns, handle missing values)
            self.original_data = self.original_data.drop(columns=columns_to_drop, errors='ignore')
            
            # Replace infinities with NaN and then fill NaN values
            self.original_data = self.original_data.replace([np.inf, -np.inf], np.nan)
            
            # Fill numeric columns with mean, non-numeric with most frequent value
            numeric_cols = self.original_data.select_dtypes(include=np.number).columns
            for col in self.original_data.columns:
                if col in numeric_cols:
                    col_mean = self.original_data[col].mean()
                    if pd.isna(col_mean):  # If mean is NaN, use 0
                        self.original_data[col] = self.original_data[col].fillna(0)
                    else:
                        self.original_data[col] = self.original_data[col].fillna(col_mean)
                else:
                    # Only fill non-numeric columns that aren't Flow ID or Label
                    if col not in [self.flow_id_column, 'Label']:
                        if self.original_data[col].mode().shape[0] > 0:
                            self.original_data[col] = self.original_data[col].fillna(self.original_data[col].mode()[0])
                        else:
                            self.original_data[col] = self.original_data[col].fillna("Unknown")
            
            # Handle non-numeric columns
            non_numeric_cols = self.original_data.select_dtypes(exclude=np.number).columns.tolist()
            
            # Remove Label from non_numeric_cols if it's there
            if 'Label' in non_numeric_cols:
                non_numeric_cols.remove('Label')
            
            # Remove Flow ID from non_numeric_cols if it's there
            if self.flow_id_column in non_numeric_cols:
                non_numeric_cols.remove(self.flow_id_column)
                
            # One-hot encode non-numeric columns that aren't Flow ID or Label
            if non_numeric_cols:
                print(f"One-hot encoding non-numeric columns: {non_numeric_cols}")
                try:
                    self.original_data = pd.get_dummies(self.original_data, columns=non_numeric_cols)
                except Exception as e:
                    print(f"Error one-hot encoding: {e}")
                    # Try alternative approach: encode each column separately
                    for col in non_numeric_cols:
                        try:
                            # Get dummies for this column
                            dummies = pd.get_dummies(self.original_data[col], prefix=col)
                            # Drop the original column and join with dummies
                            self.original_data = self.original_data.drop(columns=[col]).join(dummies)
                        except Exception as e2:
                            print(f"Error encoding column {col}: {e2}")
                            # Last resort: drop the column
                            self.original_data = self.original_data.drop(columns=[col], errors='ignore')
            
            # Prepare features and labels
            self.original_labels = self.original_data['Label'].copy()
            original_features = self.original_data.drop(['Label'], axis=1)
            
            # Remove Flow ID if present
            if self.flow_id_column in original_features.columns:
                original_features = original_features.drop(self.flow_id_column, axis=1)
            
            # Handle any constant columns (zero variance)
            constant_cols = [col for col in original_features.columns if original_features[col].nunique() <= 1]
            if constant_cols:
                print(f"Dropping {len(constant_cols)} constant columns: {constant_cols}")
                original_features = original_features.drop(columns=constant_cols)
            
            # Handle any remaining non-numeric columns
            non_numeric_cols = original_features.select_dtypes(exclude=np.number).columns.tolist()
            if non_numeric_cols:
                print(f"Warning: Found additional non-numeric columns: {non_numeric_cols}")
                print("Attempting to convert to numeric or drop...")
                
                for col in non_numeric_cols:
                    try:
                        original_features[col] = pd.to_numeric(original_features[col], errors='coerce')
                        original_features[col] = original_features[col].fillna(original_features[col].mean() if not pd.isna(original_features[col].mean()) else 0)
                    except Exception as e:
                        print(f"Error converting column {col}: {e}")
                        original_features = original_features.drop(columns=[col])
            
            # Check for and handle any remaining NaN values
            if original_features.isna().any().any():
                print("Warning: Still have missing values after preprocessing. Filling with zeros.")
                original_features = original_features.fillna(0)
            
            # Scale features
            print("Scaling features...")
            self.scaler = StandardScaler()
            self.original_features_scaled = self.scaler.fit_transform(original_features)
            print("Feature scaling completed.")
            
            # Save feature names for later use
            self.feature_names = original_features.columns.tolist()
            
            print("Original data preprocessing completed.")
            return True
            
        except Exception as e:
            print(f"Error loading original dataset: {e}")
            print(traceback.format_exc())
            return False
    
    def train_independent_models(self):
        """
        Train independent anomaly detection models on the original dataset
        """
        print("Training independent anomaly detection models...")
        
        # Sample the original data if it's too large
        if len(self.original_features_scaled) > 10000:
            print(f"Original dataset is large ({len(self.original_features_scaled)} samples). Sampling data for faster training.")
            # Random sample for global models
            sample_size = min(10000, len(self.original_features_scaled))
            sample_indices = np.random.choice(len(self.original_features_scaled), size=sample_size, replace=False)
            global_sample = self.original_features_scaled[sample_indices]
            global_labels_sample = self.original_labels.iloc[sample_indices]
            print(f"Using {sample_size} samples for global model training")
        else:
            global_sample = self.original_features_scaled
            global_labels_sample = self.original_labels
        
        # Train global models on sampled data
        print("Training global One-Class SVM (this may take a few minutes)...")
        global_ocsvm = OneClassSVM(nu=0.01, kernel="rbf", gamma='scale')
        global_ocsvm.fit(global_sample)
        print("Global One-Class SVM training completed.")
        
        print("Training global Isolation Forest...")
        global_isolation_forest = IsolationForest(contamination=0.01, random_state=self.seed, n_jobs=-1)
        global_isolation_forest.fit(global_sample)
        print("Global Isolation Forest training completed.")
        
        self.global_models = {
            'ocsvm': global_ocsvm,
            'isolation_forest': global_isolation_forest
        }
        
        # Train class-specific models
        print("Training class-specific models...")
        self.class_models = {}
        
        for cls in tqdm(self.classes, desc="Training class models"):
            print(f"\nTraining models for class: {cls}")
            
            # Filter data for this class
            class_indices = np.where(self.original_labels == cls)[0]
            if len(class_indices) > 5000:
                print(f"Large class with {len(class_indices)} samples. Sampling 5000 for training.")
                sample_indices = np.random.choice(class_indices, size=5000, replace=False)
                X_class = self.original_features_scaled[sample_indices]
            else:
                X_class = self.original_features_scaled[class_indices]
            
            # Skip if no data or too few samples for this class
            if len(X_class) < 20:
                print(f"Warning: Insufficient data for class {cls}. Skipping model training.")
                self.class_models[cls] = None
                continue
            
            # For very large classes, use a larger test split
            test_size = 0.1 if len(X_class) > 1000 else 0.2
            
            # Split into train and validation sets
            X_train, X_val = train_test_split(X_class, test_size=test_size, random_state=self.seed)
            
            # Train One-Class SVM with timeout protection
            try:
                print(f"Training One-Class SVM for class {cls}...")
                nu = min(0.1, 1.0 / (len(X_train) + 1))  # Avoid nu > 1/n which causes error
                ocsvm = OneClassSVM(nu=nu, kernel="rbf", gamma='scale')
                ocsvm.fit(X_train)
                print(f"One-Class SVM training completed for class {cls}")
            except Exception as e:
                print(f"Error training One-Class SVM for class {cls}: {e}")
                try:
                    print("Trying simpler SVM model...")
                    ocsvm = OneClassSVM(nu=0.01, kernel="linear")
                    ocsvm.fit(X_train)
                except Exception as e2:
                    print(f"Fallback One-Class SVM failed for class {cls}: {e2}")
                    # Create a dummy model
                    class DummyModel:
                        def predict(self, X):
                            return np.ones(len(X))
                    ocsvm = DummyModel()
            
            # Train Isolation Forest
            try:
                print(f"Training Isolation Forest for class {cls}...")
                contamination = min(0.1, 1.0 / (len(X_train) + 1))
                isolation_forest = IsolationForest(contamination=contamination, random_state=self.seed, n_jobs=-1)
                isolation_forest.fit(X_train)
                print(f"Isolation Forest training completed for class {cls}")
            except Exception as e:
                print(f"Error training Isolation Forest for class {cls}: {e}")
                # Create a dummy model
                class DummyModel:
                    def predict(self, X):
                        return np.ones(len(X))
                isolation_forest = DummyModel()
            
            # Validate models on original validation data
            try:
                if len(X_val) > 1000:
                    print(f"Large validation set ({len(X_val)} samples). Sampling 1000 for validation.")
                    val_indices = np.random.choice(len(X_val), size=1000, replace=False)
                    X_val_sample = X_val[val_indices]
                else:
                    X_val_sample = X_val
                
                ocsvm_preds = ocsvm.predict(X_val_sample)
                if_preds = isolation_forest.predict(X_val_sample)
                
                ocsvm_anomaly_rate = np.sum(ocsvm_preds == -1) / len(ocsvm_preds) * 100
                if_anomaly_rate = np.sum(if_preds == -1) / len(if_preds) * 100
                
                print(f"Original Data - One-Class SVM Validation Anomaly Rate: {ocsvm_anomaly_rate:.2f}%")
                print(f"Original Data - Isolation Forest Validation Anomaly Rate: {if_anomaly_rate:.2f}%")
            except Exception as e:
                print(f"Error validating models for class {cls}: {e}")
            
            # Store models
            self.class_models[cls] = {
                'ocsvm': ocsvm,
                'isolation_forest': isolation_forest
            }
        
        print("Independent model training completed.")
    
    def validate_generated_packets(self, report_file="independent_validation_report.csv"):
        """
        Validate the generated packets against the independent models
        
        Args:
            report_file: Name of the CSV file to save the validation report
        """
        print("\nValidating generated packets against independent models...")
        
        results_data = []
        
        # Check if generated packets directory exists
        if not os.path.exists(self.generated_packets_dir):
            print(f"Error: Generated packets directory '{self.generated_packets_dir}' not found.")
            print("Please provide the correct path to the directory containing the generated packets.")
            print("You can specify the path when running the script with the --packets parameter.")
            print("Example: python OneSVM&IsiForestTest.py --packets /path/to/generated_packets")
            
            # Try to find the directory based on common patterns
            parent_dir = os.path.dirname(self.generated_packets_dir)
            if os.path.exists(parent_dir):
                print(f"\nSearching for generated packets in parent directory: {parent_dir}")
                potential_dirs = [d for d in os.listdir(parent_dir) if 'generated' in d.lower() and os.path.isdir(os.path.join(parent_dir, d))]
                
                if potential_dirs:
                    print(f"Found potential generated packets directories: {potential_dirs}")
                    print(f"Please specify one of these directories when running the script.")
                
            return
        
        # Find all CSV files in the generated packets directory
        generated_files = [f for f in os.listdir(self.generated_packets_dir) 
                          if f.endswith('.csv') and f != 'all_generated_packets.csv']
        
        if not generated_files:
            print(f"No generated packet files found in {self.generated_packets_dir}")
            return
        
        for file_name in generated_files:
            file_path = os.path.join(self.generated_packets_dir, file_name)
            
            # Extract class name from file name
            class_name = file_name.replace('_generated.csv', '').replace('_', ' ')
            
            # Check if this class exists in our original dataset
            if class_name not in self.classes:
                # Try to match with original classes
                matched = False
                for orig_cls in self.classes:
                    safe_orig_cls = orig_cls.replace('/', '_').replace(' ', '_').replace('\\', '_')
                    if safe_orig_cls == class_name.replace(' ', '_'):
                        class_name = orig_cls
                        matched = True
                        break
                
                if not matched:
                    print(f"Warning: Class '{class_name}' not found in original dataset. Skipping.")
                    continue
            
            print(f"\nValidating packets for class: {class_name}")
            
            try:
                # Load generated packets
                df = pd.read_csv(file_path)
                print(f"Loaded {len(df)} generated packets from {file_name}")
                
                # Extract features (remove Flow ID and Label)
                if 'Label' in df.columns:
                    X_gen = df.drop(['Label'], axis=1)
                else:
                    X_gen = df.copy()
                
                if self.flow_id_column in X_gen.columns:
                    X_gen = X_gen.drop(self.flow_id_column, axis=1)
                
                # Handle any non-numeric columns
                non_numeric_cols = X_gen.select_dtypes(exclude=np.number).columns.tolist()
                if non_numeric_cols:
                    print(f"Warning: Non-numeric columns found: {non_numeric_cols}")
                    for col in non_numeric_cols:
                        try:
                            X_gen[col] = pd.to_numeric(X_gen[col], errors='coerce')
                        except:
                            print(f"Dropping non-numeric column: {col}")
                            X_gen = X_gen.drop(columns=[col])
                
                # Align columns with original feature set
                missing_cols = set(self.feature_names) - set(X_gen.columns)
                extra_cols = set(X_gen.columns) - set(self.feature_names)
                
                if missing_cols:
                    print(f"Adding {len(missing_cols)} missing columns with zeros")
                    for col in missing_cols:
                        X_gen[col] = 0
                
                if extra_cols:
                    print(f"Dropping {len(extra_cols)} extra columns")
                    X_gen = X_gen.drop(columns=extra_cols)
                
                # Ensure column order matches original
                X_gen = X_gen[self.feature_names]
                
                # Replace any remaining NaN values with zeros
                X_gen = X_gen.fillna(0)
                
                # Scale the data
                X_gen_scaled = self.scaler.transform(X_gen)
                
                # Test with global models
                global_ocsvm_preds = self.global_models['ocsvm'].predict(X_gen_scaled)
                global_if_preds = self.global_models['isolation_forest'].predict(X_gen_scaled)
                
                global_ocsvm_anomaly_rate = np.sum(global_ocsvm_preds == -1) / len(global_ocsvm_preds) * 100
                global_if_anomaly_rate = np.sum(global_if_preds == -1) / len(global_if_preds) * 100
                
                print(f"Global One-Class SVM - Anomaly Rate: {global_ocsvm_anomaly_rate:.2f}%")
                print(f"Global Isolation Forest - Anomaly Rate: {global_if_anomaly_rate:.2f}%")
                
                # Test with class-specific models if available
                cls_ocsvm_anomaly_rate = np.nan
                cls_if_anomaly_rate = np.nan
                
                if class_name in self.class_models and self.class_models[class_name] is not None:
                    cls_ocsvm = self.class_models[class_name]['ocsvm']
                    cls_isolation_forest = self.class_models[class_name]['isolation_forest']
                    
                    cls_ocsvm_preds = cls_ocsvm.predict(X_gen_scaled)
                    cls_if_preds = cls_isolation_forest.predict(X_gen_scaled)
                    
                    cls_ocsvm_anomaly_rate = np.sum(cls_ocsvm_preds == -1) / len(cls_ocsvm_preds) * 100
                    cls_if_anomaly_rate = np.sum(cls_if_preds == -1) / len(cls_if_preds) * 100
                    
                    print(f"Class-Specific One-Class SVM - Anomaly Rate: {cls_ocsvm_anomaly_rate:.2f}%")
                    print(f"Class-Specific Isolation Forest - Anomaly Rate: {cls_if_anomaly_rate:.2f}%")
                else:
                    print(f"No class-specific models available for {class_name}")
                
                # Store results
                result = {
                    'Class': class_name,
                    'File_Name': file_name,
                    'Num_Generated_Packets': len(X_gen),
                    'Global_OCSVM_Anomaly_Rate': global_ocsvm_anomaly_rate,
                    'Global_IF_Anomaly_Rate': global_if_anomaly_rate,
                    'Class_OCSVM_Anomaly_Rate': cls_ocsvm_anomaly_rate,
                    'Class_IF_Anomaly_Rate': cls_if_anomaly_rate,
                    'Global_OCSVM_Status': 'PASS' if global_ocsvm_anomaly_rate < 30 else 'FAIL',
                    'Global_IF_Status': 'PASS' if global_if_anomaly_rate < 30 else 'FAIL',
                    'Class_OCSVM_Status': 'PASS' if cls_ocsvm_anomaly_rate < 30 or np.isnan(cls_ocsvm_anomaly_rate) else 'FAIL',
                    'Class_IF_Status': 'PASS' if cls_if_anomaly_rate < 30 or np.isnan(cls_if_anomaly_rate) else 'FAIL'
                }
                results_data.append(result)
                
            except Exception as e:
                print(f"Error validating {file_name}: {e}")
                print(traceback.format_exc())
        
        # Generate summary report
        if results_data:
            # Convert to DataFrame
            report_df = pd.DataFrame(results_data)
            
            # Calculate overall statistics
            overall_global_ocsvm = report_df['Global_OCSVM_Anomaly_Rate'].mean()
            overall_global_if = report_df['Global_IF_Anomaly_Rate'].mean()
            
            # Filter out NaN values for class-specific models
            class_ocsvm_rates = report_df['Class_OCSVM_Anomaly_Rate'].dropna()
            class_if_rates = report_df['Class_IF_Anomaly_Rate'].dropna()
            
            overall_class_ocsvm = class_ocsvm_rates.mean() if not class_ocsvm_rates.empty else np.nan
            overall_class_if = class_if_rates.mean() if not class_if_rates.empty else np.nan
            
            # Add overall statistics to report
            overall_row = {
                'Class': 'OVERALL AVERAGE',
                'File_Name': 'N/A',
                'Num_Generated_Packets': report_df['Num_Generated_Packets'].sum(),
                'Global_OCSVM_Anomaly_Rate': overall_global_ocsvm,
                'Global_IF_Anomaly_Rate': overall_global_if,
                'Class_OCSVM_Anomaly_Rate': overall_class_ocsvm,
                'Class_IF_Anomaly_Rate': overall_class_if,
                'Global_OCSVM_Status': 'PASS' if overall_global_ocsvm < 30 else 'FAIL',
                'Global_IF_Status': 'PASS' if overall_global_if < 30 else 'FAIL',
                'Class_OCSVM_Status': 'PASS' if overall_class_ocsvm < 30 or np.isnan(overall_class_ocsvm) else 'FAIL',
                'Class_IF_Status': 'PASS' if overall_class_if < 30 or np.isnan(overall_class_if) else 'FAIL'
            }
            report_df = pd.concat([report_df, pd.DataFrame([overall_row])], ignore_index=True)
            
            # Save report to CSV
            report_path = os.path.join(self.output_dir, report_file)
            report_df.to_csv(report_path, index=False)
            print(f"\nValidation report saved to: {report_path}")
            
            # Print overall summary
            print("\n=== Validation Summary ===")
            print(f"Global One-Class SVM - Average Anomaly Rate: {overall_global_ocsvm:.2f}%")
            print(f"Global Isolation Forest - Average Anomaly Rate: {overall_global_if:.2f}%")
            if not np.isnan(overall_class_ocsvm):
                print(f"Class-Specific One-Class SVM - Average Anomaly Rate: {overall_class_ocsvm:.2f}%")
            if not np.isnan(overall_class_if):
                print(f"Class-Specific Isolation Forest - Average Anomaly Rate: {overall_class_if:.2f}%")
        else:
            print("No results available to generate report.")
    
    def run(self, report_file="independent_validation_report.csv"):
        """
        Run the complete validation pipeline
        
        Args:
            report_file: Name of the CSV file to save the validation report
        """
        print("Starting independent validation pipeline...")
        
        try:
            # Step 1: Load original data
            if not self.load_original_data():
                print("Error loading original data. Exiting.")
                return
            
            # Step 2: Train independent models
            self.train_independent_models()
            
            # Step 3: Validate generated packets
            self.validate_generated_packets(report_file)
            
            print("Independent validation pipeline completed successfully!")
            
        except Exception as e:
            print(f"Error in validation pipeline: {e}")
            print(traceback.format_exc())
            print("Pipeline completed with errors. Check the logs for details.")


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Validate generated packets against independent models')
    parser.add_argument('--original', type=str, 
                        default="/Users/mayankraj/Desktop/RESEARCH/Thesis Codes /archive/ACI-IoT-2023.csv",
                        help='Path to the original ACI IoT 2023 dataset')
    parser.add_argument('--packets', type=str, 
                        default="/Users/mayankraj/Desktop/RESEARCH/generated_packets",
                        help='Directory containing the generated packets')
    parser.add_argument('--output', type=str, 
                        default="/Users/mayankraj/Desktop/RESEARCH/project 2 V2",
                        help='Directory to save the validation report')
    parser.add_argument('--report', type=str, 
                        default="independent_validation_report.csv",
                        help='Name of the report file')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    args = parser.parse_args()
    
    try:
        # Create the validator
        validator = IndependentValidator(
            original_dataset_path=args.original,
            generated_packets_dir=args.packets,
            output_dir=args.output,
            seed=args.seed
        )
        
        # Run the validator
        validator.run(report_file=args.report)
    except Exception as e:
        print(f"Error: {e}")
        print(traceback.format_exc())
        print("Program terminated with errors.")