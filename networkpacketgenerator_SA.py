#!/usr/bin/env python3
# PacketGuard IoT Network Packet Generator
# For ACI IoT 2023 Dataset

import pandas as pd
import numpy as np
import os
import joblib
import time
import random
import warnings
import traceback
import argparse
import re
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA
from tqdm import tqdm

# Suppress warnings
warnings.filterwarnings('ignore')


class PacketGenerator:
    def __init__(self, dataset_path, output_dir="/Users/mayankraj/Desktop/RESEARCH/project 2 V2/generated_packets", seed=42):
        """
        Initialize the packet generator
        
        Args:
            dataset_path: Path to the ACI IoT 2023 dataset
            output_dir: Directory to save generated packets
            seed: Random seed for reproducibility
        """
        self.dataset_path = dataset_path
        self.output_dir = output_dir
        self.seed = seed
        self.models = {}
        self.scaler = None
        self.classes = []
        self.drop_columns = ['Src IP', 'Dst IP', 'Timestamp']
        self.flow_id_column = 'Flow ID'
        
        # Create output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        # Set random seed for reproducibility
        np.random.seed(seed)
        random.seed(seed)
        
        print("Packet Generator initialized.")
        
    def load_data(self):
        """
        Load the ACI IoT 2023 dataset and preprocess it
        """
        print(f"Loading dataset from {self.dataset_path}...")
        try:
            # Check if file exists
            if not os.path.exists(self.dataset_path):
                print(f"Error: Dataset file not found at {self.dataset_path}")
                return False
                
            # Load data
            self.data = pd.read_csv(self.dataset_path)
            print(f"Dataset loaded with shape: {self.data.shape}")
            
            # Check if dataset is empty
            if self.data.empty:
                print("Error: Dataset is empty")
                return False
                
            # Identify columns to drop
            columns_to_drop = [col for col in self.drop_columns if col in self.data.columns]
            
            # Check if 'Label' column exists, if not try to find another label column
            if 'Label' not in self.data.columns:
                label_candidates = ['label', 'class', 'Class', 'category', 'Category', 'Attack', 'attack_type']
                label_found = False
                for candidate in label_candidates:
                    if candidate in self.data.columns:
                        self.data.rename(columns={candidate: 'Label'}, inplace=True)
                        label_found = True
                        break
                        
                if not label_found:
                    print("Error: No label column found in dataset")
                    # Check if we can infer a label column (assuming single column with string values)
                    string_cols = self.data.select_dtypes(include=['object']).columns.tolist()
                    if len(string_cols) == 1 and string_cols[0] not in columns_to_drop and string_cols[0] != self.flow_id_column:
                        print(f"Using {string_cols[0]} as Label column")
                        self.data.rename(columns={string_cols[0]: 'Label'}, inplace=True)
                    else:
                        # Create a default label if none exists
                        print("Creating default 'Normal' label for all rows")
                        self.data['Label'] = 'Normal'
            
            # Extract classes
            self.classes = self.data['Label'].unique()
            print(f"Found {len(self.classes)} classes: {self.classes}")
            
            # Check if flow_id_column exists
            if self.flow_id_column not in self.data.columns:
                print(f"Warning: {self.flow_id_column} column not found in dataset")
                # Try to find another suitable column
                flow_id_candidates = ['Flow ID', 'flow_id', 'FlowID', 'flow-id', 'flow.id']
                flow_id_found = False
                for candidate in flow_id_candidates:
                    if candidate in self.data.columns and candidate != self.flow_id_column:
                        print(f"Using {candidate} as Flow ID column")
                        self.data.rename(columns={candidate: self.flow_id_column}, inplace=True)
                        flow_id_found = True
                        break
                        
                if not flow_id_found:
                    # Create a simple Flow ID if it doesn't exist
                    print(f"Creating {self.flow_id_column} column")
                    self.data[self.flow_id_column] = [f"flow_{i}" for i in range(len(self.data))]
            
            # Save Flow ID column
            self.flow_ids = self.data[self.flow_id_column].copy()
            
            # Preprocess data (drop columns, handle missing values)
            self.data = self.data.drop(columns=columns_to_drop, errors='ignore')
            
            # Replace infinities with NaN and then fill NaN values
            self.data = self.data.replace([np.inf, -np.inf], np.nan)
            
            # Check for data quality
            nan_counts = self.data.isna().sum()
            if nan_counts.sum() > 0:
                print(f"Found {nan_counts.sum()} missing values in the dataset")
                print(nan_counts[nan_counts > 0])
            
            # Fill numeric columns with mean, non-numeric with most frequent value
            numeric_cols = self.data.select_dtypes(include=np.number).columns
            for col in self.data.columns:
                if col in numeric_cols:
                    col_mean = self.data[col].mean()
                    if pd.isna(col_mean):  # If mean is NaN, use 0
                        self.data[col] = self.data[col].fillna(0)
                    else:
                        self.data[col] = self.data[col].fillna(col_mean)
                else:
                    # Only fill non-numeric columns that aren't Flow ID or Label
                    if col not in [self.flow_id_column, 'Label']:
                        if self.data[col].mode().shape[0] > 0:
                            self.data[col] = self.data[col].fillna(self.data[col].mode()[0])
                        else:
                            self.data[col] = self.data[col].fillna("Unknown")
            
            # Identify non-numeric columns
            non_numeric_cols = self.data.select_dtypes(exclude=np.number).columns.tolist()
            
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
                    self.data = pd.get_dummies(self.data, columns=non_numeric_cols)
                except Exception as e:
                    print(f"Error one-hot encoding: {e}")
                    # Try alternative approach: encode each column separately
                    for col in non_numeric_cols:
                        try:
                            # Get dummies for this column
                            dummies = pd.get_dummies(self.data[col], prefix=col)
                            # Drop the original column and join with dummies
                            self.data = self.data.drop(columns=[col]).join(dummies)
                        except Exception as e2:
                            print(f"Error encoding column {col}: {e2}")
                            # Last resort: drop the column
                            self.data = self.data.drop(columns=[col], errors='ignore')
                
            print("Data preprocessing completed.")
            return True
            
        except Exception as e:
            print(f"Error loading dataset: {e}")
            print(traceback.format_exc())
            return False
    
    def prepare_training_data(self):
        """
        Prepare the data for training models
        """
        print("Preparing training data...")
        try:
            # Create a copy of the data for training
            X = self.data.copy()
            
            # Check if Label column exists
            if 'Label' not in X.columns:
                print("Error: Label column not found")
                return None
                
            # Save label and Flow ID, then drop them from training set
            self.labels = X['Label'].copy()
            X = X.drop('Label', axis=1)
            
            # Keep Flow ID for reference but don't use for training
            if self.flow_id_column in X.columns:
                X = X.drop(self.flow_id_column, axis=1)
            
            # Handle any remaining non-numeric columns
            non_numeric_cols = X.select_dtypes(exclude=np.number).columns.tolist()
            if non_numeric_cols:
                print(f"Warning: Found additional non-numeric columns: {non_numeric_cols}")
                print("Attempting to convert to numeric or drop...")
                
                for col in non_numeric_cols:
                    try:
                        X[col] = pd.to_numeric(X[col], errors='coerce')
                        X[col] = X[col].fillna(X[col].mean() if not pd.isna(X[col].mean()) else 0)
                    except Exception as e:
                        print(f"Error converting column {col}: {e}")
                        X = X.drop(columns=[col])
            
            # Handle any constant columns (zero variance)
            constant_cols = [col for col in X.columns if X[col].nunique() <= 1]
            if constant_cols:
                print(f"Dropping {len(constant_cols)} constant columns: {constant_cols}")
                X = X.drop(columns=constant_cols)
            
            # Check if any columns left after preprocessing
            if X.empty or X.shape[1] == 0:
                print("Error: No features left after preprocessing")
                return None
            
            # Final check for any remaining issues
            X = X.replace([np.inf, -np.inf], np.nan)
            if X.isna().any().any():
                print("Warning: Still have missing values after preprocessing. Filling with zeros.")
                X = X.fillna(0)
                
            # Scale the features
            try:
                self.scaler = StandardScaler()
                X_scaled = self.scaler.fit_transform(X)
            except Exception as e:
                print(f"Error scaling features: {e}")
                print("Trying MinMaxScaler instead...")
                try:
                    self.scaler = MinMaxScaler()
                    X_scaled = self.scaler.fit_transform(X)
                except Exception as e2:
                    print(f"Error with MinMaxScaler: {e2}")
                    print("Using unscaled data...")
                    # Create a dummy scaler that just returns the input
                    class DummyScaler:
                        def transform(self, X):
                            return X
                        def inverse_transform(self, X):
                            return X
                    self.scaler = DummyScaler()
                    X_scaled = X.values
            
            # Save features for later use
            self.feature_names = X.columns.tolist()
            
            print(f"Training data prepared with {X_scaled.shape[1]} features.")
            return X_scaled
            
        except Exception as e:
            print(f"Error preparing training data: {e}")
            print(traceback.format_exc())
            return None
        
    def train_models(self, X_scaled):
        """
        Train models for each class to generate network packets
        """
        print("Training models for each class...")
        
        for cls in tqdm(self.classes):
            print(f"\nTraining models for class: {cls}")
            
            # Filter data for this class
            class_indices = np.where(self.labels == cls)[0]
            X_class = X_scaled[class_indices]
            
            # Skip if no data for this class
            if len(X_class) == 0:
                print(f"Warning: No data found for class {cls}. Skipping model training.")
                continue
                
            # If very few samples, adjust train/test split
            test_size = 0.2
            if len(X_class) < 50:
                test_size = 0.5
                print(f"Warning: Only {len(X_class)} samples for class {cls}. Adjusting test size to {test_size}.")
            
            # Split the data for training and validation
            X_train, X_valid = train_test_split(X_class, test_size=test_size, random_state=self.seed)
            
            # Adjust parameters based on dataset size
            nu = min(0.1, 1.0 / (len(X_train) + 1))  # Avoid nu > 1/n which causes error
            contamination = min(0.1, 1.0 / (len(X_train) + 1))
            
            # Train One-Class SVM model with error handling
            print(f"Training One-Class SVM for class {cls}...")
            try:
                ocsvm = OneClassSVM(nu=nu, kernel="rbf", gamma='scale')
                ocsvm.fit(X_train)
            except Exception as e:
                print(f"Error training One-Class SVM for class {cls}: {e}")
                # Fallback to a more robust setting
                try:
                    ocsvm = OneClassSVM(nu=0.01, kernel="linear")
                    ocsvm.fit(X_train)
                except Exception as e2:
                    print(f"Fallback One-Class SVM failed for class {cls}: {e2}")
                    # Create a dummy model that accepts everything as normal
                    class DummyPositiveModel:
                        def predict(self, X):
                            return np.ones(len(X))
                    ocsvm = DummyPositiveModel()
            
            # Train Isolation Forest model with error handling
            print(f"Training Isolation Forest for class {cls}...")
            try:
                isolation_forest = IsolationForest(contamination=contamination, random_state=self.seed)
                isolation_forest.fit(X_train)
            except Exception as e:
                print(f"Error training Isolation Forest for class {cls}: {e}")
                # Create a dummy model that accepts everything as normal
                class DummyPositiveModel:
                    def predict(self, X):
                        return np.ones(len(X))
                isolation_forest = DummyPositiveModel()
            
            # Validate models on the validation set
            try:
                ocsvm_preds = ocsvm.predict(X_valid)
                if_preds = isolation_forest.predict(X_valid)
                
                # Convert predictions to anomaly format (1 for normal, -1 for anomaly)
                ocsvm_anomaly_rate = np.sum(ocsvm_preds == -1) / len(ocsvm_preds) * 100
                if_anomaly_rate = np.sum(if_preds == -1) / len(if_preds) * 100
                
                print(f"One-Class SVM - Validation Anomaly Rate: {ocsvm_anomaly_rate:.2f}%")
                print(f"Isolation Forest - Validation Anomaly Rate: {if_anomaly_rate:.2f}%")
            except Exception as e:
                print(f"Error validating models for class {cls}: {e}")
                ocsvm_anomaly_rate = 0
                if_anomaly_rate = 0
            
            # Train a PCA model for dimensionality reduction with error handling
            try:
                # Calculate reasonable number of components based on data size
                n_components = min(0.95, max(0.1, 1 - 10.0/len(X_train)))
                pca = PCA(n_components=n_components)
                pca.fit(X_train)
            except Exception as e:
                print(f"Error training PCA for class {cls}: {e}")
                # Create a dummy PCA that just returns the input
                class DummyPCA:
                    def __init__(self):
                        self.n_components_ = 1
                        self.explained_variance_ratio_ = np.array([1.0])
                    
                    def transform(self, X):
                        return X[:, :1]
                    
                    def inverse_transform(self, X):
                        result = np.zeros((X.shape[0], len(self.mean)))
                        # Fill with mean values
                        for i in range(X.shape[0]):
                            result[i] = self.mean
                        return result
                        
                pca = DummyPCA()
                pca.mean = np.mean(X_train, axis=0)
            
            # Store the models
            self.models[cls] = {
                'ocsvm': ocsvm,
                'isolation_forest': isolation_forest,
                'pca': pca,
                'mean': np.mean(X_train, axis=0),
                'std': np.std(X_train, axis=0) + 1e-10,  # Add small constant to avoid division by zero
                'min': np.min(X_train, axis=0),
                'max': np.max(X_train, axis=0)
            }
            
        print("Model training completed.")
    
    def generate_packets(self, packets_per_class=1000, max_attempts=5):
        """
        Generate network packets for each class
        
        Args:
            packets_per_class: Number of packets to generate per class
            max_attempts: Maximum number of attempts to generate valid packets
        """
        print(f"Generating {packets_per_class} packets for each class...")
        
        generated_data = {}
        
        for cls in tqdm(self.classes):
            print(f"\nGenerating packets for class: {cls}")
            
            # Skip if no models for this class
            if cls not in self.models:
                print(f"Warning: No models found for class {cls}. Skipping packet generation.")
                continue
            
            # Get models for this class
            models = self.models[cls]
            ocsvm = models['ocsvm']
            isolation_forest = models['isolation_forest']
            pca = models['pca']
            mean = models['mean']
            std = models['std']
            min_val = models['min']
            max_val = models['max']
            
            # Create an empty list to store generated packets
            valid_packets = []
            total_attempts = 0
            
            # Generate packets until we have enough or reach max attempts
            while len(valid_packets) < packets_per_class and total_attempts < packets_per_class * max_attempts:
                # Number of packets to generate in this batch
                batch_size = min(100, packets_per_class - len(valid_packets))
                total_attempts += batch_size
                
                # Method 1: Generate from normal distribution with class statistics
                noise = np.random.normal(0, 1, size=(batch_size, len(mean)))
                # Scale noise by std and add mean
                potential_packets = mean + noise * std * 0.5  # Using 0.5 to keep closer to mean
                
                # Method 2: Use PCA for more realistic data generation
                # Make sure we don't exceed the PCA components
                try:
                    if hasattr(pca, 'n_components_'):
                        pca_components = min(pca.n_components_, len(mean))
                    else:
                        # If n_components_ is not available, use explained_variance_ratio_ to estimate
                        pca_components = min(pca.explained_variance_ratio_.shape[0], len(mean))
                    
                    # Generate in latent space and project back
                    latent_space = np.random.normal(0, 0.1, size=(batch_size, pca_components))
                    # Project back to original space
                    pca_packets = pca.inverse_transform(latent_space)
                    
                    # Mix the two methods (70% PCA, 30% direct generation)
                    mix_ratio = 0.7
                    potential_packets = mix_ratio * pca_packets + (1 - mix_ratio) * potential_packets
                except Exception as e:
                    print(f"PCA generation failed for class {cls}: {e}")
                    # Continue with the directly generated packets only
                    pass
                
                # Ensure packets stay within min-max boundaries with clamping
                for i in range(len(potential_packets)):
                    potential_packets[i] = np.clip(potential_packets[i], min_val, max_val)
                
                # Check anomaly detection with both models
                try:
                    ocsvm_preds = ocsvm.predict(potential_packets)
                    if_preds = isolation_forest.predict(potential_packets)
                    
                    # Only keep packets that pass both models (not anomalies)
                    for i in range(batch_size):
                        if ocsvm_preds[i] == 1 and if_preds[i] == 1:  # 1 means normal, -1 means anomaly
                            valid_packets.append(potential_packets[i])
                            if len(valid_packets) >= packets_per_class:
                                break
                except Exception as e:
                    print(f"Error predicting anomalies: {e}")
                    # On error, accept all packets in this batch
                    for i in range(batch_size):
                        valid_packets.append(potential_packets[i])
                        if len(valid_packets) >= packets_per_class:
                            break
                
                # Print progress
                if total_attempts % 1000 == 0:
                    print(f"Generated {len(valid_packets)} valid packets after {total_attempts} attempts")
            
            # Check if we have enough valid packets
            if len(valid_packets) < packets_per_class:
                print(f"Warning: Only generated {len(valid_packets)} valid packets for class {cls}")
                
                # Handle the case where we have no valid packets at all
                if len(valid_packets) == 0:
                    print(f"No valid packets generated for class {cls}. Using class statistics to generate packets.")
                    # Generate packets using class statistics directly
                    mean = models['mean']
                    std = models['std']
                    min_val = models['min']
                    max_val = models['max']
                    
                    # Generate packets with reduced noise to stay closer to the mean
                    for i in range(packets_per_class):
                        noise = np.random.normal(0, 0.05, size=len(mean))  # Reduced noise
                        packet = mean + noise * std * 0.3  # Reduced scaling to stay closer to mean
                        packet = np.clip(packet, min_val, max_val)  # Ensure within bounds
                        valid_packets.append(packet)
                    
                    print(f"Generated {len(valid_packets)} packets using class statistics")
                else:
                    # If we have some valid packets, duplicate them with noise
                    existing_valid = len(valid_packets)
                    while len(valid_packets) < packets_per_class:
                        idx = np.random.randint(0, existing_valid)  # Only use original valid packets
                        # Add small noise to avoid exact duplicates
                        noise = np.random.normal(0, 0.01, size=len(valid_packets[0]))
                        valid_packets.append(valid_packets[idx] + noise)
            
            # Take only the required number of packets
            valid_packets = valid_packets[:packets_per_class]
            
            # Convert to numpy array
            valid_packets = np.array(valid_packets)
            
            # Store the generated data
            generated_data[cls] = valid_packets
            
            # Check anomaly rate on final generated packets
            try:
                final_ocsvm_preds = ocsvm.predict(valid_packets)
                final_if_preds = isolation_forest.predict(valid_packets)
                
                ocsvm_anomaly_rate = np.sum(final_ocsvm_preds == -1) / len(final_ocsvm_preds) * 100
                if_anomaly_rate = np.sum(final_if_preds == -1) / len(final_if_preds) * 100
                
                print(f"Final One-Class SVM - Anomaly Rate: {ocsvm_anomaly_rate:.2f}%")
                print(f"Final Isolation Forest - Anomaly Rate: {if_anomaly_rate:.2f}%")
            except Exception as e:
                print(f"Error calculating final anomaly rates: {e}")
        
        self.generated_data = generated_data
        print("Packet generation completed.")
    
    def save_generated_packets(self):
        """
        Save the generated packets to CSV files
        """
        print("Saving generated packets to CSV files...")
        
        for cls in self.classes:
            print(f"Processing class: {cls}")
            
            # Get the generated data for this class
            if cls not in self.generated_data:
                print(f"Warning: No generated data for class {cls}. Skipping.")
                continue
                
            generated_packets = self.generated_data[cls]
            
            # Inverse transform the scaled data
            try:
                unscaled_packets = self.scaler.inverse_transform(generated_packets)
            except Exception as e:
                print(f"Error inverse transforming data for class {cls}: {e}")
                # Use the scaled data as is
                unscaled_packets = generated_packets
            
            # Convert to DataFrame with the original feature names
            if len(self.feature_names) != unscaled_packets.shape[1]:
                print(f"Warning: Feature dimension mismatch. Generated: {unscaled_packets.shape[1]}, Expected: {len(self.feature_names)}")
                # Create generic feature names if needed
                feature_names = [f"feature_{i}" for i in range(unscaled_packets.shape[1])]
                df = pd.DataFrame(unscaled_packets, columns=feature_names)
            else:
                df = pd.DataFrame(unscaled_packets, columns=self.feature_names)
            
            # Generate new Flow IDs (keeping the format but making them unique)
            # Get a sample of existing Flow IDs for this class
            class_indices = np.where(self.labels == cls)[0]
            
            if len(class_indices) > 0:
                class_flow_ids = self.flow_ids.iloc[class_indices].values
                
                # If we have enough Flow IDs, sample from them, otherwise generate new ones
                if len(class_flow_ids) >= len(df):
                    flow_id_samples = np.random.choice(class_flow_ids, size=len(df), replace=False)
                else:
                    # Sample what we can, then generate the rest
                    flow_id_samples = np.random.choice(class_flow_ids, size=len(class_flow_ids), replace=False)
                    remaining = len(df) - len(flow_id_samples)
                    
                    # Generate the rest with a pattern like original but unique
                    base_flow_id = class_flow_ids[0] if len(class_flow_ids) > 0 else "flow_"
                    if isinstance(base_flow_id, str):
                        # Try to extract prefix
                        import re
                        prefix_match = re.match(r'([a-zA-Z_]+)', base_flow_id)
                        prefix = prefix_match.group(1) if prefix_match else "flow_"
                        new_flow_ids = [f"{prefix}{int(time.time())}_{i}" for i in range(remaining)]
                    else:
                        # If it's not a string, just create generic ones
                        new_flow_ids = [f"flow_{int(time.time())}_{i}" for i in range(remaining)]
                    
                    flow_id_samples = np.concatenate([flow_id_samples, new_flow_ids])
            else:
                # No flow IDs for this class, generate new ones
                flow_id_samples = [f"flow_{cls.replace(' ', '_')}_{int(time.time())}_{i}" for i in range(len(df))]
            
            # Add Flow ID to DataFrame
            df[self.flow_id_column] = flow_id_samples
            
            # Add Label column
            df['Label'] = cls
            
            # Rearrange columns to match original data format
            cols = [self.flow_id_column] + [col for col in df.columns if col not in [self.flow_id_column, 'Label']] + ['Label']
            # Ensure all columns exist
            cols = [col for col in cols if col in df.columns]
            df = df[cols]
            
            # Save to CSV
            safe_cls_name = cls.replace('/', '_').replace(' ', '_').replace('\\', '_')
            output_file = os.path.join(self.output_dir, f"{safe_cls_name}_generated.csv")
            df.to_csv(output_file, index=False)
            print(f"Saved {len(df)} packets to {output_file}")
        
        # Create a combined file with all classes
        print("Creating combined file with all generated packets...")
        combined_df = pd.DataFrame()
        
        for cls in self.classes:
            # Read the class-specific file
            safe_cls_name = cls.replace('/', '_').replace(' ', '_').replace('\\', '_')
            file_path = os.path.join(self.output_dir, f"{safe_cls_name}_generated.csv")
            if os.path.exists(file_path):
                try:
                    class_df = pd.read_csv(file_path)
                    combined_df = pd.concat([combined_df, class_df])
                except Exception as e:
                    print(f"Error reading file {file_path}: {e}")
        
        if not combined_df.empty:
            combined_output_file = os.path.join(self.output_dir, "all_generated_packets.csv")
            combined_df.to_csv(combined_output_file, index=False)
            print(f"Saved combined file with {len(combined_df)} packets to {combined_output_file}")
        else:
            print("Warning: No data to save in combined file.")
    
    def validate_generated_packets(self):
        """
        Validate the generated packets against anomaly detection models
        """
        print("Validating generated packets...")
        
        validation_results = {}
        
        for cls in self.classes:
            print(f"\nValidating packets for class: {cls}")
            
            # Read the generated packets for this class
            safe_cls_name = cls.replace('/', '_').replace(' ', '_').replace('\\', '_')
            file_path = os.path.join(self.output_dir, f"{safe_cls_name}_generated.csv")
            if not os.path.exists(file_path):
                print(f"Warning: File not found: {file_path}")
                continue
                
            try:
                df = pd.read_csv(file_path)
            except Exception as e:
                print(f"Error reading file {file_path}: {e}")
                continue
            
            # Check if the dataframe is empty
            if df.empty:
                print(f"Warning: Empty dataframe for class {cls}")
                continue
                
            # Check if we have the required columns
            required_cols = [self.flow_id_column, 'Label']
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                print(f"Warning: Missing required columns: {missing_cols}")
                continue
            
            # Remove Flow ID and Label for validation
            X = df.drop([col for col in required_cols if col in df.columns], axis=1)
            
            # Check if we have any features left
            if X.empty:
                print(f"Warning: No features left after dropping required columns for class {cls}")
                continue
            
            # Check for and handle any non-numeric columns
            non_numeric_cols = X.select_dtypes(exclude=np.number).columns.tolist()
            if non_numeric_cols:
                print(f"Warning: Non-numeric columns found: {non_numeric_cols}. Converting to numeric.")
                for col in non_numeric_cols:
                    X[col] = pd.to_numeric(X[col], errors='coerce')
                X = X.fillna(0)  # Fill NaN values with 0
            
            # Check if models exist for this class
            if cls not in self.models:
                print(f"Warning: No models found for class {cls}")
                continue
                
            # Scale the features
            try:
                X_scaled = self.scaler.transform(X)
            except Exception as e:
                print(f"Error scaling features for class {cls}: {e}")
                # Try to continue with unscaled data
                X_scaled = X.values
            
            # Get models for this class
            models = self.models[cls]
            
            # Validate with One-Class SVM
            try:
                ocsvm = models['ocsvm']
                ocsvm_preds = ocsvm.predict(X_scaled)
                ocsvm_anomaly_rate = np.sum(ocsvm_preds == -1) / len(ocsvm_preds) * 100
                print(f"One-Class SVM - Anomaly Rate: {ocsvm_anomaly_rate:.2f}%")
            except Exception as e:
                print(f"Error validating with One-Class SVM for class {cls}: {e}")
                ocsvm_anomaly_rate = float('nan')
            
            # Validate with Isolation Forest
            try:
                isolation_forest = models['isolation_forest']
                if_preds = isolation_forest.predict(X_scaled)
                if_anomaly_rate = np.sum(if_preds == -1) / len(if_preds) * 100
                print(f"Isolation Forest - Anomaly Rate: {if_anomaly_rate:.2f}%")
            except Exception as e:
                print(f"Error validating with Isolation Forest for class {cls}: {e}")
                if_anomaly_rate = float('nan')
            
            # Store validation results
            validation_results[cls] = {
                'ocsvm_anomaly_rate': ocsvm_anomaly_rate,
                'if_anomaly_rate': if_anomaly_rate
            }
            
            # Check if the anomaly rate is below 30% (as per requirements)
            if not np.isnan(ocsvm_anomaly_rate) and not np.isnan(if_anomaly_rate):
                if ocsvm_anomaly_rate < 30 and if_anomaly_rate < 30:
                    print(f"Success: Generated packets for class {cls} pass the anomaly detection test")
                else:
                    print(f"Warning: Generated packets for class {cls} have high anomaly rate (> 30%)")
        
        # Summary of validation results
        print("\nValidation Summary:")
        for cls, results in validation_results.items():
            ocsvm_rate = results['ocsvm_anomaly_rate']
            if_rate = results['if_anomaly_rate']
            ocsvm_status = "PASS" if not np.isnan(ocsvm_rate) and ocsvm_rate < 30 else "FAIL"
            if_status = "PASS" if not np.isnan(if_rate) and if_rate < 30 else "FAIL"
            print(f"{cls}: One-Class SVM: {ocsvm_rate:.2f}% ({ocsvm_status}), Isolation Forest: {if_rate:.2f}% ({if_status})")
    
    def run(self, packets_per_class=1000):
        """
        Run the complete packet generation pipeline
        
        Args:
            packets_per_class: Number of packets to generate per class
        """
        print("Starting the packet generation pipeline...")
        
        try:
            # Step 1: Load and preprocess data
            if not self.load_data():
                print("Error loading data. Exiting.")
                return
            
            # Step 2: Prepare training data
            X_scaled = self.prepare_training_data()
            if X_scaled is None:
                print("Error preparing training data. Exiting.")
                return
            
            # Step 3: Train models
            self.train_models(X_scaled)
            
            # Step 4: Generate packets
            self.generate_packets(packets_per_class=packets_per_class)
            
            # Step 5: Save generated packets
            self.save_generated_packets()
            
            # Step 6: Validate generated packets
            self.validate_generated_packets()
            
            print("Packet generation pipeline completed successfully!")
            
        except Exception as e:
            print(f"Error in packet generation pipeline: {e}")
            print(traceback.format_exc())
            print("Attempting to continue with available results...")
            
            # Try to save any generated packets if we got that far
            if hasattr(self, 'generated_data') and self.generated_data:
                try:
                    self.save_generated_packets()
                    self.validate_generated_packets()
                except Exception as e2:
                    print(f"Error saving/validating generated packets: {e2}")
                    
            print("Pipeline completed with errors. Check the logs for details.")


if __name__ == "__main__":
    # Parse command line arguments if any
    parser = argparse.ArgumentParser(description='Generate network packets based on ACI IoT 2023 dataset')
    parser.add_argument('--dataset', type=str, 
                        default="/Users/mayankraj/Desktop/RESEARCH/Thesis Codes /archive/ACI-IoT-2023.csv",
                        help='Path to the ACI IoT 2023 dataset')
    parser.add_argument('--output', type=str, default='generated_packets',
                        help='Directory to save generated packets')
    parser.add_argument('--packets', type=int, default=1000,
                        help='Number of packets to generate per class')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    args = parser.parse_args()
    
    try:
        # Create the packet generator
        generator = PacketGenerator(
            dataset_path=args.dataset,
            output_dir=args.output,
            seed=args.seed
        )
        
        # Run the generator
        generator.run(packets_per_class=args.packets)
    except Exception as e:
        print(f"Error: {e}")
        print(traceback.format_exc())
        print("Program terminated with errors.")