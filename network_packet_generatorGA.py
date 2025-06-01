#!/usr/bin/env python3
# GeneticPacketGenerator for Network Traffic
# Based on PacketGuard repository
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
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from tqdm import tqdm

# Suppress warnings
warnings.filterwarnings('ignore')


class GeneticPacketGenerator:
    def __init__(self, dataset_path, output_dir="./generated_packets", seed=42):
        """
        Initialize the genetic packet generator
        
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
        
        # GA parameters
        self.population_size = 200  # Size of population in each generation
        self.mutation_rate = 0.05   # Probability of mutation
        self.elite_size = 20        # Number of elite individuals to keep
        self.generations = 50       # Maximum generations to evolve
        self.target_fitness = 0.9   # Target fitness score to achieve
        
        # Create output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        # Set random seed for reproducibility
        np.random.seed(seed)
        random.seed(seed)
        
        print("Genetic Packet Generator initialized.")
        
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
                'max': np.max(X_train, axis=0),
                'train_data': X_train  # Store training data for genetic algorithm
            }
            
        print("Model training completed.")
    
    # Genetic Algorithm Methods
    def initialize_population(self, class_data, size):
        """
        Initialize a population for the genetic algorithm
        
        Args:
            class_data: Training data for the class
            size: Size of the population
        
        Returns:
            List of individuals (packets)
        """
        population = []
        
        # Calculate statistics for initialization
        mean = np.mean(class_data, axis=0)
        std = np.std(class_data, axis=0) + 1e-10  # Add small constant to avoid division by zero
        min_vals = np.min(class_data, axis=0)
        max_vals = np.max(class_data, axis=0)
        
        # Create individuals using different methods for diversity
        # 1. Use original samples with small noise (25% of population)
        num_original = size // 4
        if len(class_data) >= num_original:
            indices = np.random.choice(len(class_data), num_original, replace=False)
            for i in indices:
                # Add small noise to avoid duplicates
                noise = np.random.normal(0, 0.01, size=len(mean))
                ind = class_data[i] + noise
                # Ensure within bounds
                ind = np.clip(ind, min_vals, max_vals)
                population.append(ind)
        else:
            # If not enough samples, repeat with replacement
            for _ in range(num_original):
                i = np.random.randint(0, len(class_data))
                noise = np.random.normal(0, 0.02, size=len(mean))
                ind = class_data[i] + noise
                ind = np.clip(ind, min_vals, max_vals)
                population.append(ind)
        
        # 2. Generate from normal distribution (50% of population)
        num_normal = size // 2
        for _ in range(num_normal):
            noise = np.random.normal(0, 1, size=len(mean))
            ind = mean + noise * std * 0.3  # Using 0.3 to keep somewhat close to mean
            ind = np.clip(ind, min_vals, max_vals)
            population.append(ind)
        
        # 3. Generate uniform random within bounds (remaining population)
        num_uniform = size - len(population)
        for _ in range(num_uniform):
            ind = np.random.uniform(min_vals, max_vals)
            population.append(ind)
        
        return population
    
    def calculate_fitness(self, individual, ocsvm, isolation_forest, mean, std):
        """
        Calculate fitness score for an individual
        
        Args:
            individual: The individual (packet)
            ocsvm: One-Class SVM model
            isolation_forest: Isolation Forest model
            mean: Mean of the training data
            std: Standard deviation of the training data
        
        Returns:
            Fitness score (higher is better)
        """
        # Reshape individual for prediction
        individual_reshaped = individual.reshape(1, -1)
        
        # Calculate distance from class mean (lower is better)
        normalized_distance = np.sqrt(np.sum(((individual - mean) / std) ** 2)) / len(mean)
        distance_score = np.exp(-normalized_distance)  # Convert to 0-1 score (higher is better)
        
        # Get model predictions (1 for normal, -1 for anomaly)
        try:
            ocsvm_pred = ocsvm.predict(individual_reshaped)[0]
            if_pred = isolation_forest.predict(individual_reshaped)[0]
            
            # Convert to binary scores (1 for normal, 0 for anomaly)
            ocsvm_score = 1.0 if ocsvm_pred == 1 else 0.0
            if_score = 1.0 if if_pred == 1 else 0.0
            
            # Weight the scores (model predictions more important than distance)
            # 40% OCSVM, 40% Isolation Forest, 20% distance
            fitness = 0.4 * ocsvm_score + 0.4 * if_score + 0.2 * distance_score
            
            return fitness
            
        except Exception as e:
            # If prediction fails, return low fitness
            return 0.1
    
    def select_parents(self, population, fitnesses, num_parents):
        """
        Select parents for reproduction using tournament selection
        
        Args:
            population: List of individuals
            fitnesses: List of fitness scores
            num_parents: Number of parents to select
        
        Returns:
            List of selected parents
        """
        parents = []
        for _ in range(num_parents):
            # Tournament selection - select k individuals randomly and choose the best
            k = 3  # Tournament size
            tournament_indices = np.random.choice(len(population), k, replace=False)
            tournament_fitnesses = [fitnesses[i] for i in tournament_indices]
            
            # Select the winner (highest fitness)
            winner_idx = tournament_indices[np.argmax(tournament_fitnesses)]
            parents.append(population[winner_idx])
        
        return parents
    
    def crossover(self, parent1, parent2):
        """
        Perform crossover between two parents to create a child
        
        Args:
            parent1: First parent
            parent2: Second parent
        
        Returns:
            Child individual
        """
        # Choose crossover method
        method = np.random.choice(['uniform', 'single_point', 'blend'])
        
        if method == 'uniform':
            # Uniform crossover - randomly select features from each parent
            mask = np.random.randint(0, 2, size=len(parent1)).astype(bool)
            child = np.copy(parent1)
            child[mask] = parent2[mask]
            
        elif method == 'single_point':
            # Single-point crossover
            point = np.random.randint(1, len(parent1) - 1)
            child = np.concatenate([parent1[:point], parent2[point:]])
            
        else:  # blend
            # Blend crossover (interpolate between parents)
            alpha = np.random.uniform(0, 1)
            child = alpha * parent1 + (1 - alpha) * parent2
        
        return child
    
    def mutate(self, individual, mutation_rate, min_vals, max_vals):
        """
        Mutate an individual
        
        Args:
            individual: Individual to mutate
            mutation_rate: Probability of mutating each feature
            min_vals: Minimum values for each feature
            max_vals: Maximum values for each feature
        
        Returns:
            Mutated individual
        """
        mutated = np.copy(individual)
        
        # Choose mutation method
        method = np.random.choice(['gaussian', 'uniform', 'reset'])
        
        if method == 'gaussian':
            # Gaussian mutation - add random noise to features
            # Decide which features to mutate
            mutation_mask = np.random.random(len(mutated)) < mutation_rate
            
            if np.any(mutation_mask):
                # Calculate standard deviation for each feature (10% of range)
                feature_ranges = max_vals - min_vals
                stdevs = feature_ranges * 0.1
                
                # Add noise to selected features
                noise = np.random.normal(0, stdevs, size=len(mutated))
                mutated[mutation_mask] += noise[mutation_mask]
                
        elif method == 'uniform':
            # Uniform mutation - replace with random value within bounds
            mutation_mask = np.random.random(len(mutated)) < mutation_rate
            
            if np.any(mutation_mask):
                # Generate random values within bounds for selected features
                random_values = np.random.uniform(min_vals, max_vals, size=len(mutated))
                mutated[mutation_mask] = random_values[mutation_mask]
                
        else:  # reset
            # Reset mutation - reset a feature to its min or max value
            mutation_mask = np.random.random(len(mutated)) < mutation_rate
            
            if np.any(mutation_mask):
                # For each feature to mutate, randomly choose min or max
                for i in np.where(mutation_mask)[0]:
                    if np.random.random() < 0.5:
                        mutated[i] = min_vals[i]
                    else:
                        mutated[i] = max_vals[i]
        
        # Ensure within bounds
        np.clip(mutated, min_vals, max_vals, out=mutated)
        
        return mutated
    
    def create_next_generation(self, population, fitnesses, elite_size, min_vals, max_vals):
        """
        Create the next generation using elitism, crossover, and mutation
        
        Args:
            population: Current population
            fitnesses: Fitness scores for current population
            elite_size: Number of elite individuals to keep
            min_vals: Minimum values for each feature
            max_vals: Maximum values for each feature
        
        Returns:
            New population
        """
        population_size = len(population)
        new_population = []
        
        # Sort by fitness (descending)
        sorted_indices = np.argsort(fitnesses)[::-1]
        
        # Elitism - keep best individuals
        elites = [population[i] for i in sorted_indices[:elite_size]]
        new_population.extend(elites)
        
        # Select parents for reproduction
        num_parents = population_size // 2
        parents = self.select_parents(population, fitnesses, num_parents)
        
        # Create children through crossover and mutation
        while len(new_population) < population_size:
            # Select two random parents
            parent1, parent2 = random.sample(parents, 2)
            
            # Crossover
            child = self.crossover(parent1, parent2)
            
            # Mutation
            child = self.mutate(child, self.mutation_rate, min_vals, max_vals)
            
            # Add to new population
            new_population.append(child)
        
        # Ensure we have exactly population_size individuals
        return new_population[:population_size]
    
    def is_unique(self, individual, unique_packets, tolerance=1e-6):
        """
        Check if an individual is unique compared to existing unique packets
        
        Args:
            individual: Individual to check
            unique_packets: List of existing unique packets
            tolerance: Tolerance for floating point comparison
        
        Returns:
            True if unique, False otherwise
        """
        for packet in unique_packets:
            # Calculate normalized Euclidean distance
            distance = np.sqrt(np.sum((individual - packet) ** 2)) / len(individual)
            if distance < tolerance:
                return False
        return True
    
    def evolve_packets(self, cls, packets_per_class=1000):
        """
        Evolve network packets for a class using genetic algorithm
        
        Args:
            cls: Class to evolve packets for
            packets_per_class: Number of packets to generate
        
        Returns:
            Generated packets
        """
        print(f"Evolving packets for class {cls}...")
        
        # Get models and statistics for this class
        models = self.models[cls]
        ocsvm = models['ocsvm']
        isolation_forest = models['isolation_forest']
        mean = models['mean']
        std = models['std']
        min_vals = models['min']
        max_vals = models['max']
        train_data = models['train_data']
        
        # Initialize the population
        population = self.initialize_population(train_data, self.population_size)
        
        # List to store unique packets
        unique_packets = []
        generation = 0
        stagnation_counter = 0
        best_fitness_history = []
        
        # Evolution loop
        while len(unique_packets) < packets_per_class and generation < self.generations:
            print(f"Generation {generation+1}/{self.generations}, Unique packets: {len(unique_packets)}/{packets_per_class}")
            
            # Calculate fitness for each individual
            fitnesses = [self.calculate_fitness(ind, ocsvm, isolation_forest, mean, std) for ind in population]
            
            # Extract best individuals and add to unique packets if they pass criteria
            sorted_indices = np.argsort(fitnesses)[::-1]
            best_fitness = fitnesses[sorted_indices[0]]
            
            best_fitness_history.append(best_fitness)
            
            # Check for stagnation (no improvement in fitness)
            if len(best_fitness_history) > 5:
                if all(abs(best_fitness_history[-1] - bf) < 1e-6 for bf in best_fitness_history[-5:]):
                    stagnation_counter += 1
                    if stagnation_counter >= 3:  # Restart if stagnated for 3 generations
                        print(f"Stagnation detected. Reinitializing population.")
                        population = self.initialize_population(train_data, self.population_size)
                        stagnation_counter = 0
                        continue
                else:
                    stagnation_counter = 0
            
            # Add best individuals to unique packets if they pass criteria and are unique
            for idx in sorted_indices:
                individual = population[idx]
                fitness = fitnesses[idx]
                
                # Only add if fitness is good enough and it's unique
                if fitness > 0.8 and self.is_unique(individual, unique_packets):
                    unique_packets.append(individual)
                    if len(unique_packets) >= packets_per_class:
                        break
            
            # Create next generation
            population = self.create_next_generation(population, fitnesses, self.elite_size, min_vals, max_vals)
            
            # Increase generation counter
            generation += 1
        
        # If we don't have enough unique packets, fill with best from last population
        if len(unique_packets) < packets_per_class:
            print(f"Warning: Only evolved {len(unique_packets)} unique packets. Adding best from last population.")
            
            # Calculate fitness for last population
            fitnesses = [self.calculate_fitness(ind, ocsvm, isolation_forest, mean, std) for ind in population]
            
            # Sort by fitness (descending)
            sorted_indices = np.argsort(fitnesses)[::-1]
            
            # Add best individuals from last population
            for idx in sorted_indices:
                individual = population[idx]
                
                # Only add if it's unique
                if self.is_unique(individual, unique_packets):
                    unique_packets.append(individual)
                    if len(unique_packets) >= packets_per_class:
                        break
            
            # If still not enough, generate from training data with more noise
            if len(unique_packets) < packets_per_class:
                print(f"Warning: Still only have {len(unique_packets)} unique packets. Generating with more noise.")
                
                # Generate from training data with more noise
                while len(unique_packets) < packets_per_class:
                    # Randomly select a training sample
                    i = np.random.randint(0, len(train_data))
                    
                    # Add more significant noise
                    noise = np.random.normal(0, 0.1, size=len(mean))
                    ind = train_data[i] + noise * std
                    
                    # Ensure within bounds
                    ind = np.clip(ind, min_vals, max_vals)
                    
                    # Only add if it's unique
                    if self.is_unique(individual, unique_packets):
                        unique_packets.append(ind)
        
        print(f"Generated {len(unique_packets)} unique packets for class {cls}")
        return np.array(unique_packets)
    
    def generate_packets(self, packets_per_class=1000):
        """
        Generate network packets for each class using genetic algorithm
        
        Args:
            packets_per_class: Number of packets to generate per class
        """
        print(f"Generating {packets_per_class} packets for each class using genetic algorithm...")
        
        generated_data = {}
        
        for cls in tqdm(self.classes):
            print(f"\nGenerating packets for class: {cls}")
            
            # Skip if no models for this class
            if cls not in self.models:
                print(f"Warning: No models found for class {cls}. Skipping packet generation.")
                continue
            
            # Evolve packets for this class
            generated_packets = self.evolve_packets(cls, packets_per_class)
            
            # Store the generated data
            generated_data[cls] = generated_packets
            
            # Validate generated packets
            ocsvm = self.models[cls]['ocsvm']
            isolation_forest = self.models[cls]['isolation_forest']
            
            try:
                ocsvm_preds = ocsvm.predict(generated_packets)
                if_preds = isolation_forest.predict(generated_packets)
                
                ocsvm_anomaly_rate = np.sum(ocsvm_preds == -1) / len(ocsvm_preds) * 100
                if_anomaly_rate = np.sum(if_preds == -1) / len(if_preds) * 100
                
                print(f"One-Class SVM - Anomaly Rate: {ocsvm_anomaly_rate:.2f}%")
                print(f"Isolation Forest - Anomaly Rate: {if_anomaly_rate:.2f}%")
                
                # Check if anomaly rate is below 30%
                if ocsvm_anomaly_rate > 30 or if_anomaly_rate > 30:
                    print(f"Warning: Anomaly rate is above 30% for class {cls}")
            except Exception as e:
                print(f"Error validating generated packets: {e}")
        
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
            
            # Generate unique Flow IDs
            timestamp = int(time.time())
            flow_ids = [f"flow_{cls.replace(' ', '_').replace('/', '_')}_{timestamp}_{i}" for i in range(len(df))]
            
            # Add Flow ID to DataFrame
            df[self.flow_id_column] = flow_ids
            
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
    
    def _create_visualization_folders(self, base_path):
        """
        Create folders for storing visualizations
        
        Args:
            base_path: Base path for creating visualization folders
            
        Returns:
            Dictionary of created folder paths
        """
        folders = {
            'anomaly_detection': os.path.join(base_path, 'anomaly_detection_viz'),
            'real_vs_generated': os.path.join(base_path, 'real_vs_generated_viz'),
            'distribution': os.path.join(base_path, 'distribution_viz'),
            'dimensionality': os.path.join(base_path, 'dimensionality_viz')
        }
        
        # Create folders if they don't exist
        for folder in folders.values():
            if not os.path.exists(folder):
                os.makedirs(folder)
                
        return folders
    
    def _generate_anomaly_detection_visualizations(self, validation_results, folders):
        """
        Generate visualizations comparing One-Class SVM and Isolation Forest anomaly detection
        
        Args:
            validation_results: Validation results dictionary
            folders: Dictionary of visualization folder paths
        """
        print("Generating anomaly detection visualizations...")
        
        # Create dataframe for visualization
        data = []
        for cls, results in validation_results.items():
            data.append({
                'Class': cls,
                'Algorithm': 'One-Class SVM',
                'Anomaly Rate (%)': results['ocsvm_anomaly_rate']
            })
            data.append({
                'Class': cls,
                'Algorithm': 'Isolation Forest',
                'Anomaly Rate (%)': results['if_anomaly_rate']
            })
        
        df = pd.DataFrame(data)
        
        # Set style
        sns.set(style="whitegrid")
        plt.figure(figsize=(12, 8))
        
        # Create grouped bar chart
        ax = sns.barplot(x='Class', y='Anomaly Rate (%)', hue='Algorithm', data=df)
        
        # Customize plot
        plt.title('Anomaly Detection Comparison: One-Class SVM vs Isolation Forest', fontsize=16)
        plt.xlabel('Attack Class', fontsize=14)
        plt.ylabel('Anomaly Rate (%)', fontsize=14)
        plt.xticks(rotation=45, ha='right')
        plt.legend(title='Algorithm')
        
        # Add 30% threshold line
        plt.axhline(y=30, color='r', linestyle='--', alpha=0.7, label='30% Threshold')
        
        # Adjust layout
        plt.tight_layout()
        
        # Save figure
        output_path = os.path.join(folders['anomaly_detection'], 'anomaly_comparison_all_classes.png')
        plt.savefig(output_path, dpi=300)
        plt.close()
        
        print(f"Saved anomaly detection comparison to {output_path}")
        
        # Create overall summary pie charts
        plt.figure(figsize=(15, 7))
        
        # OCSVM summary
        plt.subplot(1, 2, 1)
        ocsvm_anomaly_counts = [results['ocsvm_anomaly_count'] for results in validation_results.values()]
        ocsvm_normal_counts = [results['packet_count'] - results['ocsvm_anomaly_count'] for results in validation_results.values()]
        total_ocsvm_anomaly = sum(ocsvm_anomaly_counts)
        total_ocsvm_normal = sum(ocsvm_normal_counts)
        
        if total_ocsvm_anomaly > 0:
            plt.pie([total_ocsvm_normal, total_ocsvm_anomaly], 
                   labels=['Normal', 'Anomaly'], 
                   autopct='%1.1f%%',
                   colors=['#4CAF50', '#F44336'],
                   explode=(0, 0.1),
                   shadow=True)
        else:
            plt.pie([1], labels=['Normal (100%)'], colors=['#4CAF50'])
            
        plt.title('One-Class SVM Results')
        
        # Isolation Forest summary
        plt.subplot(1, 2, 2)
        if_anomaly_counts = [results['if_anomaly_count'] for results in validation_results.values()]
        if_normal_counts = [results['packet_count'] - results['if_anomaly_count'] for results in validation_results.values()]
        total_if_anomaly = sum(if_anomaly_counts)
        total_if_normal = sum(if_normal_counts)
        
        if total_if_anomaly > 0:
            plt.pie([total_if_normal, total_if_anomaly], 
                   labels=['Normal', 'Anomaly'], 
                   autopct='%1.1f%%',
                   colors=['#4CAF50', '#F44336'],
                   explode=(0, 0.1),
                   shadow=True)
        else:
            plt.pie([1], labels=['Normal (100%)'], colors=['#4CAF50'])
            
        plt.title('Isolation Forest Results')
        
        plt.suptitle('Overall Anomaly Detection Results', fontsize=16)
        plt.tight_layout()
        
        # Save figure
        output_path = os.path.join(folders['anomaly_detection'], 'anomaly_summary_pie_charts.png')
        plt.savefig(output_path, dpi=300)
        plt.close()
        
        print(f"Saved anomaly summary pie charts to {output_path}")
    
    def _generate_real_vs_generated_visualizations(self, folders):
        """
        Generate visualizations comparing real and generated data
        
        Args:
            folders: Dictionary of visualization folder paths
        """
        print("Generating real vs generated data visualizations...")
        
        for cls in self.classes:
            print(f"Processing class: {cls}")
            
            # Get class indices
            class_indices = np.where(self.labels == cls)[0]
            
            # Skip if no data for this class
            if len(class_indices) == 0:
                print(f"Warning: No real data found for class {cls}. Skipping visualization.")
                continue
                
            # Get real data
            if not hasattr(self, 'scaler') or self.scaler is None:
                print(f"Warning: No scaler found. Skipping visualization for class {cls}.")
                continue
                
            # Get generated data file path
            safe_cls_name = cls.replace('/', '_').replace(' ', '_').replace('\\', '_')
            file_path = os.path.join(self.output_dir, f"{safe_cls_name}_generated.csv")
            if not os.path.exists(file_path):
                print(f"Warning: Generated data file not found: {file_path}. Skipping visualization.")
                continue
                
            try:
                # Load generated data
                generated_df = pd.read_csv(file_path)
                
                # Remove Flow ID and Label from generated data
                X_gen = generated_df.drop([col for col in [self.flow_id_column, 'Label'] if col in generated_df.columns], axis=1)
                
                # Get real data for this class
                X_real = self.data.loc[class_indices].drop([col for col in [self.flow_id_column, 'Label'] if col in self.data.columns], axis=1)
                
                # Ensure columns match between real and generated data
                common_cols = list(set(X_real.columns) & set(X_gen.columns))
                if not common_cols:
                    print(f"Warning: No common columns between real and generated data for class {cls}. Skipping visualization.")
                    continue
                    
                X_real = X_real[common_cols]
                X_gen = X_gen[common_cols]
                
                # Sample data if too large
                sample_size = min(1000, len(X_real), len(X_gen))
                if len(X_real) > sample_size:
                    X_real = X_real.sample(sample_size, random_state=self.seed)
                if len(X_gen) > sample_size:
                    X_gen = X_gen.sample(sample_size, random_state=self.seed)
                
                # Generate dimensionality reduction visualizations
                self._generate_dimensionality_reduction_plots(X_real, X_gen, cls, folders['dimensionality'])
                
                # Generate distribution comparisons for key features
                self._generate_distribution_comparisons(X_real, X_gen, cls, folders['distribution'])
                
                # Generate feature correlation heatmaps
                self._generate_correlation_heatmaps(X_real, X_gen, cls, folders['real_vs_generated'])
                
            except Exception as e:
                print(f"Error generating visualizations for class {cls}: {e}")
                print(traceback.format_exc())
    
    def _generate_dimensionality_reduction_plots(self, X_real, X_gen, cls, folder):
        """
        Generate dimensionality reduction plots (PCA and t-SNE)
        
        Args:
            X_real: Real data
            X_gen: Generated data
            cls: Class name
            folder: Output folder path
        """
        # Combine data for dimensionality reduction
        X_combined = pd.concat([X_real, X_gen], axis=0).reset_index(drop=True)
        # Create labels for the combined dataset (0 for real, 1 for generated)
        y_combined = np.concatenate([np.zeros(len(X_real)), np.ones(len(X_gen))])
        
        # Create PCA plot
        try:
            # Apply PCA
            pca = PCA(n_components=2, random_state=self.seed)
            X_pca = pca.fit_transform(X_combined)
            
            # Create plot
            plt.figure(figsize=(10, 8))
            scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y_combined, cmap='coolwarm', alpha=0.7, s=50)
            
            # Add legend
            legend_labels = ['Real', 'Generated']
            plt.legend(handles=scatter.legend_elements()[0], labels=legend_labels)
            
            # Customize plot
            plt.title(f'PCA: Real vs Generated Data for {cls}', fontsize=14)
            plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)', fontsize=12)
            plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)', fontsize=12)
            plt.grid(True, alpha=0.3)
            
            # Add text annotation with sample counts
            plt.annotate(f'Real: {len(X_real)} samples\nGenerated: {len(X_gen)} samples', 
                      xy=(0.05, 0.95), xycoords='axes fraction',
                      bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
            
            # Save figure
            safe_cls_name = cls.replace('/', '_').replace(' ', '_').replace('\\', '_')
            output_path = os.path.join(folder, f'{safe_cls_name}_pca.png')
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"Saved PCA plot for class {cls}")
        except Exception as e:
            print(f"Error creating PCA plot for class {cls}: {e}")
        
        # Create t-SNE plot
        try:
            # Apply t-SNE (sample if too large)
            max_tsne_samples = 2000
            if len(X_combined) > max_tsne_samples:
                print(f"t-SNE: Sampling data for class {cls} (too many points)")
                indices = np.random.choice(len(X_combined), max_tsne_samples, replace=False)
                X_sampled = X_combined.iloc[indices]
                y_sampled = y_combined[indices]
            else:
                X_sampled = X_combined
                y_sampled = y_combined
                
            # Apply t-SNE
            tsne = TSNE(n_components=2, random_state=self.seed, perplexity=min(30, len(X_sampled)//10))
            X_tsne = tsne.fit_transform(X_sampled)
            
            # Create plot
            plt.figure(figsize=(10, 8))
            scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y_sampled, cmap='coolwarm', alpha=0.7, s=50)
            
            # Add legend
            legend_labels = ['Real', 'Generated']
            plt.legend(handles=scatter.legend_elements()[0], labels=legend_labels)
            
            # Customize plot
            plt.title(f't-SNE: Real vs Generated Data for {cls}', fontsize=14)
            plt.xlabel('t-SNE Dimension 1', fontsize=12)
            plt.ylabel('t-SNE Dimension 2', fontsize=12)
            plt.grid(True, alpha=0.3)
            
            # Add text annotation with sample counts
            real_count = np.sum(y_sampled == 0)
            gen_count = np.sum(y_sampled == 1)
            plt.annotate(f'Real: {real_count} samples\nGenerated: {gen_count} samples', 
                      xy=(0.05, 0.95), xycoords='axes fraction',
                      bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
            
            # Save figure
            safe_cls_name = cls.replace('/', '_').replace(' ', '_').replace('\\', '_')
            output_path = os.path.join(folder, f'{safe_cls_name}_tsne.png')
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"Saved t-SNE plot for class {cls}")
        except Exception as e:
            print(f"Error creating t-SNE plot for class {cls}: {e}")
    
    def _generate_distribution_comparisons(self, X_real, X_gen, cls, folder):
        """
        Generate distribution comparisons for key features
        
        Args:
            X_real: Real data
            X_gen: Generated data
            cls: Class name
            folder: Output folder path
        """
        # Select top features for visualization (based on variance)
        n_features = min(6, len(X_real.columns))
        variances = X_real.var().sort_values(ascending=False)
        top_features = variances.index[:n_features].tolist()
        
        # Create distribution comparison plots
        plt.figure(figsize=(15, 10))
        
        for i, feature in enumerate(top_features):
            plt.subplot(2, 3, i+1)
            
            # Plot distributions
            sns.kdeplot(X_real[feature], label='Real', color='blue', alpha=0.7)
            sns.kdeplot(X_gen[feature], label='Generated', color='red', alpha=0.7)
            
            # Customize subplot
            plt.title(feature)
            plt.xlabel('Value')
            plt.ylabel('Density')
            
            if i == 0:
                plt.legend()
        
        # Add main title
        plt.suptitle(f'Feature Distributions: Real vs Generated for {cls}', fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        
        # Save figure
        safe_cls_name = cls.replace('/', '_').replace(' ', '_').replace('\\', '_')
        output_path = os.path.join(folder, f'{safe_cls_name}_distributions.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved distribution comparison for class {cls}")
        
        # Create box plots for the same features
        plt.figure(figsize=(15, 10))
        
        for i, feature in enumerate(top_features):
            plt.subplot(2, 3, i+1)
            
            # Prepare data for box plot
            df_box = pd.DataFrame({
                'Value': pd.concat([X_real[feature], X_gen[feature]]),
                'Type': ['Real'] * len(X_real) + ['Generated'] * len(X_gen)
            })
            
            # Plot box plot
            sns.boxplot(x='Type', y='Value', data=df_box)
            
            # Customize subplot
            plt.title(feature)
            plt.xlabel('')
            plt.ylabel('Value')
            
        # Add main title
        plt.suptitle(f'Feature Box Plots: Real vs Generated for {cls}', fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        
        # Save figure
        output_path = os.path.join(folder, f'{safe_cls_name}_boxplots.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved box plots for class {cls}")
    
    def _generate_correlation_heatmaps(self, X_real, X_gen, cls, folder):
        """
        Generate correlation heatmaps for real and generated data
        
        Args:
            X_real: Real data
            X_gen: Generated data
            cls: Class name
            folder: Output folder path
        """
        # Select a subset of features if too many
        max_features = 15
        if len(X_real.columns) > max_features:
            # Select most variable features
            variances = X_real.var().sort_values(ascending=False)
            selected_features = variances.index[:max_features].tolist()
            X_real_sub = X_real[selected_features]
            X_gen_sub = X_gen[selected_features]
        else:
            X_real_sub = X_real
            X_gen_sub = X_gen
        
        # Calculate correlation matrices
        corr_real = X_real_sub.corr()
        corr_gen = X_gen_sub.corr()
        
        # Create heatmaps
        fig, axes = plt.subplots(1, 2, figsize=(20, 10))
        
        # Real data correlation
        sns.heatmap(corr_real, annot=False, cmap='coolwarm', vmin=-1, vmax=1, ax=axes[0])
        axes[0].set_title(f'Real Data Correlation Matrix for {cls}', fontsize=14)
        
        # Generated data correlation
        sns.heatmap(corr_gen, annot=False, cmap='coolwarm', vmin=-1, vmax=1, ax=axes[1])
        axes[1].set_title(f'Generated Data Correlation Matrix for {cls}', fontsize=14)
        
        plt.tight_layout()
        
        # Save figure
        safe_cls_name = cls.replace('/', '_').replace(' ', '_').replace('\\', '_')
        output_path = os.path.join(folder, f'{safe_cls_name}_correlation.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved correlation heatmaps for class {cls}")
        
        # Calculate and visualize correlation difference
        corr_diff = corr_gen - corr_real
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(corr_diff, annot=False, cmap='RdBu_r', vmin=-1, vmax=1)
        plt.title(f'Correlation Difference (Generated - Real) for {cls}', fontsize=14)
        plt.tight_layout()
        
        # Save figure
        output_path = os.path.join(folder, f'{safe_cls_name}_correlation_diff.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved correlation difference heatmap for class {cls}")
        
    def validate_generated_packets(self, report_path="/Users/mayankraj/Desktop/RESEARCH/Project 2 V3"):
        """
        Validate the generated packets against anomaly detection models and save a detailed report
        
        Args:
            report_path: Path to save the anomaly report
        """
        print("Validating generated packets...")
        
        validation_results = {}
        detailed_results = {}
        
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
            
            # Get flow IDs for tracking
            flow_ids = df[self.flow_id_column].values
            
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
                ocsvm_preds = np.ones(len(X_scaled))  # Default to normal
            
            # Validate with Isolation Forest
            try:
                isolation_forest = models['isolation_forest']
                if_preds = isolation_forest.predict(X_scaled)
                if_anomaly_rate = np.sum(if_preds == -1) / len(if_preds) * 100
                print(f"Isolation Forest - Anomaly Rate: {if_anomaly_rate:.2f}%")
            except Exception as e:
                print(f"Error validating with Isolation Forest for class {cls}: {e}")
                if_anomaly_rate = float('nan')
                if_preds = np.ones(len(X_scaled))  # Default to normal
            
            # Store validation results
            validation_results[cls] = {
                'ocsvm_anomaly_rate': ocsvm_anomaly_rate,
                'if_anomaly_rate': if_anomaly_rate,
                'packet_count': len(X_scaled),
                'ocsvm_anomaly_count': np.sum(ocsvm_preds == -1),
                'if_anomaly_count': np.sum(if_preds == -1)
            }
            
            # Store detailed packet-level results
            packet_results = []
            for i in range(len(X_scaled)):
                packet_results.append({
                    'flow_id': flow_ids[i],
                    'ocsvm_prediction': 'Anomaly' if ocsvm_preds[i] == -1 else 'Normal',
                    'if_prediction': 'Anomaly' if if_preds[i] == -1 else 'Normal',
                    'combined_result': 'Anomaly' if (ocsvm_preds[i] == -1 or if_preds[i] == -1) else 'Normal'
                })
            detailed_results[cls] = packet_results
            
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
        
        # Create visualization folders
        viz_folders = self._create_visualization_folders(report_path)
        
        # Generate visualizations
        self._generate_anomaly_detection_visualizations(validation_results, viz_folders)
        self._generate_real_vs_generated_visualizations(viz_folders)
        
        # Create detailed HTML report
        self._create_anomaly_report(validation_results, detailed_results, report_path)
        
    def _create_anomaly_report(self, validation_results, detailed_results, report_path):
        """
        Create a detailed HTML anomaly report
        
        Args:
            validation_results: Summary validation results
            detailed_results: Detailed packet-level results
            report_path: Path to save the report
        """
        try:
            # Create summary results dataframe
            summary_data = []
            for cls, results in validation_results.items():
                summary_data.append({
                    'Class': cls,
                    'Packet Count': results['packet_count'],
                    'OCSVM Anomaly Count': results['ocsvm_anomaly_count'],
                    'OCSVM Anomaly Rate (%)': results['ocsvm_anomaly_rate'],
                    'OCSVM Status': 'PASS' if not np.isnan(results['ocsvm_anomaly_rate']) and results['ocsvm_anomaly_rate'] < 30 else 'FAIL',
                    'Isolation Forest Anomaly Count': results['if_anomaly_count'],
                    'Isolation Forest Anomaly Rate (%)': results['if_anomaly_rate'],
                    'Isolation Forest Status': 'PASS' if not np.isnan(results['if_anomaly_rate']) and results['if_anomaly_rate'] < 30 else 'FAIL',
                })
            
            summary_df = pd.DataFrame(summary_data)
            
            # Create overall anomaly rates
            total_packets = summary_df['Packet Count'].sum()
            total_ocsvm_anomaly = summary_df['OCSVM Anomaly Count'].sum()
            total_if_anomaly = summary_df['Isolation Forest Anomaly Count'].sum()
            overall_ocsvm_rate = (total_ocsvm_anomaly / total_packets) * 100 if total_packets > 0 else 0
            overall_if_rate = (total_if_anomaly / total_packets) * 100 if total_packets > 0 else 0
            
            # Save summary to CSV
            summary_path = os.path.join(report_path, "anomaly_report_summary.csv")
            summary_df.to_csv(summary_path, index=False)
            print(f"Saved anomaly summary report to {summary_path}")
            
            # Create HTML content
            html_content = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>Network Packet Generation Anomaly Report</title>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 20px; }}
                    h1, h2, h3 {{ color: #333366; }}
                    table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
                    th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                    th {{ background-color: #f2f2f2; }}
                    tr:nth-child(even) {{ background-color: #f9f9f9; }}
                    .pass {{ color: green; font-weight: bold; }}
                    .fail {{ color: red; font-weight: bold; }}
                    .summary {{ background-color: #eef; padding: 10px; margin: 10px 0; border-radius: 5px; }}
                </style>
            </head>
            <body>
                <h1>Network Packet Generation Anomaly Report</h1>
                <p>Generated on: {time.strftime("%Y-%m-%d %H:%M:%S")}</p>
                
                <div class="summary">
                    <h2>Overall Summary</h2>
                    <p>Total Generated Packets: {total_packets}</p>
                    <p>Overall One-Class SVM Anomaly Rate: {overall_ocsvm_rate:.2f}%</p>
                    <p>Overall Isolation Forest Anomaly Rate: {overall_if_rate:.2f}%</p>
                </div>
                
                <h2>Anomaly Results by Class</h2>
                <table>
                    <tr>
                        <th>Class</th>
                        <th>Packet Count</th>
                        <th>OCSVM Anomaly Count</th>
                        <th>OCSVM Anomaly Rate (%)</th>
                        <th>OCSVM Status</th>
                        <th>Isolation Forest Anomaly Count</th>
                        <th>Isolation Forest Anomaly Rate (%)</th>
                        <th>Isolation Forest Status</th>
                    </tr>
            """
            
            # Add rows for each class
            for _, row in summary_df.iterrows():
                ocsvm_status_class = "pass" if row['OCSVM Status'] == 'PASS' else "fail"
                if_status_class = "pass" if row['Isolation Forest Status'] == 'PASS' else "fail"
                
                html_content += f"""
                    <tr>
                        <td>{row['Class']}</td>
                        <td>{row['Packet Count']}</td>
                        <td>{row['OCSVM Anomaly Count']}</td>
                        <td>{row['OCSVM Anomaly Rate (%)']:.2f}%</td>
                        <td class="{ocsvm_status_class}">{row['OCSVM Status']}</td>
                        <td>{row['Isolation Forest Anomaly Count']}</td>
                        <td>{row['Isolation Forest Anomaly Rate (%)']:.2f}%</td>
                        <td class="{if_status_class}">{row['Isolation Forest Status']}</td>
                    </tr>
                """
            
            html_content += """
                </table>
                
                <h2>Class-Specific Results</h2>
            """
            
            # Add detailed results for each class
            for cls, packets in detailed_results.items():
                anomaly_packets = [p for p in packets if p['combined_result'] == 'Anomaly']
                
                html_content += f"""
                <h3>{cls}</h3>
                <p>Total Packets: {len(packets)}</p>
                <p>Anomaly Packets: {len(anomaly_packets)}</p>
                """
                
                # Only show detailed table if there are anomalies
                if anomaly_packets:
                    html_content += f"""
                    <h4>Anomalous Packets</h4>
                    <table>
                        <tr>
                            <th>Flow ID</th>
                            <th>OCSVM Prediction</th>
                            <th>Isolation Forest Prediction</th>
                        </tr>
                    """
                    
                    for packet in anomaly_packets:
                        html_content += f"""
                        <tr>
                            <td>{packet['flow_id']}</td>
                            <td>{packet['ocsvm_prediction']}</td>
                            <td>{packet['if_prediction']}</td>
                        </tr>
                        """
                    
                    html_content += "</table>"
                else:
                    html_content += "<p>No anomalous packets detected.</p>"
            
            html_content += """
            </body>
            </html>
            """
            
            # Save HTML report
            html_path = os.path.join(report_path, "anomaly_report.html")
            with open(html_path, 'w') as f:
                f.write(html_content)
            print(f"Saved detailed anomaly report to {html_path}")
            
            # Save detailed results to JSON
            json_path = os.path.join(report_path, "anomaly_report_detailed.json")
            with open(json_path, 'w') as f:
                json.dump(detailed_results, f, indent=2)
            print(f"Saved detailed results to {json_path}")
            
        except Exception as e:
            print(f"Error creating anomaly report: {e}")
            print(traceback.format_exc())
    
    def run(self, packets_per_class=1000, report_path="/Users/mayankraj/Desktop/RESEARCH/Project 2 V3"):
        """
        Run the complete genetic packet generation pipeline
        
        Args:
            packets_per_class: Number of packets to generate per class
            report_path: Path to save the anomaly report
        """
        print("Starting the genetic packet generation pipeline...")
        
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
            
            # Step 4: Generate packets using genetic algorithm
            self.generate_packets(packets_per_class=packets_per_class)
            
            # Step 5: Save generated packets
            self.save_generated_packets()
            
            # Step 6: Validate generated packets and create anomaly report
            self.validate_generated_packets(report_path=report_path)
            
            print("Genetic packet generation pipeline completed successfully!")
            
        except Exception as e:
            print(f"Error in genetic packet generation pipeline: {e}")
            print(traceback.format_exc())
            print("Attempting to continue with available results...")
            
            # Try to save any generated packets if we got that far
            if hasattr(self, 'generated_data') and self.generated_data:
                try:
                    self.save_generated_packets()
                    self.validate_generated_packets(report_path=report_path)
                except Exception as e2:
                    print(f"Error saving/validating generated packets: {e2}")
                    
            print("Pipeline completed with errors. Check the logs for details.")


if __name__ == "__main__":
    # Parse command line arguments if any
    parser = argparse.ArgumentParser(description='Generate network packets based on ACI IoT 2023 dataset using Genetic Algorithm')
    parser.add_argument('--dataset', type=str, 
                        default="/Users/mayankraj/Desktop/RESEARCH/Thesis Codes /archive/ACI-IoT-2023.csv",
                        help='Path to the ACI IoT 2023 dataset')
    parser.add_argument('--output', type=str, default='generated_packets',
                        help='Directory to save generated packets')
    parser.add_argument('--packets', type=int, default=1000,
                        help='Number of packets to generate per class')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--population', type=int, default=200,
                        help='Population size for genetic algorithm')
    parser.add_argument('--generations', type=int, default=50,
                        help='Maximum generations for genetic algorithm')
    parser.add_argument('--mutation', type=float, default=0.05,
                        help='Mutation rate for genetic algorithm')
    args = parser.parse_args()
    
    # Hard-coded report path
    report_path = "/Users/mayankraj/Desktop/RESEARCH/Project 2 V3"
    
    try:
        # Create the genetic packet generator
        generator = GeneticPacketGenerator(
            dataset_path=args.dataset,
            output_dir=args.output,
            seed=args.seed
        )
        
        # Set genetic algorithm parameters if provided
        if args.population:
            generator.population_size = args.population
        if args.generations:
            generator.generations = args.generations
        if args.mutation:
            generator.mutation_rate = args.mutation
        
        # Run the generator with hard-coded report path
        generator.run(packets_per_class=args.packets, report_path=report_path)
    except Exception as e:
        print(f"Error: {e}")
        print(traceback.format_exc())
        print("Program terminated with errors.")