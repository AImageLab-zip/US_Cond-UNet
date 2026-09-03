import random
import numpy as np
from collections import defaultdict
from torch.utils.data import Sampler
from typing import Iterator, List
from pathlib import Path


class BalancedHierarchicalSampler(Sampler):
    """
    Hierarchical sampler that balances:
    1. Primary: Organ representation (using inverse sqrt frequency weighting)
    2. Secondary: Dataset representation within each organ
    
    Supports oversampling via replacement for small datasets/organs.
    """
    
    def __init__(
        self,
        dataset,
        batch_size: int,
        steps_per_epoch: int = None,
        seed: int = None,
    ):
        """
        Args:
            dataset: USdatasetOmni instance or Subset of USdatasetOmni
            batch_size: Number of samples per batch
            steps_per_epoch: Number of batches per epoch. If None, computed as len(dataset) // batch_size
            seed: Random seed for reproducibility (optional)
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.seed = seed
        
        # Handle Subset wrapper
        from torch.utils.data import Subset, ConcatDataset
        if isinstance(dataset, Subset):
            self.base_dataset = dataset.dataset
            self.subset_indices = dataset.indices
            self.is_subset = True
        elif isinstance(dataset, ConcatDataset):
            # For ConcatDataset, we'll need to handle it differently
            # We'll treat it as accessing items directly
            self.base_dataset = dataset
            self.subset_indices = None
            self.is_subset = False
        else:
            self.base_dataset = dataset
            self.subset_indices = None
            self.is_subset = False
        
        # Compute steps per epoch
        if steps_per_epoch is None:
            self.steps_per_epoch = max(1, len(dataset) // batch_size)
        else:
            self.steps_per_epoch = steps_per_epoch
            
        # Build hierarchical structure
        self._build_hierarchy()
        
        # Compute organ sampling weights (inverse sqrt frequency)
        self._compute_organ_weights()
        
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
    
    def _build_hierarchy(self):
        """
        Build the mapping structure:
        organ -> list of datasets -> list of sample indices
        
        Handles both regular datasets and Subset wrappers.
        """
        self.organ_to_datasets = defaultdict(set)
        self.dataset_to_indices = defaultdict(list)
        self.organ_to_indices = defaultdict(list)
        
        # Determine which items to iterate over
        if self.is_subset:
            # For Subset, only iterate over the subset indices
            items_to_process = [
                (subset_idx, self.base_dataset.items[original_idx]) 
                for subset_idx, original_idx in enumerate(self.subset_indices)
            ]
        else:
            # For regular dataset or ConcatDataset
            if hasattr(self.base_dataset, 'items'):
                items_to_process = [
                    (idx, item) 
                    for idx, item in enumerate(self.base_dataset.items)
                ]
            else:
                # ConcatDataset case - need to access through __getitem__
                # This is less efficient but necessary for ConcatDataset
                print("Warning: Dataset doesn't have 'items' attribute. Using __getitem__ (slower).")
                items_to_process = []
                for idx in range(len(self.base_dataset)):
                    try:
                        # Get the item to extract metadata
                        sample = self.base_dataset[idx]
                        # We need to reconstruct item dict from the sample
                        # This assumes your dataset returns a dict with these keys
                        item = {
                            'organ_label': sample.get('organ_id_metric', -1),
                            'image_path': f"concat_dataset_idx_{idx}"  # Fallback path
                        }
                        items_to_process.append((idx, item))
                    except Exception as e:
                        print(f"Error processing index {idx}: {e}")
                        continue
        
        for idx, item in items_to_process:
            # Extract dataset name from image path
            image_path = Path(item['image_path'])
            dataset_name = image_path.parent.parent.name
            
            # Handle potential organ_label retrieval
            if isinstance(item.get('organ_label'), int):
                # If organ_label is already an integer (class index), convert to name
                from utils.utils import class_to_organ_dict
                organ_label = class_to_organ_dict.get(item['organ_label'], 'unknown')
            else:
                organ_label = item['organ_label']
            
            # Build mappings - use idx which is relative to the dataset/subset
            self.organ_to_datasets[organ_label].add(dataset_name)
            self.dataset_to_indices[dataset_name].append(idx)
            self.organ_to_indices[organ_label].append(idx)
        
        # Convert sets to lists for sampling
        self.organ_to_datasets = {
            organ: list(datasets) 
            for organ, datasets in self.organ_to_datasets.items()
        }
        
        # Filter out organs with no samples (skip automatically)
        self.available_organs = [
            organ for organ in self.organ_to_datasets.keys()
            if len(self.organ_to_indices[organ]) > 0
        ]
        
        print(f"\n{'='*60}")
        print(f"BalancedHierarchicalSampler Initialization")
        if self.is_subset:
            print(f"Dataset type: Subset (original size: {len(self.base_dataset.items)}, subset size: {len(self.subset_indices)})")
        else:
            print(f"Dataset type: {'ConcatDataset' if not hasattr(self.base_dataset, 'items') else 'Regular'}")
        print(f"{'='*60}")
        print(f"Total samples: {len(self.dataset)}")
        print(f"Batch size: {self.batch_size}")
        print(f"Steps per epoch: {self.steps_per_epoch}")
        print(f"Available organs: {len(self.available_organs)}")
        print(f"\nOrgan distribution:")
        for organ in sorted(self.available_organs):
            num_samples = len(self.organ_to_indices[organ])
            num_datasets = len(self.organ_to_datasets[organ])
            print(f"  {organ:20s}: {num_samples:5d} samples across {num_datasets} dataset(s)")
            for dataset_name in sorted(self.organ_to_datasets[organ]):
                ds_samples = len(self.dataset_to_indices[dataset_name])
                print(f"    └─ {dataset_name:30s}: {ds_samples:5d} samples")
        print(f"{'='*60}\n")
    
    def _compute_organ_weights(self):
        """
        Compute sampling weights using inverse sqrt frequency (moderate balancing).
        This oversamples rare organs without being too aggressive.
        """
        organ_counts = {
            organ: len(self.organ_to_indices[organ])
            for organ in self.available_organs
        }
        
        # Inverse sqrt frequency
        weights = {
            organ: 1.0 / np.sqrt(count)
            for organ, count in organ_counts.items()
        }
        
        # Normalize weights to sum to 1
        total_weight = sum(weights.values())
        self.organ_weights = {
            organ: w / total_weight
            for organ, w in weights.items()
        }
        
        print("Organ sampling weights (normalized):")
        for organ in sorted(self.available_organs):
            print(f"  {organ:20s}: {self.organ_weights[organ]:.4f}")
        print()
    
    def _sample_batch_indices(self) -> List[int]:
        """
        Sample a single batch of indices with hierarchical balancing.
        
        Returns:
            List of sample indices for one batch
        """
        batch_indices = []
        
        # Step 1: Sample organs for this batch using weighted sampling
        organs_in_batch = np.random.choice(
            self.available_organs,
            size=self.batch_size,
            replace=True,
            p=[self.organ_weights[org] for org in self.available_organs]
        )
        
        # Step 2: For each selected organ, sample dataset then sample index
        for organ in organs_in_batch:
            # Uniformly sample a dataset within this organ
            available_datasets = self.organ_to_datasets[organ]
            selected_dataset = random.choice(available_datasets)
            
            # Uniformly sample an index from the selected dataset (with replacement)
            available_indices = self.dataset_to_indices[selected_dataset]
            selected_idx = random.choice(available_indices)
            
            batch_indices.append(selected_idx)
        
        return batch_indices
    
    def __iter__(self) -> Iterator[List[int]]:
        """
        Generate batches for one epoch.
        Each batch is independently sampled to ensure balanced representation.
        """
        for _ in range(self.steps_per_epoch):
            batch_indices = self._sample_batch_indices()
            yield batch_indices
    
    def __len__(self) -> int:
        """Return number of batches per epoch"""
        return self.steps_per_epoch

