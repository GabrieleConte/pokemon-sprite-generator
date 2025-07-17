import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
import logging

class PokemonDataset(Dataset):
    """
    PyTorch Dataset class for Pokemon sprite generation.
    
    This dataset loads Pokemon images and their descriptions from CSV file.
    Images are preprocessed and normalized for training.
    """
    
    def __init__(self, 
                 csv_path: str,
                 image_dir: str,
                 image_size: int = 215,
                 transform: Optional[transforms.Compose] = None,
                 augment: bool = True,
                 filter_missing: bool = True):
        """
        Initialize the Pokemon dataset.
        
        Args:
            csv_path: Path to the pokemon.csv file
            image_dir: Path to the directory containing PNG images
            image_size: Target size for images (default: 215)
            transform: Custom transform pipeline (if None, uses default)
            augment: Whether to apply data augmentation
            filter_missing: Whether to filter out missing images
        """
        self.csv_path = csv_path
        self.image_dir = image_dir
        self.image_size = image_size
        self.augment = augment
        
        # Load CSV data
        try:
            self.data = pd.read_csv(csv_path, sep='\t', encoding='utf-8')
        except UnicodeDecodeError:
            try:
                # Try with UTF-16 encoding (common for Windows files)
                self.data = pd.read_csv(csv_path, sep='\t', encoding='utf-16')
            except:
                # Try with latin-1 encoding
                self.data = pd.read_csv(csv_path, sep='\t', encoding='latin-1')
        
        # Check if we have the expected columns
        required_columns = ['national_number', 'english_name', 'description']
        missing_columns = [col for col in required_columns if col not in self.data.columns]
        
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}. Available columns: {list(self.data.columns)}")
        
        # Filter out Pokemon without descriptions
        self.data = self.data.dropna(subset=['description'])
        
        # Filter out entries with missing images if requested
        if filter_missing:
            self.data = self._filter_missing_images()
        
        # Setup transforms
        self.transform = transform if transform is not None else self._get_default_transform()
        
        # Setup augmentation transforms
        self.augment_transform = self._get_augmentation_transform() if augment else None
        
        logging.info(f"Loaded {len(self.data)} Pokemon samples from {csv_path}")
        
    def _filter_missing_images(self) -> pd.DataFrame:
        """Filter out entries where the corresponding image file doesn't exist."""
        existing_data = []
        missing_count = 0
        
        for _, row in self.data.iterrows():
            image_path = self._get_image_path(row['national_number'])
            if os.path.exists(image_path):
                existing_data.append(row)
            else:
                missing_count += 1
                
        if missing_count > 0:
            logging.warning(f"Filtered out {missing_count} entries with missing images")
            
        return pd.DataFrame(existing_data)
    
    def _get_image_path(self, national_number: int) -> str:
        """Get the path to the image file for a given national number."""
        filename = f"{national_number:03d}.png"
        return os.path.join(self.image_dir, filename)
    
    def _get_default_transform(self) -> transforms.Compose:
        """Get the default image preprocessing pipeline."""
        return transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize to [-1, 1]
        ])
    
    def _get_augmentation_transform(self) -> transforms.Compose:
        """Get data augmentation pipeline for training."""
        return transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
            transforms.RandomResizedCrop(size=(self.image_size, self.image_size), 
                                       scale=(0.9, 1.0), ratio=(0.9, 1.1)),
        ])
    
    def __len__(self) -> int:
        """Return the total number of samples in the dataset."""
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, Union[torch.Tensor, str, int]]:
        """
        Get a single sample from the dataset.
        
        Args:
            idx: Index of the sample to retrieve
            
        Returns:
            Dictionary containing:
            - 'image': Preprocessed image tensor [3, H, W]
            - 'description': Text description
            - 'national_number': Pokemon national number
            - 'name': Pokemon name
            - 'primary_type': Primary Pokemon type
            - 'secondary_type': Secondary Pokemon type (can be empty)
        """
        if torch.is_tensor(idx):
            idx = int(idx.item())  # Convert tensor to int
            
        # Get row data
        row = self.data.iloc[idx]
        
        # Load and preprocess image
        image_path = self._get_image_path(row['national_number'])
        image = Image.open(image_path).convert('RGB')
        
        # Apply augmentation if training
        if self.augment and self.augment_transform is not None:
            image = self.augment_transform(image)
            
        # Apply standard transforms
        image = self.transform(image)
        
        # Prepare description text
        description = self._clean_description(row['description'])
        
        # Create comprehensive text description
        full_description = self._create_full_description(row)
        
        return {
            'image': image,
            'description': description,
            'full_description': full_description,
            'national_number': int(row['national_number']),
            'name': str(row['english_name']),
            'primary_type': str(row['primary_type']),
            'secondary_type': str(row['secondary_type']) if pd.notna(row['secondary_type']) else '',
        }
    
    def _clean_description(self, description: str) -> str:
        """Clean and normalize description text."""
        if pd.isna(description):
            return ""
        
        # Remove extra quotes and clean whitespace
        description = str(description).strip()
        if description.startswith('"') and description.endswith('"'):
            description = description[1:-1]
        
        return description
    
    def _create_full_description(self, row: pd.Series) -> str:
        """Create a comprehensive description including type and other info."""
        parts = []
        
        # Add type information
        if pd.notna(row['primary_type']):
            type_info = f"A {row['primary_type']} type"
            if pd.notna(row['secondary_type']):
                type_info += f" and {row['secondary_type']} type"
            type_info += " pokemon"
            parts.append(type_info)
        
        # Add name
        parts.append(f"named {row['english_name']}")
        
        # Add classification if available
        if pd.notna(row['classification']):
            parts.append(f"classified as {row['classification']}")
        
        # Add original description
        description = self._clean_description(row['description'])
        if description:
            parts.append(description)
        
        return ". ".join(parts) + "."

def create_data_loaders(csv_path: str,
                       image_dir: str,
                       batch_size: int = 32,
                       val_split: float = 0.1,
                       test_split: float = 0.1,
                       image_size: int = 215,
                       num_workers: int = 4,
                       pin_memory: bool = True,
                       seed: int = 42) -> Dict[str, DataLoader]:
    """
    Create train, validation, and test data loaders.
    
    Args:
        csv_path: Path to the pokemon.csv file
        image_dir: Path to the directory containing PNG images
        batch_size: Batch size for training
        val_split: Fraction of data to use for validation
        test_split: Fraction of data to use for testing
        image_size: Target image size
        num_workers: Number of workers for data loading
        pin_memory: Whether to pin memory for faster GPU transfer
        seed: Random seed for reproducible splits
        
    Returns:
        Dictionary with 'train', 'val', and 'test' DataLoaders
    """
    # Set random seed for reproducible splits
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Create full dataset
    full_dataset = PokemonDataset(
        csv_path=csv_path,
        image_dir=image_dir,
        image_size=image_size,
        augment=False  # We'll handle augmentation separately
    )
    
    # Calculate split sizes
    total_size = len(full_dataset)
    test_size = int(total_size * test_split)
    val_size = int(total_size * val_split)
    train_size = total_size - val_size - test_size
    
    # Create splits
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        full_dataset, 
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(seed)
    )
    
    # Create augmented training dataset
    train_augmented_dataset = PokemonDataset(
        csv_path=csv_path,
        image_dir=image_dir,
        image_size=image_size,
        augment=True
    )
    
    # Apply the same indices to the augmented dataset
    train_indices = list(train_dataset.indices)
    train_augmented_dataset.data = train_augmented_dataset.data.iloc[train_indices]
    
    # Create data loaders
    train_loader = DataLoader(
        train_augmented_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False
    )
    
    logging.info(f"Created data loaders: train={len(train_loader)} batches, "
                f"val={len(val_loader)} batches, test={len(test_loader)} batches")
    
    return train_loader, val_loader, test_loader

def get_dataset_statistics(csv_path: str, image_dir: str) -> Dict[str, Any]:
    """
    Get statistics about the dataset.
    
    Args:
        csv_path: Path to the pokemon.csv file
        image_dir: Path to the directory containing PNG images
        
    Returns:
        Dictionary with dataset statistics
    """
    dataset = PokemonDataset(csv_path, image_dir, augment=False)
    
    # Basic stats
    stats = {
        'total_samples': len(dataset),
        'image_dir': image_dir,
        'csv_path': csv_path,
        'missing_images': 0,
        'missing_descriptions': 0
    }
    
    # Type distribution
    type_counts = {}
    description_lengths = []
    
    for i in range(min(len(dataset), 100)):  # Sample for efficiency
        sample = dataset[i]
        primary_type = sample['primary_type']
        secondary_type = sample['secondary_type']
        
        type_counts[primary_type] = type_counts.get(primary_type, 0) + 1
        if secondary_type:
            type_counts[secondary_type] = type_counts.get(secondary_type, 0) + 1
            
        description_lengths.append(len(str(sample['description']).split()))
    
    stats['type_distribution'] = dict(sorted(type_counts.items(), key=lambda x: x[1], reverse=True))
    stats['avg_description_length'] = np.mean(description_lengths)
    stats['description_length_std'] = np.std(description_lengths)
    
    return stats

if __name__ == "__main__":
    # Test the dataset
    csv_path = "/Users/gabrieleconte/Developer/pokemon-sprite-generator/data/pokemon.csv"
    image_dir = "/Users/gabrieleconte/Developer/pokemon-sprite-generator/data/small_images"
    
    # Get dataset statistics
    stats = get_dataset_statistics(csv_path, image_dir)
    print("Dataset Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # Create data loaders
    data_loaders = create_data_loaders(csv_path, image_dir, batch_size=4)
    
    # Test a batch
    train_loader = data_loaders['train']
    batch = next(iter(train_loader))
    
    print(f"\nBatch shapes:")
    print(f"  Images: {batch['image'].shape}")
    print(f"  Descriptions: {len(batch['description'])}")
    print(f"  Sample description: {batch['description'][0]}")
