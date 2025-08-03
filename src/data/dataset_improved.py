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
    PyTorch Dataset class for Pokemon sprite generation with proper transparency handling.
    
    This dataset loads Pokemon images and their descriptions from CSV file.
    Images are preprocessed with proper background handling for transparent PNGs.
    """
    
    def __init__(self, 
                 csv_path: str,
                 image_dir: str,
                 image_size: int = 215,
                 transform: Optional[transforms.Compose] = None,
                 augment: bool = True,
                 filter_missing: bool = True,
                 background_color: Union[str, Tuple[int, int, int]] = 'white'):
        """
        Initialize the Pokemon dataset.
        
        Args:
            csv_path: Path to the pokemon.csv file
            image_dir: Path to the directory containing PNG images
            image_size: Target size for images (default: 215)
            transform: Custom transform pipeline (if None, uses default)
            augment: Whether to apply data augmentation
            filter_missing: Whether to filter out missing images
            background_color: Background color for transparent areas. 
                            Can be 'white', 'black', or RGB tuple like (255, 255, 255)
        """
        self.csv_path = csv_path
        self.image_dir = image_dir
        self.image_size = image_size
        self.augment = augment
        self.background_color = self._parse_background_color(background_color)
        
        try:
            self.data = pd.read_csv(csv_path, sep=';', encoding='utf-8', header=None)
            if len(self.data.columns) == 2:
                self.data.columns = ['english_name', 'description']
                self.data['national_number'] = range(1, len(self.data) + 1)
                self.data = self.data[['national_number', 'english_name', 'description']]
            else:
                # Fall back to tab-separated
                self.data = pd.read_csv(csv_path, sep='\t', encoding='utf-8')
        except UnicodeDecodeError:
            try:
                self.data = pd.read_csv(csv_path, sep=';', encoding='utf-16', header=None)
                if len(self.data.columns) == 2:
                    self.data.columns = ['english_name', 'description']
                    self.data['national_number'] = range(1, len(self.data) + 1)
                    self.data = self.data[['national_number', 'english_name', 'description']]
                else:
                    self.data = pd.read_csv(csv_path, sep='\t', encoding='utf-16')
            except UnicodeDecodeError:
                try:
                    self.data = pd.read_csv(csv_path, sep='\t', encoding='utf-16')
                except:
                    self.data = pd.read_csv(csv_path, sep='\t', encoding='latin-1')
        
        required_columns = ['national_number', 'english_name', 'description']
        missing_columns = [col for col in required_columns if col not in self.data.columns]
        
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}. Available columns: {list(self.data.columns)}")
        
        self.data = self.data.dropna(subset=['description'])
        if filter_missing:
            self.data = self._filter_missing_images()
        self.transform = transform if transform is not None else self._get_default_transform()
        
        self.augment_transform = self._get_augmentation_transform() if augment else None
        
        logging.info(f"Loaded {len(self.data)} Pokemon samples from {csv_path}")
        logging.info(f"Using background color: {self.background_color}")
        
    def _parse_background_color(self, background_color: Union[str, Tuple[int, int, int]]) -> Tuple[int, int, int]:
        """Parse background color specification."""
        if isinstance(background_color, str):
            if background_color.lower() == 'white':
                return (255, 255, 255)
            elif background_color.lower() == 'black':
                return (0, 0, 0)
            elif background_color.lower() == 'gray' or background_color.lower() == 'grey':
                return (128, 128, 128)
            else:
                raise ValueError(f"Unknown background color: {background_color}")
        elif isinstance(background_color, (tuple, list)) and len(background_color) == 3:
            return (int(background_color[0]), int(background_color[1]), int(background_color[2]))
        else:
            raise ValueError(f"Invalid background color format: {background_color}")
        
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
    
    def _load_image_with_background(self, image_path: str) -> Image.Image:
        """Load PNG image and properly handle transparency with consistent background."""
        original_img = Image.open(image_path)
        
        if original_img.mode in ('RGBA', 'LA') or (original_img.mode == 'P' and 'transparency' in original_img.info):
            background = Image.new('RGB', original_img.size, self.background_color)
            
            if original_img.mode == 'RGBA':
                background.paste(original_img, mask=original_img.split()[-1])  # Use alpha channel as mask
            elif original_img.mode == 'LA':
                background.paste(original_img, mask=original_img.split()[-1])  # Use alpha channel as mask
            elif original_img.mode == 'P':
                background.paste(original_img, mask=original_img.convert('RGBA').split()[-1])
            
            return background
        else:
            return original_img.convert('RGB')
    
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
            
        row = self.data.iloc[idx]
        
        image_path = self._get_image_path(row['national_number'])
        image = self._load_image_with_background(image_path)
        
        if self.augment and self.augment_transform is not None:
            image = self.augment_transform(image)

        image = self.transform(image)
        
        description = self._clean_description(row['description'])
        
        full_description = self._create_full_description(row)
        
        return {
            'image': image,
            'description': description,
            'full_description': full_description,
            'national_number': int(row['national_number']),
            'name': str(row['english_name']),
        }
    
    def _clean_description(self, description: str) -> str:
        """Clean and normalize description text."""
        if pd.isna(description):
            return ""
        
        description = str(description).strip()
        if description.startswith('"') and description.endswith('"'):
            description = description[1:-1]
        
        return description
    
    def _create_full_description(self, row: pd.Series) -> str:
        """Create a comprehensive description including type and other info."""
        parts = []
        
        parts.append(f"Pokemon named {row['english_name']}")
        
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
                       seed: int = 42,
                       background_color: Union[str, Tuple[int, int, int]] = 'white') -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, validation, and test data loaders with proper background handling.
    
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
        background_color: Background color for transparent areas        Returns:
            Tuple with (train_loader, val_loader, test_loader)
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    full_dataset = PokemonDataset(
        csv_path=csv_path,
        image_dir=image_dir,
        image_size=image_size,
        augment=False,  # We'll handle augmentation separately
        background_color=background_color
    )
    
    total_size = len(full_dataset)
    test_size = int(total_size * test_split)
    val_size = int(total_size * val_split)
    train_size = total_size - val_size - test_size
    
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        full_dataset, 
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(seed)
    )
    
    train_augmented_dataset = PokemonDataset(
        csv_path=csv_path,
        image_dir=image_dir,
        image_size=image_size,
        augment=True,
        background_color=background_color
    )
    
    train_indices = list(train_dataset.indices)
    train_augmented_dataset.data = train_augmented_dataset.data.iloc[train_indices]
    
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
    
    logging.info(f"Created data loaders with background color {background_color}: train={len(train_loader)} batches, "
                f"val={len(val_loader)} batches, test={len(test_loader)} batches")
    
    return train_loader, val_loader, test_loader

def get_dataset_statistics(csv_path: str, image_dir: str, background_color: Union[str, Tuple[int, int, int]] = 'white') -> Dict[str, Any]:
    """
    Get statistics about the dataset.
    
    Args:
        csv_path: Path to the pokemon.csv file
        image_dir: Path to the directory containing PNG images
        background_color: Background color for transparent areas
        
    Returns:
        Dictionary with dataset statistics
    """
    dataset = PokemonDataset(csv_path, image_dir, augment=False, background_color=background_color)
    
    stats = {
        'total_samples': len(dataset),
        'image_dir': image_dir,
        'csv_path': csv_path,
        'background_color': background_color,
        'missing_images': 0,
        'missing_descriptions': 0
    }
    
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
    csv_path = "/Users/gabrieleconte/Developer/pokemon-sprite-generator/data/text_description_concat.csv"
    image_dir = "/Users/gabrieleconte/Developer/pokemon-sprite-generator/data/small_images"
    
    # Test different background colors
    for bg_color in ['white', 'black', (128, 128, 128)]:
        print(f"\n{'='*50}")
        print(f"Testing with background color: {bg_color}")
        print(f"{'='*50}")
        
        # Get dataset statistics
        # stats = get_dataset_statistics(csv_path, image_dir, background_color=bg_color)
        # print("Dataset Statistics:")
        # for key, value in stats.items():
        #     print(f"  {key}: {value}")
        
        # Create data loaders
        train_loader, val_loader, test_loader = create_data_loaders(
            csv_path, image_dir, batch_size=4, background_color=bg_color
        )
        
        # Test a batch
        batch = next(iter(train_loader))
        print(f"\nBatch shapes:")
        print(f"  Images: {batch['image'].shape}")
        print(f"  Descriptions: {len(batch['description'])}")
        print(f"  Sample description: {batch['description'][0]}")
        print(f" Full description: {batch['full_description'][0]}")
        
        # Check image statistics
        images = batch['image']
        print(f"  Image tensor stats: min={images.min():.3f}, max={images.max():.3f}, mean={images.mean():.3f}")
