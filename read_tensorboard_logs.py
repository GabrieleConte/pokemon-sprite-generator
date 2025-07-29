#!/usr/bin/env python3
"""
Script to read and analyze tensorboard logs for diagnosing loss explosion.
"""

import os
import struct
import numpy as np
from pathlib import Path

def read_tensorboard_event_file(file_path):
    """
    Read tensorboard event file and extract scalar data.
    """
    print(f"Reading tensorboard file: {file_path}")
    
    try:
        # Try using tensorboard's EventAccumulator if available
        try:
            from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
            
            ea = EventAccumulator(file_path)
            ea.Reload()
            
            print("\nAvailable scalar tags:")
            scalar_tags = ea.Tags()['scalars']
            for tag in scalar_tags:
                print(f"  - {tag}")
            
            # Extract training and validation losses with correct tag names
            results = {}
            tag_mapping = {
                'Diffusion Train/Loss': 'train/loss',
                'Diffusion Val/Loss': 'val/loss', 
                'Diffusion Train/Gradient_Norm': 'train/grad_norm',
                'Diffusion Train/Learning_Rate': 'learning_rate'
            }
            
            for original_tag, mapped_tag in tag_mapping.items():
                if original_tag in scalar_tags:
                    scalars = ea.Scalars(original_tag)
                    results[mapped_tag] = [(s.step, s.value) for s in scalars]
                    print(f"\nFound {len(results[mapped_tag])} entries for {original_tag}")
            
            return results
            
        except ImportError:
            print("TensorBoard not available, trying TensorFlow...")
            
        # Try using TensorFlow
        try:
            import tensorflow as tf
            
            results = {
                'train/loss': [],
                'val/loss': [],
                'train/grad_norm': [],
                'learning_rate': []
            }
            
            tag_mapping = {
                'Diffusion Train/Loss': 'train/loss',
                'Diffusion Val/Loss': 'val/loss', 
                'Diffusion Train/Gradient_Norm': 'train/grad_norm',
                'Diffusion Train/Learning_Rate': 'learning_rate'
            }
            
            for record in tf.data.TFRecordDataset(file_path):
                event = tf.compat.v1.Event()
                event.ParseFromString(record.numpy())
                
                if event.HasField('summary'):
                    for value in event.summary.value:
                        if value.tag in tag_mapping:
                            mapped_tag = tag_mapping[value.tag]
                            results[mapped_tag].append((event.step, value.simple_value))
            
            print("\nExtracted data using TensorFlow:")
            for tag, data in results.items():
                if data:
                    print(f"  {tag}: {len(data)} entries")
            
            return results
            
        except ImportError:
            print("TensorFlow not available, using raw file analysis...")
            
        # Fallback: raw file inspection
        with open(file_path, 'rb') as f:
            data = f.read()
        
        print(f"\nFile size: {len(data)} bytes")
        
        # Look for common patterns
        data_str = str(data)
        patterns = ['train/loss', 'val/loss', 'grad_norm', 'learning_rate']
        
        print("\nFound patterns:")
        for pattern in patterns:
            if pattern in data_str:
                print(f"  ✓ {pattern}")
            else:
                print(f"  ✗ {pattern}")
        
        return None
        
    except Exception as e:
        print(f"Error reading file: {e}")
        return None

def analyze_loss_explosion(results):
    """
    Analyze the training data to identify loss explosion patterns.
    """
    if not results:
        print("No data to analyze")
        return
    
    print("\n" + "="*60)
    print("LOSS EXPLOSION ANALYSIS")
    print("="*60)
    
    # Analyze training loss
    if 'train/loss' in results and results['train/loss']:
        train_data = results['train/loss']
        epochs, losses = zip(*train_data)
        
        print(f"\nTraining Loss Analysis:")
        print(f"  Total epochs: {max(epochs)}")
        print(f"  Data points: {len(losses)}")
        
        # Find explosion point
        losses_array = np.array(losses)
        
        # Look for sudden increases
        if len(losses) > 1:
            loss_ratios = losses_array[1:] / losses_array[:-1]
            explosion_indices = np.where(loss_ratios > 2.0)[0]  # 2x increase
            
            if len(explosion_indices) > 0:
                explosion_epoch = epochs[explosion_indices[0] + 1]
                print(f"\n🔥 LOSS EXPLOSION DETECTED at epoch {explosion_epoch}")
                print(f"  Loss jumped from {losses[explosion_indices[0]]:.6f} to {losses[explosion_indices[0] + 1]:.6f}")
                print(f"  Ratio: {loss_ratios[explosion_indices[0]]:.2f}x")
        
        # Show progression around epoch 50
        print(f"\nLoss progression around epoch 50:")
        explosion_found = False
        for epoch, loss in train_data:
            if 45 <= epoch <= 55:
                status = "🔥" if loss > 1.0 else "✓"
                print(f"  {status} Epoch {epoch:3d}: {loss:.6f}")
        
        # Show progression around the actual explosion point
        explosion_step = None
        for i, (epoch, loss) in enumerate(train_data[:-1]):
            next_loss = train_data[i+1][1]
            if next_loss / loss > 2.0:
                explosion_step = epoch
                break
        
        if explosion_step:
            print(f"\nDetailed progression around explosion (step {explosion_step}):")
            for epoch, loss in train_data:
                if explosion_step - 200 <= epoch <= explosion_step + 200:
                    status = "🔥" if loss > 0.5 else "✓"
                    if abs(epoch - explosion_step) <= 10:
                        status = "💥" if loss > 0.5 else "⚠️"
                    print(f"  {status} Step {epoch:5d}: {loss:.6f}")
        
        # Statistical analysis
        min_loss = min(losses)
        max_loss = max(losses)
        mean_loss = np.mean(losses)
        std_loss = np.std(losses)
        
        print(f"\nStatistics:")
        print(f"  Min loss: {min_loss:.6f}")
        print(f"  Max loss: {max_loss:.6f}")
        print(f"  Mean loss: {mean_loss:.6f}")
        print(f"  Std loss: {std_loss:.6f}")
        print(f"  Max/Min ratio: {max_loss/min_loss:.2f}x")
    
    # Analyze gradient norms
    if 'train/grad_norm' in results and results['train/grad_norm']:
        grad_data = results['train/grad_norm']
        epochs, grad_norms = zip(*grad_data)
        
        print(f"\nGradient Norm Analysis:")
        grad_norms_array = np.array(grad_norms)
        
        # Look for gradient explosion
        high_grad_indices = np.where(grad_norms_array > 1.0)[0]
        if len(high_grad_indices) > 0:
            print(f"  🚨 High gradients detected at {len(high_grad_indices)} epochs")
            for i in high_grad_indices[:10]:  # Show first 10
                print(f"    Epoch {epochs[i]}: grad_norm = {grad_norms[i]:.6f}")
        
        print(f"\nGradient progression around epoch 50:")
        for epoch, grad_norm in grad_data:
            if 45 <= epoch <= 55:
                status = "🚨" if grad_norm > 1.0 else "✓"
                print(f"  {status} Epoch {epoch:3d}: {grad_norm:.6f}")

def diagnose_potential_causes():
    """
    Print potential causes and solutions for loss explosion.
    """
    print("\n" + "="*60)
    print("POTENTIAL CAUSES & SOLUTIONS")
    print("="*60)
    
    print("""
🔍 Common causes of loss explosion in diffusion models:

1. LEARNING RATE TOO HIGH
   - Current config uses lr=0.0005 for U-Net
   - Try reducing to 0.0001 or 0.00005
   
2. GRADIENT EXPLOSION
   - Check if grad_norm consistently > 1.0
   - Current grad clipping: 0.7 (might be too high)
   - Try reducing to 0.5 or 0.3
   
3. DTYPE MISMATCHES (MPS/Apple Silicon)
   - float64 tensors cause issues on MPS
   - Ensure all tensors are float32
   
4. NOISE SCHEDULE ISSUES
   - Cosine schedule might be too aggressive
   - Try linear schedule or lower beta_end
   
5. U-NET ARCHITECTURE INSTABILITIES
   - Very large model (640M parameters)
   - Cross-attention might be unstable
   - GroupNorm with wrong group sizes
   
6. BATCH SIZE TOO SMALL
   - Current: batch_size=2
   - Very small batches can cause instability
   - Try batch_size=4 or 8 if memory allows

🛠️ Recommended fixes:
   1. Lower learning rate to 0.0001
   2. Reduce gradient clipping to 0.5
   3. Fix dtype issues in noise scheduler
   4. Add better error handling in attention blocks
   5. Increase batch size if possible
""")

def main():
    # Find the tensorboard file
    log_dir = Path("logs/v6")
    event_files = list(log_dir.glob("events.out.tfevents.*"))
    
    if not event_files:
        print("No tensorboard event files found in logs/v6/")
        return
    
    event_file = event_files[0]
    print(f"Analyzing tensorboard logs from: {event_file}")
    
    # Read and analyze the data
    results = read_tensorboard_event_file(str(event_file))
    
    if results:
        analyze_loss_explosion(results)
    
    # Always show potential causes
    diagnose_potential_causes()

if __name__ == "__main__":
    main()
