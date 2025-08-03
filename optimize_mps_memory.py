#!/usr/bin/env python3
"""
Memory optimization script for MPS (Apple Silicon) training
"""

import torch
import os

def setup_mps_memory_optimization():
    """Configure optimal MPS memory settings for 16GB RAM systems"""
    
    print("🔧 Configuring MPS memory optimization for 16GB system...")
    
    # Set MPS memory management
    if torch.backends.mps.is_available():
        # Enable unified memory (allows using system RAM)
        os.environ['PYTORCH_MPS_ENABLE_UNIFIED_MEMORY'] = '1'
        
        # Set memory fraction to use more of available memory
        # This allows MPS to use more than the default 8GB
        os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0'  # Use as much as needed
        
        # Enable memory management optimizations
        os.environ['PYTORCH_MPS_ENABLE_MEMORY_POOL'] = '1'
        
        # Reduce memory fragmentation
        torch.mps.empty_cache()
        
        print("✅ MPS unified memory enabled")
        print("✅ Memory pool optimization enabled")
        print("✅ High watermark ratio set to use more system memory")
        
        # Check available memory
        if hasattr(torch.mps, 'current_allocated_memory'):
            allocated = torch.mps.current_allocated_memory() / (1024**3)
            print(f"📊 Current GPU memory allocated: {allocated:.2f} GB")
    
    else:
        print("❌ MPS not available on this system")
        return False
    
    return True

def get_optimal_batch_size_for_memory():
    """Calculate optimal batch size based on available memory"""
    try:
        # Test allocation to find optimal batch size
        test_sizes = [16, 12, 8, 6, 4]
        
        for batch_size in test_sizes:
            try:
                # Test tensor allocation similar to our training data
                test_tensor = torch.randn(batch_size, 8, 27, 27, device='mps')
                test_text = torch.randn(batch_size, 36, 256, device='mps')
                
                # Test some operations
                result = test_tensor * 2.0 + test_text.mean(dim=1, keepdim=True).unsqueeze(-1)
                
                # Clean up
                del test_tensor, test_text, result
                torch.mps.empty_cache()
                
                print(f"✅ Batch size {batch_size} works without memory issues")
                return batch_size
                
            except Exception as e:
                print(f"❌ Batch size {batch_size} failed: {str(e)[:50]}...")
                torch.mps.empty_cache()
                continue
        
        return 4  # Conservative fallback
        
    except Exception as e:
        print(f"⚠️ Memory test failed: {e}")
        return 4

if __name__ == "__main__":
    setup_mps_memory_optimization()
    optimal_batch = get_optimal_batch_size_for_memory()
    print(f"🎯 Recommended batch size: {optimal_batch}")
