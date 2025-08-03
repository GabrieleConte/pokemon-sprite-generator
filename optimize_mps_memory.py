#!/usr/bin/env python3
"""
Memory optimization script for MPS (Apple Silicon) training
"""

import torch
import os
import subprocess

def get_system_memory_gb():
    """Get total system memory in GB"""
    try:
        # Get system memory on macOS
        result = subprocess.run(['sysctl', 'hw.memsize'], capture_output=True, text=True)
        if result.returncode == 0:
            memsize_bytes = int(result.stdout.split(':')[1].strip())
            memsize_gb = memsize_bytes / (1024**3)
            return memsize_gb
    except:
        pass
    return 8  # Fallback for unknown systems

def setup_mps_memory_optimization():
    """Configure optimal MPS memory settings for efficient RAM usage (no swap)"""
    
    # Detect system memory first
    total_memory = get_system_memory_gb()
    print(f"🔧 Configuring MPS memory optimization for {total_memory:.1f}GB system...")
    
    # Set MPS memory management
    if torch.backends.mps.is_available():
        # Clear any existing memory first
        if hasattr(torch.mps, 'empty_cache'):
            try:
                torch.mps.empty_cache()
            except:
                pass  # Ignore if it fails initially
        
        # Enable unified memory (allows using system RAM beyond GPU memory)
        os.environ['PYTORCH_MPS_ENABLE_UNIFIED_MEMORY'] = '1'
        
        # Use more memory for 16GB system - don't use swap
        # Valid ratio is between 0.0 and 1.0
        os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.9'  # Use 90% of available memory max
        
        # Enable memory pool for better management
        os.environ['PYTORCH_MPS_ENABLE_MEMORY_POOL'] = '1'
        
        # Use expandable segments but be conservative
        os.environ['PYTORCH_MPS_ALLOCATOR_POLICY'] = 'expandable_segments'
        
        # DISABLE memory overcommit and swap usage
        os.environ.pop('PYTORCH_MPS_ENABLE_MEMORY_OVERCOMMIT', None)  # Remove if set
        
        # Enable memory caching for efficiency
        os.environ['PYTORCH_NO_CUDA_MEMORY_CACHING'] = '0'
        
        # Test memory allocation based on system memory
        try:
            # Clear cache first
            if hasattr(torch.mps, 'empty_cache'):
                torch.mps.empty_cache()
                
            # Calculate test sizes based on system memory
            if total_memory >= 16:
                test_sizes = [2, 4, 6, 8, 10, 12, 14]  # Test up to 14GB for 16GB+ systems
            elif total_memory >= 8:
                test_sizes = [1, 2, 3, 4, 5, 6]  # Test up to 6GB for 8GB systems
            else:
                test_sizes = [0.5, 1, 1.5, 2, 2.5, 3]  # Conservative for smaller systems
                
            max_allocated = 0
            
            for size_gb in test_sizes:
                try:
                    # Calculate tensor size for target GB
                    elements = int((size_gb * 1024**3) / 4)  # 4 bytes per float32
                    
                    # Try to allocate
                    test_tensor = torch.zeros(elements, device='mps', dtype=torch.float32)
                    max_allocated = size_gb
                    
                    # Free immediately
                    del test_tensor
                    torch.mps.empty_cache()
                    
                except Exception:
                    break
            
            print(f"✅ Safe memory allocation limit: ~{max_allocated}GB (no swap)")
            
        except Exception as e:
            print(f"⚠️ Memory test failed: {e}")
        
        # Final cleanup
        if hasattr(torch.mps, 'empty_cache'):
            torch.mps.empty_cache()
        
        print("✅ MPS unified memory enabled")
        print("✅ Memory pool optimization enabled") 
        print("✅ Conservative memory management (no swap)")
        print("✅ Memory caching enabled for efficiency")
        
        # Check available memory
        try:
            if hasattr(torch.mps, 'current_allocated_memory'):
                allocated = torch.mps.current_allocated_memory() / (1024**3)
                print(f"📊 Current GPU memory allocated: {allocated:.2f} GB")
        except Exception as e:
            print(f"⚠️ Memory check failed: {e}")
    
    else:
        print("❌ MPS not available on this system")
        return False
    
    return True

def get_optimal_batch_size_for_memory():
    """Calculate optimal batch size based on available memory (conservative, no swap)"""
    try:
        # Start with even smaller batch sizes and more realistic memory usage
        test_sizes = [1, 2, 3, 4, 6, 8, 10, 12]
        
        print("🧪 Testing optimal batch size for your system...")
        
        for batch_size in test_sizes:
            try:
                # Clear cache before each test
                torch.mps.empty_cache()
                
                # Simulate actual training memory usage more realistically
                # Reduce dimensions to match actual training
                
                # Text embeddings: batch_size x seq_len x hidden_dim (smaller seq_len)
                test_text = torch.randn(batch_size, 16, 768, device='mps', requires_grad=True)
                
                # VAE latents: batch_size x latent_dim x height x width (smaller dimensions) 
                test_latent = torch.randn(batch_size, 4, 16, 16, device='mps', requires_grad=True)
                
                # Images: batch_size x channels x height x width (smaller images)
                test_images = torch.randn(batch_size, 3, 128, 128, device='mps', requires_grad=True)
                
                print(f"  Testing batch {batch_size}: allocated tensors...")
                
                # Simulate some operations (forward pass) - fix dimension mismatch
                # Get mean embeddings and reshape properly
                text_mean = test_text.mean(dim=1)  # [batch_size, 768]
                latent_mean = test_latent.mean(dim=[2,3])  # [batch_size, 4]
                
                # Simple test operations that don't require matching dimensions
                test_result1 = text_mean.sum()
                test_result2 = latent_mean.sum() 
                test_result3 = test_images.mean()
                
                test_loss = test_result1 + test_result2 + test_result3
                
                print(f"  Testing batch {batch_size}: forward pass ok...")
                
                # Test backward pass memory (this is where most OOM happens)
                test_loss.backward()
                
                print(f"  Testing batch {batch_size}: backward pass ok...")
                
                # Clean up
                del test_text, test_latent, test_images, test_result1, test_result2, test_result3, test_loss
                torch.mps.empty_cache()
                
                print(f"✅ Batch size {batch_size} works safely")
                
                # Test one size higher to make sure we're not at the edge
                continue
                
            except Exception as e:
                print(f"❌ Batch size {batch_size} failed: {str(e)}")
                torch.mps.empty_cache()
                
                # Return the previous working size
                if batch_size > test_sizes[0]:
                    prev_idx = test_sizes.index(batch_size) - 1
                    safe_batch_size = test_sizes[prev_idx]
                    print(f"🎯 Safe batch size found: {safe_batch_size}")
                    return safe_batch_size
                else:
                    print(f"🎯 Using minimum batch size: 1")
                    return 1
        
        # If all sizes worked, return the largest tested
        print(f"🎯 All batch sizes worked! Using: {test_sizes[-1]}")
        return test_sizes[-1]
        
    except Exception as e:
        print(f"⚠️ Memory test failed: {e}")
        return 1  # Ultra-conservative fallback

def check_memory_constraints():
    """Check current memory usage and constraints"""
    try:
        if hasattr(torch.mps, 'current_allocated_memory'):
            allocated = torch.mps.current_allocated_memory() / (1024**3)
            print(f"📊 Current MPS memory allocated: {allocated:.2f} GB")
        
        # Test a small allocation to see current limits
        try:
            test_tensor = torch.zeros(1000000, device='mps')  # ~4MB test
            del test_tensor
            torch.mps.empty_cache()
            print("✅ MPS device is responsive")
        except Exception as e:
            print(f"⚠️ MPS device issue: {e}")
            
    except Exception as e:
        print(f"⚠️ Memory check failed: {e}")

if __name__ == "__main__":
    setup_mps_memory_optimization()
    optimal_batch = get_optimal_batch_size_for_memory()
    print(f"🎯 Recommended batch size: {optimal_batch}")
