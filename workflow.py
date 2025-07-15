#!/usr/bin/env python3
"""
Complete workflow script for Pokemon Sprite Generator
Usage: python workflow.py [command]
Commands:
  - train: Start training the model
  - test: Test the trained model
  - generate: Generate Pokemon sprites
  - interactive: Start interactive generation
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path

def run_command(cmd, description):
    """Run a command with proper error handling"""
    print(f"\n🚀 {description}")
    print(f"Running: {cmd}")
    print("-" * 50)
    
    try:
        result = subprocess.run(cmd, shell=True, check=True, capture_output=False)
        print(f"✅ {description} completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed with error code {e.returncode}")
        return False

def check_file_exists(filepath, description):
    """Check if a file exists and print status"""
    if os.path.exists(filepath):
        print(f"✅ {description}: {filepath}")
        return True
    else:
        print(f"❌ {description}: {filepath} (missing)")
        return False

def main():
    parser = argparse.ArgumentParser(description='Pokemon Sprite Generator Workflow')
    parser.add_argument('command', choices=['train', 'test', 'generate', 'interactive', 'setup'], 
                       help='Command to run')
    parser.add_argument('--checkpoint', default='checkpoints/best_model.pth',
                       help='Path to model checkpoint')
    parser.add_argument('--config', default='config/train_config.yaml',
                       help='Path to training config')
    
    args = parser.parse_args()
    
    # Print header
    print("=" * 60)
    print("🎮 POKEMON SPRITE GENERATOR WORKFLOW 🎮")
    print("=" * 60)
    
    if args.command == 'setup':
        print("\n📋 CHECKING SETUP...")
        
        # Check Python files
        files_to_check = [
            ('train.py', 'Training script'),
            ('tests/test_model.py', 'Model testing script'),
            ('tests/test_generation.py', 'Generation script'),
            ('tests/interactive_generation.py', 'Interactive script'),
            ('config/train_config.yaml', 'Training configuration'),
            ('data/pokemon.csv', 'Pokemon dataset'),
        ]
        
        print("\n📁 File Status:")
        all_files_exist = True
        for filepath, description in files_to_check:
            if not check_file_exists(filepath, description):
                all_files_exist = False
        
        # Check data directory
        data_dir = Path('data/small_images')
        if data_dir.exists():
            num_images = len(list(data_dir.glob('*.png')))
            print(f"✅ Pokemon images: {num_images} found in {data_dir}")
        else:
            print(f"❌ Pokemon images directory missing: {data_dir}")
            all_files_exist = False
        
        if all_files_exist:
            print("\n🎉 Setup complete! You can now:")
            print("  python workflow.py train      # Start training")
            print("  python workflow.py test       # Test model architecture")
            print("  python workflow.py generate   # Generate Pokemon (after training)")
            print("  python workflow.py interactive # Interactive generation")
        else:
            print("\n⚠️  Some files are missing. Please check the setup.")
    
    elif args.command == 'train':
        print(f"\n🎯 STARTING TRAINING...")
        
        if not check_file_exists(args.config, "Training config"):
            return
        
        if not check_file_exists('data/pokemon.csv', "Pokemon dataset"):
            return
            
        # Create checkpoints directory
        os.makedirs('checkpoints', exist_ok=True)
        print("✅ Created checkpoints directory")
        
        # Start training
        cmd = f"python3 train.py --config {args.config}"
        if run_command(cmd, "Training Pokemon Sprite Generator"):
            print(f"\n🎉 Training completed! Check 'checkpoints/' for saved models.")
            print(f"Next steps:")
            print(f"  python workflow.py test --checkpoint {args.checkpoint}")
            print(f"  python workflow.py generate --checkpoint {args.checkpoint}")
    
    elif args.command == 'test':
        print(f"\n🧪 TESTING MODEL...")
        
        if os.path.exists(args.checkpoint):
            cmd = f"python3 tests/test_model.py {args.checkpoint}"
            description = f"Testing trained model: {args.checkpoint}"
        else:
            cmd = "python3 tests/test_model.py"
            description = "Testing model architecture (without checkpoint)"
        
        if run_command(cmd, description):
            print(f"\n🎉 Model test completed successfully!")
            if os.path.exists(args.checkpoint):
                print(f"Ready for generation:")
                print(f"  python workflow.py generate --checkpoint {args.checkpoint}")
    
    elif args.command == 'generate':
        print(f"\n🎨 GENERATING POKEMON SPRITES...")
        
        if not check_file_exists(args.checkpoint, "Model checkpoint"):
            print("❌ Please train the model first:")
            print("  python workflow.py train")
            return
        
        # Create output directory
        os.makedirs('generated_samples', exist_ok=True)
        
        cmd = f"python3 tests/test_generation.py --checkpoint {args.checkpoint}"
        if run_command(cmd, f"Generating Pokemon sprites from {args.checkpoint}"):
            print(f"\n🎉 Generation completed!")
            print(f"Check 'generated_samples/' for your new Pokemon!")
            print(f"\nTry more generation options:")
            print(f"  python tests/test_generation.py --checkpoint {args.checkpoint} --description 'A red fire dragon Pokemon'")
            print(f"  python tests/test_generation.py --checkpoint {args.checkpoint} --variations 6")
    
    elif args.command == 'interactive':
        print(f"\n🎮 STARTING INTERACTIVE GENERATION...")
        
        cmd = "python3 tests/interactive_generation.py"
        run_command(cmd, "Interactive Pokemon Generation Session")
    
    print("\n" + "=" * 60)
    print("🎮 Pokemon Sprite Generator Workflow Complete!")
    print("=" * 60)

if __name__ == "__main__":
    main()
