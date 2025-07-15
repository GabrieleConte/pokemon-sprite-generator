# Testing Your Trained Pokemon Sprite Generator

After training your model, you can test it in several ways. Here's a complete guide:

## 1. Quick Model Test (Without Checkpoint)

Test that your model architecture works correctly:

```bash
python3 tests/test_model.py
```

## 2. Test With Trained Checkpoint

Test your trained model with a checkpoint file:

```bash
python3 tests/test_model.py checkpoints/best_model.pth
```

## 3. Generate Single Images

### Generate with default descriptions:

```bash
python3 tests/test_generation.py --checkpoint checkpoints/best_model.pth
```

### Generate with custom description:

```bash
python3 tests/test_generation.py --checkpoint checkpoints/best_model.pth --description "A small yellow electric mouse Pokemon with red cheeks"
```

### Generate multiple variations:

```bash
python3 tests/test_generation.py --checkpoint checkpoints/best_model.pth --description "A blue dragon Pokemon with crystal wings" --variations 4
```

## 4. Batch Generation

### Generate from text file:

```bash
python3 tests/test_generation.py --checkpoint checkpoints/best_model.pth --descriptions-file tests/test_descriptions.txt
```

### Generate with custom output directory:

```bash
python3 tests/test_generation.py --checkpoint checkpoints/best_model.pth --output-dir my_pokemon_output
```

## 5. Interactive Generation

For a fun interactive session:

```bash
python3 tests/interactive_generation.py
```

Then enter your checkpoint path and start generating Pokemon interactively!

## 6. Expected Outputs

Your model will generate:
- **215x215 pixel images** (PNG format)
- **RGB color images** of Pokemon sprites
- **Multiple variations** for the same description (due to random noise)

## 7. Output Structure

Generated images will be saved in the following structure:
```
generated_samples/
├── generated_001.png
├── generated_002.png
├── ...
└── batch_grid.png          # Grid of all generated images
```

For variations:
```
generated_samples/
├── variation_1.png
├── variation_2.png
├── ...
└── variations_grid.png     # Grid of all variations
```

## 8. Troubleshooting

### Common Issues:

1. **FileNotFoundError**: Make sure your checkpoint path is correct
2. **CUDA/MPS errors**: The script will automatically fallback to CPU
3. **Memory issues**: Reduce batch size or use CPU device

### Memory Usage:
- **GPU**: ~2-4GB VRAM for inference
- **CPU**: ~1-2GB RAM for inference

## 9. Example Descriptions

Try these interesting descriptions:

```
A small yellow electric mouse Pokemon with red cheeks and a lightning bolt tail
A large blue dragon Pokemon with crystal wings and ice breath
A green plant Pokemon with flower petals and vine attacks
A purple ghost Pokemon with glowing eyes and shadow abilities
A red fire Pokemon with flames and molten rock armor
A silver steel Pokemon with metallic plating and sharp spikes
A pink fairy Pokemon with butterfly wings and sparkles
A brown ground Pokemon with boulder armor and earthquake powers
```

## 10. Advanced Usage

### Force specific device:

```bash
python3 tests/test_generation.py --checkpoint checkpoints/best_model.pth --device cuda
```

### Custom output directory:

```bash
python3 tests/test_generation.py --checkpoint checkpoints/best_model.pth --output-dir custom_output
```

## 11. Model Performance

Your model should:
- Generate coherent Pokemon-like sprites
- Respond to color descriptions (red, blue, yellow, etc.)
- Incorporate type-based features (fire, water, electric, etc.)
- Create varied outputs for the same description

## 12. Next Steps

Once you're happy with your model:

1. **Share your results**: Save your best generated images
2. **Experiment with descriptions**: Try creative combinations
3. **Fine-tune training**: Adjust hyperparameters for better results
4. **Create datasets**: Generate large sets of Pokemon for other projects

## 13. File Overview

- `tests/test_model.py`: Test model architecture and basic inference
- `tests/test_generation.py`: Comprehensive generation script with batching
- `tests/interactive_generation.py`: Interactive session for fun generation
- `tests/test_descriptions.txt`: Sample descriptions for testing
- `generate.py`: Original generation script (if you prefer it)

Happy generating! 🎮✨
