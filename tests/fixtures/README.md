# Test Fixtures

This directory contains test images for integration tests.

## Required Test Images

Integration tests use sample images from either of the following sources:

1. **Generated images**: Fixtures in `conftest.py` create temporary images during test execution
2. **Custom images**: Add images to this directory for specific test cases

## Sample Images for Manual Testing

Create permanent test images for manual integration testing with the following command:

```bash
# Example: Create simple test images with PIL
python3 -c "
from PIL import Image, ImageDraw

# Red square
img = Image.new('RGB', (200, 200), color='white')
draw = ImageDraw.Draw(img)
draw.rectangle([50, 50, 150, 150], fill='red')
img.save('tests/fixtures/red_square.png')

# Blue circle
img = Image.new('RGB', (200, 200), color='white')
draw = ImageDraw.Draw(img)
draw.ellipse([50, 50, 150, 150], fill='blue')
img.save('tests/fixtures/blue_circle.png')
"
```

## Automatically Generated Fixtures

The `sample_image_path` fixture in `conftest.py` creates a temporary 100x100 red square
PNG for each test and removes it after the test completes.

## For Humanities Benchmarks

When testing with real humanities data:
- Place sample manuscript images here
- Use diverse samples (different scripts, languages, conditions)
- Keep each image smaller than 5 MB
- Document what each image tests
