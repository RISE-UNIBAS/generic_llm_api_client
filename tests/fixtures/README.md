# Test Fixtures

This directory contains test images for integration tests.

## Required Test Images

For running integration tests, you'll need sample images. You can:

1. **Use the auto-generated images**: The conftest.py will automatically create temporary test images when running tests
2. **Add your own images**: Place test images here for specific test cases

## Sample Images for Manual Testing

If you want to add permanent test images for integration testing:

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

## What's Automatically Generated

The `conftest.py` fixture automatically creates:
- **sample_image_path**: A temporary 100x100 red square PNG for each test
- Cleaned up automatically after tests complete

## For Humanities Benchmarks

When testing with real humanities data:
- Place sample manuscript images here
- Use diverse samples (different scripts, languages, conditions)
- Keep file sizes reasonable (< 5MB each)
- Document what each image tests
