#!/usr/bin/env python3
from PIL import Image, ImageDraw, ImageFont

# Create 1920x1080 image with transparency
img = Image.new('RGBA', (1920, 1080), (0, 0, 0, 0))
draw = ImageDraw.Draw(img)

# Green rectangle (green screen background)
draw.rectangle([100, 100, 400, 400], fill=(0, 177, 64, 255))

# White text
try:
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 72)
except:
    font = ImageFont.load_default()

draw.text((150, 200), "LIVE", fill=(255, 255, 255, 255), font=font)

# Add logo/overlay area (semi-transparent)
draw.rectangle([800, 50, 1100, 200], fill=(255, 255, 255, 200))
draw.text((850, 100), "LOGO", fill=(0, 0, 0, 255), font=font)

# Save
img.save('foreground.png')
print("Created foreground.png (1920x1080)")
