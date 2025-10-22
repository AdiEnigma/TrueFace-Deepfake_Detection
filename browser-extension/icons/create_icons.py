#!/usr/bin/env python3
"""
Create TrueFace Extension Icons
Generates icons in different sizes for the browser extension
"""

from PIL import Image, ImageDraw, ImageFont
import os

def create_icon(size, filename):
    """Create a TrueFace icon of specified size"""
    
    # Create image with transparent background
    img = Image.new('RGBA', (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    
    # Colors
    bg_color = (102, 126, 234)  # Primary blue
    accent_color = (118, 75, 162)  # Purple accent
    text_color = (255, 255, 255)  # White text
    
    # Draw gradient background circle
    center = size // 2
    radius = size // 2 - 2
    
    # Create gradient effect
    for i in range(radius):
        alpha = int(255 * (1 - i / radius))
        color = (*bg_color, alpha)
        draw.ellipse([center - radius + i, center - radius + i, 
                     center + radius - i, center + radius - i], 
                    fill=color)
    
    # Draw main circle
    draw.ellipse([2, 2, size-2, size-2], fill=bg_color, outline=accent_color, width=2)
    
    # Draw "TF" text or shield icon based on size
    if size >= 32:
        try:
            # Try to use a nice font
            font_size = max(8, size // 4)
            font = ImageFont.truetype("arial.ttf", font_size)
        except:
            # Fallback to default font
            font = ImageFont.load_default()
        
        # Draw "TF" text
        text = "TF"
        bbox = draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        
        text_x = (size - text_width) // 2
        text_y = (size - text_height) // 2 - 2
        
        # Draw text shadow
        draw.text((text_x + 1, text_y + 1), text, fill=(0, 0, 0, 128), font=font)
        # Draw main text
        draw.text((text_x, text_y), text, fill=text_color, font=font)
    else:
        # For smaller icons, draw a simple shield shape
        points = [
            (center, size // 4),
            (size - size // 4, size // 3),
            (size - size // 4, size - size // 3),
            (center, size - size // 6),
            (size // 4, size - size // 3),
            (size // 4, size // 3)
        ]
        draw.polygon(points, fill=text_color, outline=accent_color)
    
    # Add subtle highlight
    highlight_radius = radius - 4
    draw.arc([center - highlight_radius, center - highlight_radius,
              center + highlight_radius, center + highlight_radius],
             start=225, end=315, fill=(255, 255, 255, 64), width=2)
    
    # Save the image
    img.save(filename, 'PNG')
    print(f"Created {filename} ({size}x{size})")

def main():
    """Create all required icon sizes"""
    
    # Create icons directory if it doesn't exist
    os.makedirs('icons', exist_ok=True)
    
    # Icon sizes required for Chrome extension
    sizes = [16, 32, 48, 128]
    
    for size in sizes:
        filename = f'icons/icon{size}.png'
        create_icon(size, filename)
    
    print(f"\n✅ Created {len(sizes)} icon files")
    print("Icons are ready for the browser extension!")

if __name__ == "__main__":
    main()
