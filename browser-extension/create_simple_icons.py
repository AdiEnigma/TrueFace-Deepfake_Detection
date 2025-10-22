#!/usr/bin/env python3
"""
Create Simple TrueFace Icons
Creates basic icons for the browser extension
"""

import os
from PIL import Image, ImageDraw, ImageFont

def create_simple_icon(size, filename):
    """Create a simple TrueFace icon"""
    
    # Create image with blue background
    img = Image.new('RGBA', (size, size), (102, 126, 234, 255))  # Blue background
    draw = ImageDraw.Draw(img)
    
    # Draw white circle
    margin = size // 8
    draw.ellipse([margin, margin, size-margin, size-margin], fill=(255, 255, 255, 255))
    
    # Draw "TF" text
    try:
        font_size = max(8, size // 3)
        try:
            font = ImageFont.truetype("arial.ttf", font_size)
        except:
            font = ImageFont.load_default()
        
        text = "TF"
        bbox = draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        
        text_x = (size - text_width) // 2
        text_y = (size - text_height) // 2 - 2
        
        draw.text((text_x, text_y), text, fill=(102, 126, 234, 255), font=font)
    except:
        # Fallback: draw a simple shape
        center = size // 2
        draw.rectangle([center-size//6, center-size//6, center+size//6, center+size//6], 
                      fill=(102, 126, 234, 255))
    
    # Save the image
    img.save(filename, 'PNG')
    print(f"Created {filename}")

def create_icons_without_pil():
    """Create simple SVG icons and convert to PNG using basic method"""
    
    sizes = [16, 32, 48, 128]
    
    for size in sizes:
        # Create a simple colored square as PNG
        try:
            img = Image.new('RGBA', (size, size), (102, 126, 234, 255))
            draw = ImageDraw.Draw(img)
            
            # Draw white circle
            margin = 2
            draw.ellipse([margin, margin, size-margin, size-margin], fill=(255, 255, 255, 255))
            
            # Draw blue center
            center_margin = size // 4
            draw.ellipse([center_margin, center_margin, size-center_margin, size-center_margin], 
                        fill=(102, 126, 234, 255))
            
            filename = f'icons/icon{size}.png'
            img.save(filename, 'PNG')
            print(f"✅ Created {filename}")
            
        except ImportError:
            print(f"❌ Cannot create icon{size}.png - PIL not available")
            return False
    
    return True

def main():
    """Create all required icon sizes"""
    
    # Create icons directory
    os.makedirs('icons', exist_ok=True)
    
    try:
        # Try to create icons with PIL
        success = create_icons_without_pil()
        if success:
            print("\n✅ All icons created successfully!")
            return True
    except Exception as e:
        print(f"❌ Error creating icons: {e}")
    
    # If PIL fails, create placeholder files
    print("⚠️ Creating placeholder icon files...")
    sizes = [16, 32, 48, 128]
    
    for size in sizes:
        filename = f'icons/icon{size}.png'
        # Create a minimal PNG file (1x1 pixel)
        try:
            with open(filename, 'wb') as f:
                # Minimal PNG file header for 1x1 transparent pixel
                png_data = bytes([
                    0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A,  # PNG signature
                    0x00, 0x00, 0x00, 0x0D, 0x49, 0x48, 0x44, 0x52,  # IHDR chunk
                    0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01,  # 1x1 dimensions
                    0x08, 0x06, 0x00, 0x00, 0x00, 0x1F, 0x15, 0xC4,  # RGBA, CRC
                    0x89, 0x00, 0x00, 0x00, 0x0B, 0x49, 0x44, 0x41,  # IDAT chunk
                    0x54, 0x08, 0x1D, 0x01, 0x00, 0x00, 0x00, 0x00,  # Compressed data
                    0x00, 0x37, 0x6E, 0xF9, 0x24, 0x00, 0x00, 0x00,  # End
                    0x00, 0x49, 0x45, 0x4E, 0x44, 0xAE, 0x42, 0x60, 0x82  # IEND
                ])
                f.write(png_data)
            print(f"📝 Created placeholder {filename}")
        except Exception as e:
            print(f"❌ Failed to create {filename}: {e}")
            return False
    
    print("\n⚠️ Placeholder icons created. Extension will load but icons will be minimal.")
    print("💡 Install Pillow (pip install Pillow) and run again for better icons.")
    return True

if __name__ == "__main__":
    main()
