# TrueFace Extension Icons

This folder contains the icons for the TrueFace browser extension.

## Required Icons

The extension needs these icon files:
- `icon16.png` - 16x16 pixels (toolbar icon)
- `icon32.png` - 32x32 pixels (extension management)
- `icon48.png` - 48x48 pixels (extension management)
- `icon128.png` - 128x128 pixels (Chrome Web Store)

## Creating Icons

### Option 1: Use the Python Script
```bash
python create_icons.py
```
This will generate all required icons automatically.

### Option 2: Create Manually
Create PNG icons with the TrueFace logo or "TF" text:
- Use a blue/purple gradient background (#667eea to #764ba2)
- White text or logo
- Clean, modern design
- Transparent background optional

### Option 3: Use Existing Icons
If you have existing icons, rename them to match the required filenames above.

## Design Guidelines

- **Colors**: Use the TrueFace brand colors (blue #667eea, purple #764ba2)
- **Style**: Modern, clean, professional
- **Text**: "TF" or shield/security symbol
- **Background**: Solid color or gradient
- **Format**: PNG with transparency support

The icons represent security and trust, fitting the deepfake detection theme.
