#!/bin/bash

# Setup Kaggle API credentials safely
echo "========================================"
echo "Setup Kaggle API Credentials"
echo "========================================"

KAGGLE_DIR="$HOME/.kaggle"
KAGGLE_JSON="$KAGGLE_DIR/kaggle.json"

# Create .kaggle directory if it doesn't exist
mkdir -p "$KAGGLE_DIR"

# Create kaggle.json with the provided token
cat > "$KAGGLE_JSON" << 'EOF'
{"username":"YOUR_KAGGLE_USERNAME","key":"YOUR_KAGGLE_API_KEY"}
EOF

echo ""
echo "Created template at: $KAGGLE_JSON"
echo ""
echo "⚠ IMPORTANT: Edit this file and replace:"
echo "  - YOUR_KAGGLE_USERNAME: your Kaggle username"
echo "  - YOUR_KAGGLE_API_KEY: your API key"
echo ""
echo "You can:"
echo "1. Manually edit the file:"
echo "   nano $KAGGLE_JSON"
echo ""
echo "2. Or, if you saved kaggle.json from Kaggle:"
echo "   cp ~/Downloads/kaggle.json $KAGGLE_JSON"
echo ""

# Set permissions
chmod 600 "$KAGGLE_JSON"
echo "✓ Set permissions to 600"
echo ""

# Verify
if [ -f "$KAGGLE_JSON" ]; then
    echo "✓ Setup complete!"
    echo "Your credentials are now ready for: bash download_and_test_aptos.sh"
else
    echo "❌ Setup failed!"
fi
