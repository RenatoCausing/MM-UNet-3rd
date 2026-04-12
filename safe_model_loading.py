"""
Robust model loading with fallbacks for CUDA/dependency issues
"""

import torch
import torch.nn as nn
import sys
import warnings

def load_model_safe(config, device='cpu'):
    """
    Load model with fallbacks for import/compilation issues
    """
    try:
        # Try standard import first
        from src.models import give_model
        model = give_model(config).to(device)
        return model
    except (ImportError, TypeError, AttributeError) as e:
        print(f"⚠ Warning: Standard model import failed: {e}")
        print("Attempting fallback model loading...")
        
        # Fallback: Try to load just the state dict without full import
        try:
            return load_model_from_checkpoint_only(config, device)
        except Exception as e2:
            print(f"⚠ Warning: Fallback also failed: {e2}")
            print("Will attempt checkpoint loading without model architecture...")
            return None


def load_model_from_checkpoint_only(config, device='cpu'):
    """
    Load model checkpoint without requiring full source imports
    Creates a minimal model structure
    """
    try:
        # Try to import just the necessary pieces
        from src.UM_Net.MMUNet import MM_Net
        model = MM_Net(num_classes=1).to(device)
        return model
    except (ImportError, ModuleNotFoundError, AttributeError) as e:
        try:
            # Try alternative import path
            from src.UM_Net.UM_Net import UM_Net
            model = UM_Net(num_classes=1).to(device)
            return model
        except (ImportError, ModuleNotFoundError, AttributeError) as e2:
            try:
                # Last attempt: try direct import
                import sys
                from pathlib import Path
                src_path = Path('src/UM_Net/MMUNet.py')
                if src_path.exists():
                    import importlib.util
                    spec = importlib.util.spec_from_file_location("MMUNet", src_path)
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)
                    model = module.MM_Net(num_classes=1).to(device)
                    return model
                else:
                    print(f"⚠ MMUNet.py not found at {src_path}")
                    return None
            except Exception as e3:
                # Last resort: return a simple placeholder
                print(f"⚠ Could not load MM_Net architecture ({e}, {e2}, {e3})")
                return None


def safe_load_checkpoint(model, checkpoint_path, device):
    """
    Safely load checkpoint, handling state dict mismatches
    """
    if model is None:
        print(f"⚠ Model is None, cannot load checkpoint")
        return False
    
    if not torch.cuda.is_available() and device.type == 'cuda':
        device = torch.device('cpu')
        model = model.to(device)
    
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        if isinstance(checkpoint, dict):
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint
        else:
            state_dict = checkpoint
        
        # Try to load state dict
        try:
            model.load_state_dict(state_dict, strict=False)
            print(f"✓ Checkpoint loaded (non-strict mode)")
            return True
        except Exception as e:
            print(f"⚠ Warning: State dict loading issue: {e}")
            print("  Attempting key-by-key loading...")
            
            # Try loading keys that match
            model_keys = set(model.state_dict().keys())
            checkpoint_keys = set(state_dict.keys())
            
            matching_keys = model_keys & checkpoint_keys
            print(f"  Found {len(matching_keys)} matching keys")
            
            if matching_keys:
                partial_state = {k: state_dict[k] for k in matching_keys}
                model.load_state_dict(partial_state, strict=False)
                print(f"✓ Partial checkpoint loaded ({len(matching_keys)} weights)")
                return True
            else:
                print(f"⚠ No matching keys found")
                return False
    
    except Exception as e:
        print(f"❌ Error loading checkpoint: {e}")
        return False
