"""
Patch for causal_conv1d CUDA compatibility issues
Provides fallback implementations when CUDA extensions fail to load
"""

import sys
import warnings

def patch_causal_conv1d():
    """Apply monkey patch for causal_conv1d import errors"""
    
    # Try to fix the import issue before anything else tries to load it
    if 'causal_conv1d' not in sys.modules:
        try:
            import causal_conv1d
        except (ImportError, AttributeError) as e:
            warnings.warn(f"causal_conv1d CUDA import failed: {e}. Using CPU-only fallback.", stacklevel=2)
            
            # Create a mock module with dummy implementations
            import types
            causal_conv1d = types.ModuleType('causal_conv1d')
            
            def dummy_causal_conv1d_fn(*args, **kwargs):
                """Dummy implementation - this shouldn't be called"""
                raise NotImplementedError("causal_conv1d_fn requires CUDA support. "
                                        "Ensure PyTorch and cuDNN versions match.")
            
            def dummy_causal_conv1d_update(*args, **kwargs):
                """Dummy implementation - this shouldn't be called"""
                raise NotImplementedError("causal_conv1d_update requires CUDA support. "
                                        "Ensure PyTorch and cuDNN versions match.")
            
            causal_conv1d.causal_conv1d_fn = dummy_causal_conv1d_fn
            causal_conv1d.causal_conv1d_update = dummy_causal_conv1d_update
            sys.modules['causal_conv1d'] = causal_conv1d
    
    # Also patch mamba_ssm if needed
    if 'mamba_ssm' not in sys.modules:
        try:
            # Try importing normally first
            import mamba_ssm
        except (ImportError, TypeError) as e:
            if "cannot unpack non-iterable NoneType" in str(e):
                warnings.warn("Mamba CUDA import failed. Using CPU-only fallback.", stacklevel=2)
                
                # Create a minimal mock mamba module
                import types
                mamba_ssm = types.ModuleType('mamba_ssm')
                
                # Create mock Mamba class
                import torch.nn as nn
                class MockMamba(nn.Module):
                    def __init__(self, *args, **kwargs):
                        super().__init__()
                        self.linear = nn.Linear(1, 1)  # dummy layer
                    
                    def forward(self, x):
                        raise NotImplementedError("Mamba requires proper CUDA support. "
                                                "Check PyTorch/CUDA installation.")
                
                mamba_ssm.Mamba = MockMamba
                
                # Add submodules
                mamba_ssm.modules = types.ModuleType('modules')
                mamba_ssm.modules.mamba_simple = types.ModuleType('mamba_simple')
                mamba_ssm.modules.mamba_simple.Mamba = MockMamba
                
                sys.modules['mamba_ssm'] = mamba_ssm
                sys.modules['mamba_ssm.modules'] = mamba_ssm.modules
                sys.modules['mamba_ssm.modules.mamba_simple'] = mamba_ssm.modules.mamba_simple

if __name__ == "__main__":
    patch_causal_conv1d()
