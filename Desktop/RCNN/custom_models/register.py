from mmdet.registry import MODELS
from .custom_cascade_with_meta import CustomCascadeWithMeta
from .custom_heads import FCHead, RegHead

def register_custom_models():
    """Register all custom models and heads.
    Note: The actual registration is done via decorators, this function
    just ensures the modules are imported."""
    pass  # All modules are already registered via decorators

# No need for explicit registration here 