from app.ml.registry import MLRegistry, ml_registry

def get_ml_registry() -> MLRegistry:
    """Dependency injection provider for ML singleton."""
    return ml_registry
