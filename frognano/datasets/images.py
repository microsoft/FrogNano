from __future__ import annotations


def resolve_image_registry(
    image: str,
    image_registry: str | None,
    *,
    default_registry: str | None = None,
) -> str:
    registry = image_registry or default_registry
    if not registry:
        return image
    registry = registry.strip().rstrip("/")
    if not registry or registry.startswith("/") or "://" in registry:
        raise ValueError("image registry must be a hostname with an optional prefix")
    first, separator, path = image.partition("/")
    qualified = bool(
        separator and ("." in first or ":" in first or first == "localhost")
    )
    if qualified and not image_registry:
        return image
    if image.startswith(f"{registry}/"):
        return image
    return f"{registry}/{path if qualified else image}"
