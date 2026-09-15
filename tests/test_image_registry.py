import pytest

from frognano.datasets.images import resolve_image_registry


@pytest.mark.parametrize(
    "image,path",
    [
        ("ubuntu", "ubuntu"),
        ("ubuntu:22.04", "ubuntu:22.04"),
        ("my.image:latest", "my.image:latest"),
        ("owner/image:tag", "owner/image:tag"),
        ("ghcr.io/owner/image:tag", "owner/image:tag"),
        ("docker.io/library/python:3.12", "library/python:3.12"),
        ("localhost/owner/image:tag", "owner/image:tag"),
        ("localhost:5000/owner/image:tag", "owner/image:tag"),
        ("source.test:5000/team/image:tag", "team/image:tag"),
        ("ubuntu@sha256:" + "a" * 64, "ubuntu@sha256:" + "a" * 64),
        (
            "ghcr.io/owner/image:tag@sha256:" + "a" * 64,
            "owner/image:tag@sha256:" + "a" * 64,
        ),
    ],
)
@pytest.mark.parametrize("registry", ["mirror.example", "mirror.example:5000/prefix/"])
def test_explicit_registry_replaces_host_but_preserves_image_reference(
    image, path, registry
):
    assert resolve_image_registry(image, registry) == f"{registry.rstrip('/')}/{path}"
    assert resolve_image_registry(image, None) == image


@pytest.mark.parametrize("registry", ["mirror.example", "mirror.example:5000/prefix"])
def test_registry_override_is_idempotent(registry):
    image = f"{registry}/owner/image:tag"
    assert resolve_image_registry(image, registry) == image


@pytest.mark.parametrize(
    "image,expected",
    [
        ("ubuntu:22.04", "fallback.test/ubuntu:22.04"),
        ("owner/image:tag", "fallback.test/owner/image:tag"),
        ("ghcr.io/owner/image:tag", "ghcr.io/owner/image:tag"),
    ],
)
def test_default_registry_only_qualifies_unqualified_references(image, expected):
    assert (
        resolve_image_registry(image, None, default_registry="fallback.test/")
        == expected
    )
    assert resolve_image_registry(
        image, "selected.test", default_registry="fallback.test"
    ).startswith("selected.test/")


@pytest.mark.parametrize(
    "registry", [" ", "/", "///", "/mirror", "https://mirror.test"]
)
def test_registry_override_rejects_invalid_prefixes(registry):
    with pytest.raises(ValueError, match="image registry must be a hostname"):
        resolve_image_registry("owner/image:tag", registry)
