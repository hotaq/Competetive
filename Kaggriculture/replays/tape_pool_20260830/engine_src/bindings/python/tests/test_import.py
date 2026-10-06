from __future__ import annotations

import importlib.util

import kaggriculture_rs


def test_installed_extension_imports_through_package() -> None:
    native = importlib.util.find_spec("kaggriculture_rs._native")
    assert native is not None
    assert native.origin is not None
    assert native.origin.endswith((".so", ".pyd"))
    assert kaggriculture_rs.Engine is not None
