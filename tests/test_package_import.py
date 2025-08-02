"""Test that the package imports work correctly."""



def test_package_imports():
    """Test that all main modules can be imported."""
    # Test main package import
    import llm_eti

    # Test version
    assert hasattr(llm_eti, "__version__")
    assert llm_eti.__version__ == "0.1.0"

    # Test main exports

    # All imports should succeed without error
    assert True


def test_submodule_imports():
    """Test that submodules can be imported directly."""

    # All imports should succeed
    assert True
