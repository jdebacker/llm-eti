"""Test that the package imports work correctly."""

import pytest


def test_package_imports():
    """Test that all main modules can be imported."""
    # Test main package import
    import llm_eti
    
    # Test version
    assert hasattr(llm_eti, '__version__')
    assert llm_eti.__version__ == '0.1.0'
    
    # Test main exports
    from llm_eti import (
        Config,
        GPTClient,
        TaxSimulation,
        SimulationParams,
    )
    
    # All imports should succeed without error
    assert True


def test_submodule_imports():
    """Test that submodules can be imported directly."""
    from llm_eti import config
    from llm_eti import gpt_utils
    from llm_eti import tax_utils
    from llm_eti import data_utils
    from llm_eti import simulation_engine
    from llm_eti import regression_utils
    from llm_eti import plotting
    from llm_eti import table_utils
    
    # All imports should succeed
    assert True