"""Tests for check_missing_images.py script."""

import sys
import tempfile
from pathlib import Path

import pytest

# Add the book/scripts directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent / "book" / "scripts"))

from check_missing_images import find_image_references, resolve_image_path


class TestFindImageReferences:
    """Test finding image references in markdown files."""
    
    def test_markdown_image_syntax(self):
        """Test finding images with markdown syntax ![alt](path)."""
        content = """
        # Test Document
        
        Here is an image: ![alt text](images/test.png)
        And another: ![](../figures/chart.jpg)
        """
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as f:
            f.write(content)
            f.flush()
            
            images = find_image_references(f.name)
            
        Path(f.name).unlink()
        
        assert len(images) == 2
        assert "images/test.png" in images
        assert "../figures/chart.jpg" in images
    
    def test_link_syntax(self):
        """Test finding images with link syntax [text](path)."""
        content = """
        # Test Document
        
        [Click here for image](images/diagram.svg)
        [View chart](../data/plot.png)
        """
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as f:
            f.write(content)
            f.flush()
            
            images = find_image_references(f.name)
            
        Path(f.name).unlink()
        
        assert len(images) == 2
        assert "images/diagram.svg" in images
        assert "../data/plot.png" in images
    
    def test_jupyterbook_figure_syntax(self):
        """Test finding images with JupyterBook figure syntax."""
        content = """
        # Test Document
        
        ```{figure} ../figures/model_eti_comparison.png
        :name: fig-eti-comparison
        :width: 80%
        
        Comparison of ETI estimates
        ```
        
        ```{figure} images/response_rate.jpeg
        Response rates by model
        ```
        """
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as f:
            f.write(content)
            f.flush()
            
            images = find_image_references(f.name)
            
        Path(f.name).unlink()
        
        assert len(images) == 2
        assert "../figures/model_eti_comparison.png" in images
        assert "images/response_rate.jpeg" in images
    
    def test_mixed_content(self):
        """Test finding images in content with mixed syntax."""
        content = """
        # Results
        
        ![Model comparison](../figures/comparison.png)
        
        See the [detailed plot](charts/detail.svg) for more info.
        
        ```{figure} ../images/summary.jpg
        :width: 100%
        Summary figure
        ```
        
        Regular text with no images.
        """
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as f:
            f.write(content)
            f.flush()
            
            images = find_image_references(f.name)
            
        Path(f.name).unlink()
        
        assert len(images) == 3
        assert "../figures/comparison.png" in images
        assert "charts/detail.svg" in images
        assert "../images/summary.jpg" in images
    
    def test_case_insensitive_extensions(self):
        """Test that image extensions are matched case-insensitively."""
        content = """
        ![](image1.PNG)
        ![](image2.Jpg)
        ![](image3.JPEG)
        ![](image4.GIF)
        """
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as f:
            f.write(content)
            f.flush()
            
            images = find_image_references(f.name)
            
        Path(f.name).unlink()
        
        assert len(images) == 4
        assert "image1.PNG" in images
        assert "image2.Jpg" in images


class TestResolveImagePath:
    """Test resolving image paths."""
    
    def test_relative_path_same_directory(self, tmp_path):
        """Test resolving path in same directory as markdown file."""
        md_file = tmp_path / "results" / "test.md"
        md_file.parent.mkdir(parents=True)
        
        img_path = resolve_image_path("figure.png", md_file, tmp_path)
        
        assert img_path == tmp_path / "results" / "figure.png"
    
    def test_relative_path_subdirectory(self, tmp_path):
        """Test resolving path in subdirectory."""
        md_file = tmp_path / "docs" / "test.md"
        md_file.parent.mkdir(parents=True)
        
        img_path = resolve_image_path("images/figure.png", md_file, tmp_path)
        
        assert img_path == tmp_path / "docs" / "images" / "figure.png"
    
    def test_parent_directory_path(self, tmp_path):
        """Test resolving path with .. references."""
        md_file = tmp_path / "book" / "results" / "test.md"
        md_file.parent.mkdir(parents=True)
        
        img_path = resolve_image_path("../figures/chart.png", md_file, tmp_path)
        
        assert img_path == tmp_path / "book" / "figures" / "chart.png"
    
    def test_multiple_parent_references(self, tmp_path):
        """Test resolving path with multiple .. references."""
        md_file = tmp_path / "book" / "chapters" / "results" / "test.md"
        md_file.parent.mkdir(parents=True)
        
        img_path = resolve_image_path("../../images/logo.svg", md_file, tmp_path)
        
        assert img_path == tmp_path / "book" / "images" / "logo.svg"


def test_integration_missing_images(tmp_path):
    """Integration test: Check script detects missing images."""
    # Create directory structure
    book_dir = tmp_path / "book"
    book_dir.mkdir()
    
    results_dir = book_dir / "results"
    results_dir.mkdir()
    
    figures_dir = book_dir / "figures"
    figures_dir.mkdir()
    
    # Create markdown file with image references
    md_content = """
    # Test Results
    
    ```{figure} ../figures/existing.png
    This image exists
    ```
    
    ```{figure} ../figures/missing.png
    This image is missing
    ```
    """
    
    md_file = results_dir / "test.md"
    md_file.write_text(md_content)
    
    # Create only one of the images
    (figures_dir / "existing.png").touch()
    
    # Run the check (this would be the main() function logic)
    from check_missing_images import main
    
    # Mock sys.argv to avoid issues
    original_argv = sys.argv
    sys.argv = ['check_missing_images.py']
    
    # Change to book directory
    import os
    original_cwd = os.getcwd()
    os.chdir(book_dir)
    
    try:
        # The script should exit with code 1 due to missing image
        with pytest.raises(SystemExit) as exc_info:
            # Temporarily redirect __file__ in the module
            import check_missing_images
            original_file = check_missing_images.__file__
            check_missing_images.__file__ = str(book_dir / "scripts" / "check_missing_images.py")
            
            try:
                main()
            finally:
                check_missing_images.__file__ = original_file
        
        assert exc_info.value.code == 1
    finally:
        sys.argv = original_argv
        os.chdir(original_cwd)


def test_integration_all_images_exist(tmp_path):
    """Integration test: Check script passes when all images exist."""
    # Create directory structure
    book_dir = tmp_path / "book"
    book_dir.mkdir()
    
    results_dir = book_dir / "results"
    results_dir.mkdir()
    
    figures_dir = book_dir / "figures"
    figures_dir.mkdir()
    
    # Create markdown file with image references
    md_content = """
    # Test Results
    
    ```{figure} ../figures/chart1.png
    First chart
    ```
    
    ```{figure} ../figures/chart2.png
    Second chart
    ```
    """
    
    md_file = results_dir / "test.md"
    md_file.write_text(md_content)
    
    # Create all referenced images
    (figures_dir / "chart1.png").touch()
    (figures_dir / "chart2.png").touch()
    
    # Run the check
    from check_missing_images import main
    
    # Mock sys.argv
    original_argv = sys.argv
    sys.argv = ['check_missing_images.py']
    
    # Change to book directory
    import os
    original_cwd = os.getcwd()
    os.chdir(book_dir)
    
    try:
        # The script should exit with code 0 (success)
        with pytest.raises(SystemExit) as exc_info:
            # Temporarily redirect __file__ in the module
            import check_missing_images
            original_file = check_missing_images.__file__
            check_missing_images.__file__ = str(book_dir / "scripts" / "check_missing_images.py")
            
            try:
                main()
            finally:
                check_missing_images.__file__ = original_file
        
        assert exc_info.value.code == 0
    finally:
        sys.argv = original_argv
        os.chdir(original_cwd)