#!/usr/bin/env python3
"""
Check for missing images referenced in markdown files.
"""

import re
import sys
from pathlib import Path


def find_image_references(markdown_file):
    """Find all image references in a markdown file."""
    with open(markdown_file, "r") as f:
        content = f.read()

    # Pattern for markdown image references
    # The ![alt](path) pattern already matches [text](path) when alt is empty
    # So we need to be more specific
    patterns = [
        r"!\[.*?\]\((.*?\.(?:png|jpg|jpeg|svg|gif))\)",  # ![alt](image.png)
        r"(?<!!)\[.*?\]\((.*?\.(?:png|jpg|jpeg|svg|gif))\)",  # [text](image.png) but not ![
        r"```\{figure\}\s*(.*?\.(?:png|jpg|jpeg|svg|gif))",  # ```{figure} image.png
    ]

    images = []
    for pattern in patterns:
        matches = re.findall(pattern, content, re.IGNORECASE)
        images.extend(matches)

    # Remove duplicates while preserving order
    seen = set()
    unique_images = []
    for img in images:
        if img not in seen:
            seen.add(img)
            unique_images.append(img)

    return unique_images


def resolve_image_path(image_ref, markdown_file, book_dir):
    """Resolve relative image path to absolute path."""
    markdown_path = Path(markdown_file)

    # Handle relative paths
    if image_ref.startswith("../"):
        # Go up from markdown file location
        base_path = markdown_path.parent
        path_parts = image_ref.split("/")

        # Process .. references
        while path_parts and path_parts[0] == "..":
            base_path = base_path.parent
            path_parts.pop(0)

        return base_path / "/".join(path_parts)
    else:
        # Relative to markdown file directory
        return markdown_path.parent / image_ref


def main():
    book_dir = Path(__file__).parent.parent
    missing_images = []

    # Find all markdown files, excluding _build directory
    markdown_files = [
        f
        for f in book_dir.glob("**/*.md")
        if "_build" not in str(f.relative_to(book_dir))
    ]

    for md_file in markdown_files:
        images = find_image_references(md_file)

        for img_ref in images:
            img_path = resolve_image_path(img_ref, md_file, book_dir)

            if not img_path.exists():
                relative_md = md_file.relative_to(book_dir)
                missing_images.append(
                    {
                        "markdown": str(relative_md),
                        "reference": img_ref,
                        "expected_path": str(img_path.relative_to(book_dir)),
                    }
                )

    if missing_images:
        print("❌ Missing images found:")
        print("-" * 60)
        for item in missing_images:
            print(f"File: {item['markdown']}")
            print(f"  Reference: {item['reference']}")
            print(f"  Expected at: {item['expected_path']}")
            print()

        print(f"Total missing images: {len(missing_images)}")
        sys.exit(1)
    else:
        print("✅ All referenced images exist!")
        sys.exit(0)


if __name__ == "__main__":
    main()
