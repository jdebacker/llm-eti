import os


def stitch_py_files(root_dir, output_file):
    """
    Collect all Python files in a directory and its subdirectories,
    and combine their content into a single output text file,
    skipping files in .venv directories.

    Args:
        root_dir (str): Path to the root directory to search for .py files.
        output_file (str): Path to the output file.
    """
    with open(output_file, "w", encoding="utf-8") as outfile:
        for root, _, files in os.walk(root_dir):
            # Skip directories containing '.venv'
            if ".venv" in root:
                continue
            for file in files:
                if file.endswith(".py"):
                    file_path = os.path.join(root, file)
                    try:
                        with open(
                            file_path, "r", encoding="utf-8", errors="ignore"
                        ) as infile:
                            outfile.write(
                                f"# File: {file_path}\n"
                            )  # Add header for each file
                            outfile.write(
                                infile.read() + "\n\n"
                            )  # Append file content
                    except Exception as e:
                        print(f"Error reading file {file_path}: {e}")
    print(f"All .py files have been stitched into {output_file}")


# Specify the root directory and output file
root_directory = "."
output_file = "stitched_files.txt"

stitch_py_files(root_directory, output_file)
