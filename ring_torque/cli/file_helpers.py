"""Contains various helper functions that deal with files and directories."""

# Standard libraries
import logging
import os


log = logging.getLogger(__name__)


# List of file extensions
VIDEO_EXTENSIONS: list[str] = [".mp4", ".gif"]
DATA_EXTENSIONS: list[str] = [".hdf5"]


def format_output_path(output_path: str, supported_extensions: list[str]) -> str:
    """Formats the output path to have the correct file extension.

    Parameters
    ----------
    output_path : str
        The path to the output file.
    supported_extensions : list[str]
        A list of supported file extensions. The extension must include the leading `.`.

    Returns
    -------
    output_path : str
        The path to the output file.
    """
    assert len(supported_extensions) > 0
    root, extension = os.path.splitext(output_path)
    if extension not in supported_extensions:
        log.warning(
            f"The given file format {extension} is not supported! Supported formats include {supported_extensions}"
        )
        log.warning(f"Defaulting to {supported_extensions[0]}...")
        extension = supported_extensions[0]
    output_path = f"{root}{extension}"
    return output_path


def check_both_files(input_path: str, output_path: str) -> bool:
    """Check that the given input and output paths are both files or can be a file in the case of the output path.

    Parameters
    ----------
    input_path : str
        The input path.
    output_path : str
        The output path.

    Returns
    -------
    both_files : bool
        Whether the given input and output paths are both files.
    """
    # Check that the first file is readable
    both_files = os.path.isfile(input_path)
    # Check that the second file is writable to or can be created if it does not exist
    both_files &= os.path.isfile(output_path) or os.access(
        os.path.dirname(output_path), os.W_OK
    )

    return both_files


def check_all_files(input_paths: list[str], output_path: str) -> bool:
    """Check that the given input paths are all files or can be a file in the case of the output path.

    Parameters
    ----------
    input_paths : list[str]
        The input paths.
    output_path : str
        The output path.

    Returns
    -------
    all_files : bool
        Whether the given input paths are all files.
    """
    # Check that the first file is readable
    all_files = all([os.path.isfile(input_path) for input_path in input_paths])
    # Check that the second file is writable to or can be created if it does not exist
    all_files &= os.path.isfile(output_path) or os.access(
        os.path.dirname(output_path), os.W_OK
    )

    return all_files


def check_both_directories(input_path: str, output_path: str) -> bool:
    """Check that the given input and output paths are both directories.

    Parameters
    ----------
    input_path : str
        The input path.
    output_path : str
        The output path.

    Returns
    -------
    both_directories : bool
        Whether the given input and output paths are both directories.
    """
    both_directories = os.path.isdir(input_path) and os.path.isdir(output_path)
    return both_directories


def check_file_and_directory(input_path: str, output_path: str) -> bool:
    """Check that the given input path is a file and the output path is a directory.

    Parameters
    ----------
    input_path : str
        The input path.
    output_path : str
        The output path.

    Returns
    -------
    file_and_directory : bool
        Whether the given input path is a file and the output path is a directory.
    """
    file_and_directory = os.path.isfile(input_path) and os.path.isdir(output_path)
    return file_and_directory
