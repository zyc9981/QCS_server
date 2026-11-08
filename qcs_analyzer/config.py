"""Shared file-dialog configuration."""

from . import _implementation

FileSelectionCancelled = _implementation.FileSelectionCancelled
set_default_file_dialog_directory = _implementation.set_default_file_dialog_directory


def get_default_file_dialog_directory():
    """Return the folder used as the initial location in file dialogs."""
    return _implementation.DEFAULT_FILE_DIALOG_DIRECTORY
