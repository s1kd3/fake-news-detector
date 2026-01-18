In Python, __init__.py is a special file that tells the Python interpreter that a directory should be treated as a package.

When a folder contains an __init__.py file, Python allows modules inside that folder to be imported using dot notation, such as:
from src.model import train_model
Without __init__.py, Python may not recognize the folder as a package, which can lead to import errors, especially in structured projects.

Purpose of __init__.py

Marks a directory as a Python package

Enables clean and reliable imports

Helps organize large projects into logical modules

Avoids the need for manual sys.path manipulation

Improves compatibility with tools like Streamlit, pip, and deployment platforms
