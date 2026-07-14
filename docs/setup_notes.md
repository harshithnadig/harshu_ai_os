# Harshu AI OS Setup Notes

1. Python installation 
- Your Python version = "Python 3.14.6"
- The Python executable path = "C:\Python314\python.exe"
- The pip path = "pip 26.1.2 from C:\Users\harsh\AppData\Roaming\Python\Python314\site-packages\pip (python 3.14)"

2. Virtual environment
- VEnv creates an isolated Python environment specifically for the current project in current directory.
- The Python executable path = "C:\Users\harsh\OneDrive\Documents\python ai enginner\.venv\Scripts\python.exe"
- The pip path = "pip 26.1.2 from C:\Users\harsh\OneDrive\Documents\python ai enginner\.venv\Lib\site-packages\pip (python 3.14)"
- Activation places the .venv Scripts folder earlier in PATH, so python and pip use the project environment.

3. Package installation
- Name: python-dotenv
- Version: 1.2.2
- Summary: Read key-value pairs from a .env file and set them as environment variables
- Home-page: 
- Author: 
- Author-email: Saurabh Kumar <me+github@saurabh-kumar.com>
- License: BSD-3-Clause
- Location: C:\Users\harsh\OneDrive\Documents\python ai enginner\.venv\Lib\site-packages
- Requires: 
- Required-by: 
4. Dependency freeze
- What pip freeze produced = "python-dotenv==1.2.2"
- Why another developer would need requirements.txt = it helps to install the same dependencies that the original 
developer used, ensuring consistency across different environments.
- Why the package version is recorded = to ensure that the same version of the package is installed, preventing potential 
compatibility issues.
5. Environment variables
- Variable name: HARSHU_AI_OS_MODE
- Variable value: development
- Why configuration outside code is useful. = it allows you to change the configuration without modifying the code.

Environment variable: value available to the running process.
.env file: convenient local file from which variables can be loaded.
load_dotenv(): bridge that loads the file’s values.
6. Deactivation
- After deactivation, the (.venv) marker disappeared.
- The Python executable changed from the .venv Python back to C:\Python314\python.exe.
- This shows that the terminal stopped using the project-specific virtual environment.