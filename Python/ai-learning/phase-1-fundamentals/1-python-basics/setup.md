# Python Development Environment Setup Guide

## Required Tools Overview

### 1. Python 3

- **What**: The Python programming language interpreter
- **Why**: The core tool we'll use for all our programming
- **Version**: Python 3.10 or higher recommended

### 2. VS Code

- **What**: A powerful and lightweight code editor
- **Why**: Provides excellent Python support and debugging capabilities
- **Required Extensions**:
  - Python (Microsoft)
  - Python IntelliSense
  - Python Docstring Generator

### 3. Git

- **What**: Version control system
- **Why**: Track changes in our code and collaborate

## Step-by-Step Installation (Linux Mint)

### 1. Update Your System

```bash
sudo apt update
sudo apt upgrade -y
```

### 2. Install Python 3

```bash
# Check if Python is installed
python3 --version

# If not installed:
sudo apt install python3 python3-pip -y
```

### 3. Install Virtual Environment Tools

```bash
# Install venv module
sudo apt install python3-venv -y

# Install pip (Python package manager)
sudo apt install python3-pip -y
```

### 4. Install VS Code

1. Open Software Manager in Linux Mint
2. Search for "Visual Studio Code"
3. Install VS Code
4. Open VS Code and install extensions:
   - Click Extensions icon (or Ctrl+Shift+X)
   - Search and install:
     - "Python" by Microsoft
     - "Python IntelliSense"
     - "Python Docstring Generator"

### 5. Install Git

```bash
sudo apt install git -y
```

### 6. Configure Git

```bash
git config --global user.name "Your Name"
git config --global user.email "your.email@example.com"
```

## Verify Installation

Run these commands to verify everything is installed correctly:

```bash
# Check Python
python3 --version

# Check pip
pip3 --version

# Check git
git --version
```

## Setting Up a Python Project

1. Create a project directory:

```bash
mkdir python-practice
cd python-practice
```

2. Create a virtual environment:

```bash
python3 -m venv venv
```

3. Activate the virtual environment:

```bash
source venv/bin/activate
```

4. Your prompt should change to show `(venv)` at the beginning

5. Test your setup:

```bash
# Create a test file
echo 'print("Hello, Python!")' > test.py

# Run it
python test.py
```

## Troubleshooting

### Common Issues and Solutions

1. **Python Command Not Found**

   ```bash
   sudo apt install python3
   ```

2. **Pip Not Found**

   ```bash
   sudo apt install python3-pip
   ```

3. **Permission Denied When Installing Packages**

   - Always use `pip` with the `--user` flag when not in a virtual environment

   ```bash
   pip3 install --user package_name
   ```

4. **Virtual Environment Not Activating**
   - Ensure you're in the correct directory
   - Try using the full path:
   ```bash
   source /full/path/to/venv/bin/activate
   ```

## Next Steps

After installation:

1. Open VS Code in your project directory:
   ```bash
   code .
   ```
2. Select your Python interpreter in VS Code:
   - Press `Ctrl+Shift+P`
   - Type "Python: Select Interpreter"
   - Choose the one from your virtual environment

You're now ready to start coding! Head to the first lesson in the `lessons` folder.

## Need Help?

If you encounter any issues:

1. Check the error message carefully
2. Google the exact error message
3. Check Stack Overflow
4. Ask for help in our repository's issues section

Remember: Always activate your virtual environment before starting to code:

```bash
source venv/bin/activate
```
