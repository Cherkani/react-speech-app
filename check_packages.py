import subprocess

# List of all the packages in the requirements.txt
packages = [
    "torch", "torchaudio", "soundfile", "omegaconf", "epitran", "audioread",
    "requests", "dtwalign", "eng_to_ipa", "pandas", "flask", "flask_cors",
    "pickle_mixin", "sqlalchemy", "transformers", "sentencepiece", "ortools"
]

# Function to check if the package is installed
def check_package_installed(package):
    try:
        result = subprocess.run(['pip', 'show', package], capture_output=True, text=True)
        if result.returncode == 0:
            print(f"{package} is installed.")
        else:
            print(f"{package} is NOT installed.")
    except Exception as e:
        print(f"Error checking {package}: {e}")

# Check all packages
for package in packages:
    check_package_installed(package)
