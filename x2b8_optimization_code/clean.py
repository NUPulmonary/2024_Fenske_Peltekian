import re

def clean_requirements(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    cleaned_lines = []
    for line in lines:
        # Replace file:// paths with just the package name and version
        cleaned_line = re.sub(r'@ file://.+?/([^/]+)', r'==\1', line)
        cleaned_lines.append(cleaned_line)

    with open(file_path, 'w') as file:
        file.writelines(cleaned_lines)

clean_requirements('x2b8_requirements.txt')
