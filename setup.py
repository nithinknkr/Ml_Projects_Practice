from setuptools import setup, find_packages

def get_requirements(file_path):
    with open(file_path) as f:
        requirements = f.read().splitlines()
    return requirements

setup(
    name = 'p1',
    version = '0.1',
    author = 'Nithin',
    author_email='konudulanithin234@gmail.com',
    packages = find_packages(),
    install_requires=get_requirements('requirements.txt'),
)