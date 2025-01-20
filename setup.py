from setuptools import setup, find_packages

# Učitaj sadržaj requirements.txt
def parse_requirements(filename):
    """Parse a requirements.txt file into a list of dependencies."""
    with open(filename, "r") as file:
        lines = file.read().splitlines()
        return [line.strip() for line in lines if line.strip() and not line.startswith("#")]

setup(
    name='Spotify_Recommender',
    version='1.0',
    description='Spotify song recommender system using machine learning and data processing techniques.',
    packages=find_packages(),
    include_package_data=True,
    install_requires=parse_requirements("requirements.txt"),
    extras_require={
        "dev": ["pytest", "flake8"],
    },
    entry_points={
        'console_scripts': [
            'spotify-recommender=agent_app:main',  # Provjerite je li funkcija main implementirana
        ],
    },
    classifiers=[
        'Programming Language :: Python :: 3',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.9',
)
