from setuptools import setup

APP = ['UI.py']
DATA_FILES = ['HokuLike Logo.png', 'magnifying-glass.png', 'highlighter.png']
OPTIONS = {
    'argv_emulation': True,
    'excludes': ['PyInstaller'],  # Exclude PyInstaller explicitly
}

setup(
    app=APP,
    data_files=DATA_FILES,
    options={'py2app': OPTIONS},
    setup_requires=['py2app']
)
