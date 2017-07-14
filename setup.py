from setuptools import setup, find_packages

INSTALL_REQUIRES = []
NAMESPACE_PACKAGES = []
DISTNAME = 'astrology'
DOWNLOAD_URL = ''
AUTHOR = 'Robert M. Johnson'
VERSION = 1.0
DESCRIPTION = 'Various machine learning algorithms'


def setup_package():
    metadata = dict(name=DISTNAME,
                    packages=find_packages(),
                    author=AUTHOR,
                    description=DESCRIPTION,
                    version=VERSION,
                    install_requires=INSTALL_REQUIRES,
                    namespace_packages=NAMESPACE_PACKAGES)
    setup(**metadata)
if __name__ == '__main__':
    setup_package()
