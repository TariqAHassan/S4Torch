"""

    Setup

"""
import shlex
from subprocess import check_call

from setuptools import find_packages, setup
from setuptools.command.install import install


class Install(install):
    def run(self) -> None:
        super().run()
        check_call(shlex.split("pip install -r requirements.txt"))


setup(
    packages=find_packages(exclude=["tests", "experiments"]),
    cmdclass={"install": Install},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3 :: Only",
        "Natural Language :: English",
    ],
)
