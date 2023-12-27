#!/usr/bin/env python3
from setuptools import setup, find_packages

description = "Audible alerting with OpenTTS for MagAO-X"

setup(
    name="audibleAlerts",
    version="0.0.1.dev",
    url="https://github.com/magao-x/MagAOX",
    description=description,
    author="Joseph D. Long",
    author_email="me@joseph-long.com",
    packages=["audibleAlerts"],
    package_data={
        "audibleAlerts": ["default.xml"],
    },
    install_requires=[
        "purepyindi2>=0.0.0",
    ],
    entry_points={
        "console_scripts": [
            "audibleAlerts=audibleAlerts.core:Maggie.console_app",
        ],
    },
)
