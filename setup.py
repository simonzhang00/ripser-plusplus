#!/usr/bin/env python
# -*- coding: utf-8 -*-

import ast
import io
import re
import os
import platform
import pathlib
import struct
from setuptools import find_packages, setup, Extension
from setuptools.command.build_ext import build_ext as build_ext_orig

DEPENDENCIES = ['cmake','numpy','scipy']
CURDIR = os.path.abspath(os.path.dirname(__file__))

with io.open(os.path.join(CURDIR, "README.md"), "r", encoding="utf-8") as f:
    README = f.read()

def read_text(file_name: str):
    return open(os.path.join("",file_name)).read()

#https://stackoverflow.com/questions/42585210/extending-setuptools-extension-to-use-cmake-in-setup-py/48015772?noredirect=1#comment116786887_48015772
class CMakeExtension(Extension):

    def __init__(self, name):
        # don't invoke the original build_ext for this special extension
        if struct.calcsize("P") * 8 != 64:
            raise Exception("Requires a 64 bit architecture")
        if(platform.system() != 'Linux'):
            raise Exception("Requires Linux operating system")    
        
        super().__init__(name, sources=[])
        
class build_ext(build_ext_orig):

    def run(self):
        for ext in self.extensions:
            self.build_cmake(ext)
        super().run()

    def build_cmake(self, ext):
        
        cwd = pathlib.Path().absolute()
        ##print(cwd)

        # these dirs will be created in build_py, so if you don't have
        # any python sources to bundle, the dirs will be missing

        build_temp = pathlib.Path(self.build_temp)
        ##print(build_temp)
        build_temp.mkdir(parents=True, exist_ok=True)
        extdir = pathlib.Path(self.get_ext_fullpath(ext.name))
        extdir.mkdir(parents=True, exist_ok=True)

        # example of cmake args
        config = 'Debug' if self.debug else 'Release'
        cmake_args = [
            '-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=' + str(extdir.parent.absolute()),
            '-DCMAKE_BUILD_TYPE=' + config
        ]

        # example of build args
        build_args = [
            '--config', config,
            '--', '-j4'
        ]

        os.chdir(str(build_temp))
        self.spawn(['cmake', str(cwd)] + cmake_args)
        if not self.dry_run:
            self.spawn(['cmake', '--build', '.'] + build_args)
        # Troubleshooting: if fail on line above then delete all possible
        # temporary CMake files including "CMakeCache.txt" in top level dir.
        os.chdir(str(cwd))

setup(
    name="ripserplusplus",
    version="1.0.9",
    author="Simon Zhang - Ripser++, Birkan Gokbag, Ryan DeMilt, Simon Zhang - Python Binding",
    author_email="szhang31415@gmail.com, demilt.ryan@gmail.com, birkan.gokbag@gmail.com ",
    description="Python binding for Ripser++.",
    long_description=README,
    long_description_content_type = 'text/markdown',
    url="https://github.com/simonzhang00/ripser-plusplus",
    #map between name and dir
    #package_dir={'ripserplusplus': 'ripserplusplus'},
    #packages=['ripserplusplus'],
    ##include_package_data=True,
    ext_modules=[CMakeExtension('ripserplusplus/ripserplusplus')],
    #keywords=[],
    #scripts=[],
    packages=find_packages(),
    cmdclass=dict(build_ext=build_ext),
    zip_safe=False,
    install_requires=DEPENDENCIES,
    # license and classifier list:
    # https://pypi.org/pypi?%3Aaction=list_classifiers
    #license="License :: OSI Approved :: MIT License",
    license=read_text("LICENSE"),
    classifiers=[
        "Programming Language :: Python",
        #"Operating System :: OS Independent",
        "Topic :: System :: Operating System Kernels :: Linux",
    ],
)
