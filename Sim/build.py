#!/usr/bin/python3
from distutils.core import setup, Extension

module1 = Extension('c_exp_direct',
                    sources = ['c_exp_direct.c'])

setup (name = 'c_exp_direct',
       version = '1.0',
       description = 'c cexp wrapper',
       ext_modules = [module1])