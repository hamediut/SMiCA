"""
Micro_GUI - Microstructure Analysis GUI Application

A professional GUI application for analyzing microstructure images using
correlation functions and statistical analysis methods.
"""

__version__ = '0.3.0'
__author__ = 'Your Name'
__email__ = 'your.email@example.com'

from .analysis.smds import calculate_s2, calculate_s2_3d

__all__ = ['calculate_s2', 'calculate_s2_3d']
