import pytest
import sys
import subprocess
import os

def test_runexamples():
    s = subprocess.run(['python', 'main.py', 'example_param_file'], text=True, capture_output=True)
    assert s.returncode == 1, s.stdout + s.stderr
    assert s.stderr == '', s.stderr
    s = subprocess.run(['python', 'main.py', 'example_param_file2'], text=True, capture_output=True)
    assert s.returncode == 1, s.stdout + s.stderr
    assert s.stderr == '', s.stderr
    s = subprocess.run(['python', 'main.py', 'example_param_file3'], text=True, capture_output=True)
    assert s.returncode == 1, s.stdout + s.stderr
    assert s.stderr == '', s.stderr
    