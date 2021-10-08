import pytest
import subprocess
import os
import shutil


def test_runexamples():
    os.chdir('./tests')
    os.mkdir('./temp_data')
    s = subprocess.run(['python', '../main.py', 'example_param_file'], text=True, capture_output=True)
    assert s.returncode == 0, s.stderr
    with open('./temp_data/logs/report.txt') as f:
        rd = f.read().split('\n')
        assert len([x for x in rd if 'Optimization Status: Successful' in x]) == 4, rd
        assert len([x for x in rd if 'Optimization Status: Failed' in x]) == 2, rd
