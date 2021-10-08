import pytest
import sys
import subprocess
import os
import shutil


def test_analyzer():
    os.chdir('./tests')
    os.mkdir('./temp_data')
    sys.path = ['..'] + sys.path
    import analyzer
    data = [f'./test_data/output{x}' for x in range(1, 4)]
    ans = analyzer.DefaultAnalysis(data, ['ma', 'mb'],
                                   ['GaussianEMRestraint_None', 'CrossLinkingMassSpectrometryRestraint_Distance_'],
                                   './temp_data')
    assert ans[0]
    assert all([(x in ans[2]) for x in ['ma', 'mb']])
    assert all([(ans[2][x][1] < 1e-8) for x in ['ma', 'mb']])
    shutil.rmtree('./temp_data')
    os.chdir('..')


def test_runexamples():
    os.chdir('./tests')
    os.mkdir('./temp_data')
    sys.path = ['..'] + sys.path
    import main
    main.main(['', 'example_param_file'])
    assert os.path.isfile('./temp_data/logs/report.txt')
    with open('./temp_data/logs/report.txt') as f:
        rd = f.read().split('\n')
        assert len([x for x in rd if 'Optimization Status: Successful' in x]) == 4, rd
        assert len([x for x in rd if 'Optimization Status: Failed' in x]) == 2, rd
    shutil.rmtree('./temp_data')
    os.mkdir('./temp_data')
    main.main(['', 'example_param_file2'])
    assert os.path.isfile('./temp_data/logs/report.txt')
    with open('./temp_data/logs/report.txt') as f:
        rd = f.read().split('\n')
        assert len([x for x in rd if 'Optimization Status: Successful' in x]) == 2, rd
    shutil.rmtree('./temp_data')
    os.chdir('..')
