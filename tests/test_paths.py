import importlib
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from facetad import paths

class Paths(unittest.TestCase):
    def test_manifest_relative_without_root(self):
        with patch.dict(os.environ,{},clear=True):
            p=importlib.reload(paths)
            self.assertEqual(p.manifest_path(Path('/tmp/m/list.jsonl'),'x.png'),Path('/tmp/m/x.png').resolve())
    def test_root_and_absolute(self):
        with patch.dict(os.environ,{'FACETAD_DATA_ROOT':'/tmp/data'},clear=True):
            p=importlib.reload(paths)
            self.assertEqual(p.manifest_path(Path('/tmp/m/list.jsonl'),'x.png'),Path('/tmp/data/x.png').resolve())
            self.assertEqual(p.manifest_path(Path('/tmp/m/list.jsonl'),'/elsewhere/x.png'),Path('/elsewhere/x.png').resolve())
    def test_new_path_precedes_legacy(self):
        with patch.dict(os.environ,{'FACETAD_WEIGHTS_ROOT':'/tmp/new','R1_WEIGHTS':'/tmp/old'},clear=True):
            self.assertEqual(importlib.reload(paths).WEIGHTS,Path('/tmp/new').resolve())
        with patch.dict(os.environ,{'R1_WEIGHTS':'/tmp/old'},clear=True):
            self.assertEqual(importlib.reload(paths).WEIGHTS,Path('/tmp/old').resolve())

if __name__=='__main__': unittest.main()
