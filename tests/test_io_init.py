import unittest

from simsoliq.io import init_mdtraj


class TestIOinit(unittest.TestCase):
    
    def test_filenotfound(self):
        with self.assertRaises(FileNotFoundError):
            init_mdtraj("data/Pt111_24H2O_x/notthere.out")

    def test_wrongformat(self):
        with self.assertRaises(NotImplementedError):
            init_mdtraj("data/Pt111_24H2O_x/OUTCAR", fmat='notarealformat')
    
    def test_wrongfileformat(self):
        with self.assertRaises(NotImplementedError):
            init_mdtraj("data/Pt111_24H2O_x/OSZICAR", fmat='vasp')


if __name__ == '__main__':
    unittest.main()
