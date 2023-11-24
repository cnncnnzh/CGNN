# -*- coding: utf-8 -*-
"""
Created on Wed Nov 22 14:35:43 2023

@author: zhuhe
"""

import phonopy
from phonopy import Phonopy
from phonopy.interface.vasp import read_vasp
from phonopy.file_IO import parse_FORCE_SETS
from pymatgen.io.vasp.inputs import Poscar
from pymatgen.io.phonopy import get_phonon_band_structure_from_fc
from pymatgen.transformations import InsertSitesTransformation, TranslateSitesTransformation
from mp_api.client import MPRester

from pymatgen.io.vasp.inputs import Kpoints, Incar
from pymatgen.io.vasp.sets import MPRelaxSet
import pandas
import json
import numpy as np
import os
import re
import copy

def set_VASP(docs, root, n):
    for d in docs:
        if d.nsites <= 5:
            mp_id, structure = str(d.material_id), d.structure
            new_dir = os.path.join(root, mp_id)
            # create folder
            os.mkdir(new_dir)
            # write poscar
            get_distorted_structure(structure)
            structure.to(fmt='poscar', filename=os.path.join(new_dir, 'POSCAR'))
            # write KPOINTS
            kpt = Kpoints.automatic_gamma_density(d.structure, 1200)
            kpt.write_file(os.path.join(new_dir, 'KPOINTS'))
            ## write INCAR
            INCAR = MPRelaxSet(d.structure).incar
            INCAR.pop('MAGMOM')
            INCAR['ISPIN'] = 1
            INCAR['ALGO'] = 'Normal'
            INCAR['IBRION'] = 6
            INCAR['NSW'] = 1
            INCAR['EDIFF'] = min(INCAR['EDIFF'] / 10, 10**-6)
            Incar.from_dict(INCAR).write_file(os.path.join(new_dir, 'INCAR'))

def get_distorted_structure(structure):
    pass
    


MP_API_KEY = "E2wL3InSvRFPLkSldR0aF2MYsaHkI865"    

org_dir = r'D:\Dropbox\Vasp_home\Machine_learning\neg_freq'
material_ids = []
for file in os.listdir(org_dir):
    if file.startswith('mp-'):
        material_ids.append( file.split('.')[0] )
        
with MPRester(MP_API_KEY) as mpr:
    docs = mpr.summary.search(
        material_ids=material_ids,
        fields=['material_id', 'structure', 'nsites']
    )
        
        
        
        
        
        