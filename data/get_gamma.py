# -*- coding: utf-8 -*-
"""
Created on Mon May  1 16:40:54 2023

@author: zhuhe

"""

import phonopy
from phonopy import Phonopy
from phonopy.interface.vasp import read_vasp
from phonopy.file_IO import parse_FORCE_SETS
from pymatgen.io.vasp.inputs import Poscar
from pymatgen.io.phonopy import get_phonon_band_structure_from_fc
from mp_api.client import MPRester
import pandas

import json
import numpy as np
import os
import re
import copy

def convert_to_float(frac_str):
    try:
        return float(frac_str)
    except ValueError:
        try:
            num, denom = frac_str.split('/')
        except ValueError:
            return None
        try:
            leading, num = num.split(' ')
        except ValueError:
            return float(num) / float(denom)        
        if float(leading) < 0:
            sign_mult = -1
        else:
            sign_mult = 1
        return float(leading) + sign_mult * (float(num) / float(denom))

def read_dim(directory):
    DIMPATTERN = "DIM = (\S+) (\S+) (\S+) (\S+) (\S+) (\S+) (\S+) (\S+) (\S+)"
    PRIPATTERN = "PRIMITIVE_AXIS = (\S+) (\S+) (\S+) (\S+) (\S+) (\S+) (\S+) (\S+) (\S+)"
    file = os.path.join(directory, 'phonopy.conf')
    lines = []
    with open(file) as f:
        line = f.readline()
        lines.append(line)
        while line:
            line = f.readline()
            lines.append(line)
    dims = re.findall(DIMPATTERN, lines[0])[0]
    if len(lines) == 3:
        patterns = re.findall(PRIPATTERN, lines[1])[0]
    dim_list = [
                [0,0,0],
                [0,0,0],
                [0,0,0]
               ]
    pri_list = [
                [0,0,0],
                [0,0,0],
                [0,0,0]
               ]
    for i in range(9):
        row = i // 3
        col = i % 3
        dim_list[row][col] = convert_to_float(dims[i])
        if len(lines) == 3:
            pri_list[row][col] = convert_to_float(patterns[i])
    if len(lines) == 2:
        pri_list = 'auto'
    return dim_list, pri_list

def read_Born_line(line):
    line = line.split()
    line = np.array(list(map(float, line))).reshape(3,3)
    return copy.deepcopy(line)

def parse_Born(directory):
    born_file = os.path.join(directory, 'BORN')
    if os.path.exists(born_file):
        nac_params = {'factor': 14.4}
        with open(born_file) as b:
            b.readline()
            
            # read dielectric constants
            nac_params['dielectric'] = read_Born_line(b.readline())
            
            # read born effective charge
            born = []
            line = b.readline()
            while line:
                born.append(read_Born_line(line))
                line = b.readline()
            nac_params['born'] = np.array(born)
        return nac_params
    else:
        return None
    
def gen_cif(mp_id, target_folder, MP_API_KEY, symprec):
    with MPRester(MP_API_KEY) as mpr:
        target_cif = os.path.join(target_folder, mp_id+".cif")
        if os.path.exists(target_cif):
            print('file already exists:', target_cif)
            return
        structure = mpr.get_structure_by_material_id(mp_id)
        structure.to(fmt="cif", filename=target_cif, symprec=symprec)

def get_Kyoto(directory, p_list, target_folder, MP_API_KEY, symprec=0.06):
    max_length = 0
    all_phonons = [x[0] for x in os.walk(directory)]
    # directory = r'D:\Dropbox\Vasp_home\Machine_learning on TP\Kyoto-phonopy\mp-8039-20180417'
    # directory = r'D:\Dropbox\Vasp_home\Machine_learning on TP\Kyoto-phonopy\mp-4624-20180417'
    errors = []
    for sub_directory in all_phonons[1:]:
        # print(sub_directory)
        
        # generate cif file
        mp_id = re.findall(r'\S+(mp-\d+)-20180417', sub_directory)[0]
        print(mp_id)
        gen_cif(mp_id, target_folder, MP_API_KEY, symprec)
        
        try:
            unitcell = read_vasp(os.path.join(sub_directory, 'POSCAR-unitcell'))
        except:
            errors.append(mp_id)
            continue
        
        dim, primitive = read_dim(sub_directory)
        # nac_params = parse_Born(sub_directory)
        force_sets = parse_FORCE_SETS(filename=os.path.join(sub_directory, 'FORCE_SETS'))
        # phonon = phonopy.load(
        #                       os.path.join(sub_directory, 'POSCAR-unitcell.yaml'),
        #                       dim,
        #                       primitive_matrix=primitive,
        #                       born_filename = os.path.join(sub_directory, 'BORN'),
        #                       force_sets_filename=os.path.join(sub_directory, 'FORCE_SETS')
        #                       )
        
        # generate Phonopy object
        # Unable to read BORN effective charge, why? 
        phonon = Phonopy(
                unitcell,
                dim,
                primitive_matrix=primitive,
                # nac_params=nac_params,
            )
        phonon.dataset = force_sets

        # generate force constants
        phonon.produce_force_constants()
        force_constants = phonon.force_constants
        phonon._set_dynamical_matrix()
        phonon.set_force_constants(force_constants)
        qpoint = np.array([[0,0,0]])
        freqs = list(phonon.get_frequencies(qpoint))
        # structure = Poscar.from_file(os.path.join(sub_directory, 'POSCAR-unitcell')).structure
        # bs = get_phonon_band_structure_from_fc(
        #     structure,
        #     dim,
        #     force_constants,
        #     primitive_matrix=primitive,
        #     # nac_params=nac_params
        #     )
        
        # # get Gamma point frequencies
        # print(bs.qpoints[0].frac_coords)
        # freqs = list(bs.bands[0])
        # max_length = max(len(freqs), max_length)
        
        row = [mp_id] + freqs
        p_list.append(copy.deepcopy(row))
        
        # except:
        #     print("Reading mp unsuccessfully:" ,sub_directory)
        #     continue
    
    return p_list, max_length

def get_Guido(directory, p_dict):
    all_phonons = os.listdir(directory)
    for p in all_phonons:
        with open(os.path.join(directory, p), 'r') as f:
            data = json.load(f)
            

def main():
    # SPECIFY YOUR MP_API_KEY
    MP_API_KEY = "YOUR_API_KEY"    
    # SPECIFY THE KYOTO DATA PATH  
    directory = r'D:\Dropbox\Vasp_home\Machine_learning\Kyoto-phonopy'
    # SPECIFY YOUR TARGET PATH
    target_folder = r'D:\Dropbox\Vasp_home\Machine_learning\deeperGATGNN\test_data' 
    
    symprec=0.06
    p_list = []

    p_list, max_length = get_Kyoto(directory, p_list, target_folder, MP_API_KEY, symprec=symprec)
    pd_frame = pandas.DataFrame(p_list)
    pd_frame = pd_frame.fillna(0)
    pd_frame.to_csv(os.path.join(target_folder, "target.csv"))
    