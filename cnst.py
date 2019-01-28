# -*- coding: utf-8 -*-
"""
Created on Mon Feb  6 16:46:34 2012

@author: Sahar
"""
import os
import pdb
import re

#__all__ = ['get_drn']
    
ngt_mltp = 1.0
nv_contb_mltp = 1.0
fsyn_mltp = 1
fncwd_mltp = 1

sparsity_hf=0.25

popnoun_flg =0

flip_states_flg =0#2
flip_states_prop =.0 #**
flip_states_prop_inact =0.#.25

xlog_flg = False
avg_over_tot = True

WDL_FILE = 'wdlnas.out'
AVG_FILE = 'avgstd.out'
PATS_FILE = 'pats_149.txt'


MAINPATH ='/Users/sahar/Documents/Academics/Research/TrevesLab/wordrep-github/'
SEMPATH = 'result/neurep-ngt1.0,fsyn1.0,nvcb1.0,avgt1,fwd1.0,popn0,flp0,fprp0.00,fiact0.00,sphf0.25-1527-31Dec12/'
SYNPATH = 'result/syneurep-ngt1.0,fsyn1.0,nvcb1.0,avgt1,fwd1.0-1851-04Oct13/'#'result/synrep-ngt1.0,fsyn1.0,nvcb1.0,avgt1,fwd1.0-15:17-06Jul13/' #'result/synrep-ngt0.0,fsyn1.0,nvcb1.0,avgt1,fwd1.0-18:09-23Aug12/'
FPATH = 'result/fneurep-1928-20Oct13/' #fullrep-20:40-24Oct12/'#fullrep-19:26-22Oct12/'#fullrep-13:54-22Oct12/' #fullrep-12:13-22Oct12/' #fullrep-18:07-15Oct12/' #fullrep-12:36-16Oct12/' #'result/fullrep-18:07-15Oct12/' #fullrep-19:45-23Aug12/' #fullrep-17:51-17Sep12/' #fullrep-15:58-17Sep12/' #


NEU = 'neurep' #'neurep', 'syneurep', 'fneurep'

print '*'*30

# The files representing the BLISS word dependencies
dt = '23Aug12' #'04Oct13'
F1 = 'myFACTORS_FNCWDS_syneurep_' + '23Aug12'  + '.csv' #NEVER CHANGE THIS (FIXED IN THE NET CODE)
F2 = 'mylxcs_synfactors_matrix_' + dt + '.csv'
F3 = 'mycontwds_synfactors_matrix_' + dt + '.csv'
F4 = 'mycontwds_synfactors_' + dt + '.txt'
F4_rsc = 'mycontwds_synfactors_' + dt + '_rscl.txt'
F5_rsc = 'fnc_cnt_ord_normlzd_3Feb12_sbjwd_5Nov_10m_rscl.txt' #******
F6 = 'mycontwds_allsynfactors_' + dt + '.txt'
F7 = 'mycontwds_allsynfactors_sform_' + dt + '.txt'
F8 = 'mycontwds_allsynfactors+sform_' + dt + '.txt'
F9 = 'mycontwds_allsynfactors_fin_' + dt + '.txt'

F10 = 'myfncwds_factors_matrix_' + dt + '.csv'
F_SYNFCT = 'BLISS_factorsyn.txt'

CONTWD_CATS = ['nsg', 'npl', 'vsg', 'vpl', 'adj', 'psg', 'ppl']
FWD_CATS = ['fwd'] #['cnj', 'prep', 'aux', 'art', 'dem']
WD_CATS = CONTWD_CATS + FWD_CATS
No_CONTCATS = len(CONTWD_CATS)
NO_CATS = len(WD_CATS)
FWD_CNJ = ['thatc']
FWD_PREP = ['of', 'in', 'with', 'on', 'to', 'for']
FWD_AUX = ["don't", "doesn't"]
FWD_ART = ['the', 'a']
FWD_DEM = ['this', 'that', 'these', 'those']



