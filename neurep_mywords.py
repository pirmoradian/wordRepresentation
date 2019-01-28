# -*- coding: utf-8 -*-
"""
Created on Tue Jul  5 16:25:50 2011

@author: Sahar

Containing all functions relevant to the process of neural representation of
BLISS words in the Potts network
"""
import pdb
import matplotlib
#matplotlib.use('Agg')
import blissutil as bu
import random
import copy
import itertools
import matplotlib.pyplot as plt
import math
import prjUtil as pu
import re
import os
import blissplot
from cnst import *
import numpy as np
__all__ = ['extr_myfeats_fr_mcraedb', 'neurep_feats', 'read_featcorr_mcraefile', 'read_concscorr_mcraefile','neurep_nouns', 'combinations','get_drn']

def extr_myfeats_fr_mcraedb(myfeats_infile='myFEATS_uniq.txt',
                            mcraedb_feats_file='FEATS_brm.csv',
                            feats_place=None, myfeats_outfile='out.txt'):
    '''
    Return the full record of my features extracted from McRae database
    
    -------------------------
    ##>>> import neurep_mywords
    ##>>> neurep_mywords.extr_myfeats_fr_mcraedb(myfeats_infile='myFEATS_uniq.txt', mcraedb_feats_file='FEATS_brm.csv', feats_place=[0],myfeats_outfile='myFEATS_brm.csv')
                                    
    ##>>> sum(1 for line in open('myFEATS_brm.csv'))
    190
    
    param1: myfeats_infile -- e.g. myFEATS_uniq.txt:
        a_baby_cow
        an_animal
        ... (up to 190 records)
    param2: mcraedb_feats_file -- e.g. FEATS_brm.csv:
        a_baby_cow	1\tD\t1.000\t10\t3\ttaxonomic\ta baby cow
        ... (up to 2500)
    param3: feats_place = feats_place or []:
        The features are placed as the 0th element of mcraedb_feats_file
    param3: feats_place = feats_place or []:
        The features are placed as the 0th element of mcraedb_feats_file
   
    param4: myfeats_outfile -- e.g. myFEATS_brm.csv:
            a_baby_cow	1\tD\t1.000\t10\t3\ttaxonomic\ta baby cow
        ... (up to 190)
    return: nothing
    -------------------------
    ##>>> neurep_mywords.extr_myfeats_fr_mcraedb(myfeats_infile='myFEATS_uniq.txt', mcraedb_feats_file='CorrelatedPairs.csv', feats_place=[2,5],myfeats_outfile='myCorrelatedPairs.csv')
    ##>>> sum(1 for line in open('myCorrelatedPairs.csv'))
    554
    
    param2: mcraedb_feats_file = CorrelatedPairs.csv:
        0.65\t41.91\tan_animal\ttaxonomic\tsuperordinate\ta_mammal\ttaxonomic
        \tsuperordinate\t42
        ... (up to 6000)
    param4: myfeats_outfile = myCorrelatedPairs.csv:
        0.65\t41.91\tan_animal\ttaxonomic\tsuperordinate\ta_mammal\ttaxonomic
        \tsuperordinate\t42
        ... (up to 562 pairs comparison)
    return: nothing
    TODO: convert xls to csv automatically    
    '''
    feats_place = feats_place or []    
    
    myfile = open(myfeats_infile, 'rU')
    myfeats_l = myfile.read().splitlines()
    
    mcraefile = open(mcraedb_feats_file, 'rU')
    
    myfile_out = open(myfeats_outfile, 'w')
        
    for line in mcraefile:
        line_sp = line.split('\t')
        # to check if my features are on the relevant place in this line of db
        prop_l = [(line_sp[f_p] in myfeats_l) for f_p in feats_place]
        if not (prop_l.__contains__(False)):
            myfile_out.write(line)
    
    
    myfile_out.close()

def neurep_feats(n_units=541, n_states=7, feats_infile='myFEATS_brm.csv',
                 corr_infile='myCorrelatedPairs_avgshifted.csv',
                feats_neurep_outfile='myFEATS_neurep.csv'):
    '''
    Return neural representations of my features, influenced by other feats reps
    -------------------------
    >>> import neurep_mywords

    %corr_infile='myCorrelatedPairs_avgshifted.csv'
    >>> neurep_mywords.neurep_feats(n_units=541, n_states=7, feats_infile='myFEATS_brm.csv', corr_infile='myCorrelatedPairs_avgshifted.csv',feats_neurep_outfile='myFEATS_neurep.csv')
    -------------------------
    >>> featpairs_ntog_d['a_mammal an_animal']
    36.891459074733099    
    >>> feats_cpf_srt_ltup[0]
    ('made_of_metal', 133)
    >>> feats_cpf_srt_ltup[192]    
    ('worn_by_kings', 1)u
    -------------------------
    param1: n_units number of units with which a feature is represented
    param2: n_states number of states each unit has
    param3: feats_infile -- e.g. myFEATS_brm.csv:
        a_baby_cow	1\tD\t1.000\t10\t3\ttaxonomic\ta baby cow
        ... (up to 190)
    param4: corr_infile -- e.g. 'myCorrelatedPairs.csv':
        0.65\t41.91\tan_animal\ttaxonomic\tsuperordinate\ta_mammal\ttaxonomic
        \tsuperordinate\t42
        ... (up to 562 pairs comparison)
    param5: feats_neurep_outfile -- e.g. 'myFEATS_neurep.csv':
        made_of_metal\t0 3 5 0 1 2...
        ... (up to 190)
    return: nothing
    '''
    # read feats_infile
    feats_cpf_d = {}
    f_in = open(feats_infile, 'rU')
    for line in f_in.readlines():
        line_sp = line.split('\t')
        feats_cpf_d[line_sp[0]] = int(line_sp[1])
    feats_cpf_srt_ltup = bu.UtilDict().sort_dictionary(d=feats_cpf_d, 
                                                           sort='value')
                                                           
    # read corr_infile
    featpairs_ntog_d = read_featcorr_mcraefile(corr_infile, feats_place=[2, 5],
                                               cor_place=8)
    
    ########### representation of features ###########
    feats_rep_d = {}
    file_out = open(feats_neurep_outfile, 'w')
    for curfeat, cpf in feats_cpf_srt_ltup:
        #pdb.set_trace()        
        curfeat_neurep_l = [0] * n_units
        # TEST
        # cpf = 3
        # TEST
        # assigning states to the units of a feature
        #smp_unit_id = random.sample(range(n_units), cpf)
        addtv = 33
        mltp = 3
        smp_unit_id = random.sample(range(n_units), (addtv+cpf)*mltp)
        for unit_id in smp_unit_id:
            curfeat_fields_aunit_l = get_flds_stats_aunit(curfeat, unit_id,  
                                                       n_states, 
                                                       featpairs_ntog_d, 
                                                       feats_rep_d)
            # choosing the state with maximum field
            if max(curfeat_fields_aunit_l):
                curfeat_neurep_l[unit_id] = curfeat_fields_aunit_l.index(max(
                                                        curfeat_fields_aunit_l))
            # choosing a state (excluding the zero state) with nonnegative field
            elif min(curfeat_fields_aunit_l) < 0:
                zfld_idxs = [idx for idx, item 
                            in enumerate(curfeat_fields_aunit_l) 
                            if not item]                
                zfld_idxs.remove(0)
                if not zfld_idxs:
                    #pdb.set_trace()
                    print 'WARNING:The zero state chosen due to negative fields'
                    #SAHAR zfld_idxs = [0]
                    temp = [i for i in curfeat_fields_aunit_l]
                    temp.remove(0)
                    zfld_idxs = [curfeat_fields_aunit_l.index(max(temp))]    
                curfeat_neurep_l[unit_id] = random.choice(zfld_idxs)
            # choosing a state randomly
            else:
                curfeat_neurep_l[unit_id] = random.randint(1, n_states)
        # saving the neu representation of a feature in a dictionary
        feats_rep_d[curfeat] = curfeat_neurep_l
        #pdb.set_trace()
        # writing the representation of features into a file


        curfeat_neurep_l = [str(state) for state in curfeat_neurep_l] 
        neurep_s = ' '.join(curfeat_neurep_l)
        file_out.write(curfeat + '\t' + neurep_s + '\n')
    
    file_out.close()
def read_featcorr_mcraefile(corr_infile='myCorrelatedPairs_avgshifted.csv',
                            feats_place=[0, 1], cor_place=2, sp='\t', *args, **kwds):
    '''
    Return a dictionary whose keys are pair of features and whose values are
    the correlation (Ntog) of every pair in the Mcrae database.
    -------------------------
    >>> neurep_mywords.read_featcorr_mcraefile(corr_infile='myCorrelatedPairs_avgshifted.csv', feats_place=[2, 5], cor_place=8)
    >>> d['a_mammal an_animal']
    36.891459074733099
    -------------------------
    >>> neurep_mywords.read_featcorr_mcraefile(corr_infile='myCONCS_FEATS_concstats_brm.csv', feats_place=[0, 1], cor_place=11)
    >>> d['calf an_animal']
    0.01
    -------------------------
    >>> neurep_mywords.read_featcorr_mcraefile(corr_infile='nvb_sbjwd_5Nov_10m.txt', feats_place=[0, 1], cor_place=2)
    -------------------------
    param1: corr_infile -- e.g. 'myCorrelatedPairs_avgshifted.csv':
        0.65\t41.91\tan_animal\ttaxonomic\tsuperordinate\ta_mammal\ttaxonomic
        \tsuperordinate\t42
        ... (up to 562 pairs comparison)
    param2: feats_place place of features in corr_infile
    	    e.g. feats_place = [2, 5]
    param3: cor_place place of the correlation of a feature pair in corr_infile
    return: featcorrs_d = {'a_mammal an_animal':36.89, 
                                'an_animal is_brown':8, ...}    
    '''
    #pdb.set_trace()
    featcorrs_d = {}
    f_in = open(corr_infile, 'rU')
    for line in f_in.readlines():
        line = line.rstrip()
        line_sp = line.split(sp)
        ft_l = [line_sp[f_p] for f_p in feats_place]
        feat_pair = ' '.join(ft_l)
        featcorrs_d[feat_pair] = float(line_sp[cor_place])
        
    return featcorrs_d
    
def neurep_nouns(n_units=541, n_states=7, sparsity=0.25, 
                 nouns_infile='BLISS_nouns_sg.csv',
                 concs_feats_infile='myCONCS_FEATS_concstats_brm.csv',
                 feats_neurep_infile='myFEATS_neurep.csv',
	       feats_place=[0,1], cor_place=11, noise_max=0.00001, n_hf=10, 
                 colored_noise=False, sparsity_hf=0.25,
                 nouns_neurep_outfile='myNOUNS_neurep.csv', modfy_cntb=True, *args, **kwds):
    '''
    Return neural representations of my nouns, influenced by features rep
    -------------------------
    >>> from neurep_mywords import *
    >>> neurep_mywords.neurep_nouns(n_units=541, n_states=7, sparsity=0.25, nouns_infile='BLISS_nouns_sg.txt', concs_feats_infile='myCONCS_FEATS_concstats_brm.csv', feats_neurep_infile='myFEATS_neurep.csv',  feats_place=[0,1], cor_place=11, noise_max=0.001, nouns_neurep_outfile='myNOUNS_neurep.csv')
    %>>> concsfeats_dstg_d['calf an_animal']
    0.01
     -------------------------
    >>> from neurep_mywords import *
    >>> neurep_mywords.neurep_nouns(n_units=541, n_states=7, sparsity=0.25, nouns_infile='BLISS_verbs_bs.txt', concs_feats_infile='nvb_sbjwd_5Nov_10m.txt', feats_neurep_infile='myNOUNS_neurep.csv',  feats_place=[0,1], cor_place=2, noise_max=0.000001, colored_noise=True,nouns_neurep_outfile='myVERBS_neurep.csv')
    %>>> concsfeats_dstg_d['house give']
    0.0269597991345
    -------------------------
    >>> neurep_mywords.neurep_nouns(n_units=541, n_states=7, sparsity=0.25, nouns_infile='BLISS_adjs.txt', concs_feats_infile='nadj_sbjwd_5Nov_10m.txt', feats_neurep_infile='myNOUNS_neurep.csv',  feats_place=[0,1], cor_place=2, noise_max=0.000001, nouns_neurep_outfile='myADJS_neurep.csv')
    -------------------------
    >>> neurep_mywords.neurep_nouns(n_units=359, n_states=7, sparsity=0.5, nouns_infile='BLISS_fncwds.txt', concs_feats_infile='myfncwds_factors_matrix.csv', feats_neurep_infile='myFACTORS_FNCWDS_syneurep.csv',  noise_max=0.001, nouns_neurep_outfile='myFNCWDS_syneurep.csv')
    -------------------------
    >>> neurep_mywords.neurep_nouns(n_units=359, n_states=7, sparsity=0.25, nouns_infile='BLISS_nouns_sg.txt', concs_feats_infile='fncn_sbjwd_5Nov_10m.txt', feats_neurep_infile='myFNCWDS_syneurep.csv', feats_place=[0,1], cor_place=2, noise_max=0.0000001, nouns_neurep_outfile='myNOUNS_syneurep.csv')
    -------------------------
    >>> neurep_mywords.neurep_nouns(n_units=359, n_states=7, sparsity=0.25, nouns_infile='BLISS_verbs_bs.txt', concs_feats_infile='fncvb_sbjwd_5Nov_10m.txt', feats_neurep_infile='myFNCWDS_syneurep.csv', feats_place=[0,1], cor_place=2, noise_max=0.0000001, nouns_neurep_outfile='myVERBS_syneurep.csv')
    -------------------------
    >>> neurep_mywords.neurep_nouns(n_units=359, n_states=7, sparsity=0.25, nouns_infile='BLISS_adjs.txt', concs_feats_infile='fncadj_sbjwd_5Nov_10m.txt', feats_neurep_infile='myFNCWDS_syneurep.csv', feats_place=[0,1], cor_place=2, noise_max=0.000001, nouns_neurep_outfile='myADJS_syneurep.csv')
    -------------------------
    >>> neurep_mywords.neurep_nouns(n_units=541, n_states=7, sparsity=1./12, nouns_infile='BLISS_fncwds.txt', concs_feats_infile='fncont_intgd_sbjwd_5Nov_10m.txt', feats_neurep_infile='myCONTWDS_neurep-vnoise.02.csv', feats_place=[0,1], cor_place=2, noise_max=0.00000001, nouns_neurep_outfile='myFNCWDS_intgd_neurep-vnoise.02.csv')
    -------------------------
    >>> neurep_mywords.neurep_nouns(n_units=541, n_states=7, sparsity=1./12, nouns_infile='BLISS_fncwds.txt', concs_feats_infile='fncont_sbjwd_5Nov_10m.txt', feats_neurep_infile='myCONTWDS_neurep-vnoise.02.csv', feats_place=[0,1], cor_place=2, noise_max=0.00000001, nouns_neurep_outfile='myFNCWDS_neurep-vnoise.02.csv')
    -------------------------
    >>> neurep_mywords.neurep_nouns(n_units=359, n_states=7, sparsity=0.25, nouns_infile='BLISS_nouns_sg.txt', concs_feats_infile='sgn_fncn_sbjwd_5Nov_10m.txt', feats_neurep_infile='myFNCWDSGPL_syneurep.csv', feats_place=[0,1], cor_place=2, noise_max=0.0000001, nouns_neurep_outfile='mySGNOUNS_syneurep.csv')
    -------------------------
    param1: n_units number of units with which a feature is represented
    param2: n_states number of states each unit has
    param3: sparsity proportion of active units to all units in neurep    
    param4: nouns_infile -- e.g. 'BLISS_nouns_sg.csv'
        calf\ndeer\ndog... (up to 18)    
    param5: concs_feats_infile -- e.g. 'myCONCS_FEATS_concstats_brm.csv':
        calf\tan_animal\tsuperordinate\tc\t...
        ... (up to 272)
    param4: feats_neurep_infile -- e.g. 'myFEATS_neurep.csv':
        made_of_metal\t0 3 5 0 1 2...
        ... (up to 190)
    param5: OMITED fnc_featcorr_mcraefile function which returns a dictionary containing noun-feature correlations
    param6: feats_place place of pairs of (noun, feature) in function fnc_featcorr_mcraefile
    param7: cor_place place of correlation of pairs (noun, feature) in function fnc_featcorr_mcraefile
    param8: noise_max maximum noise of fields assoicated to states of units
    param9: nouns_neurep_outfile -- e.g. 'myNOUNS_neurep.csv':
        calf\t0 3 5 0 1 2...
        ... (up to 18)
    return: nothing
    '''
    concsfeats_dstg_d_org = filereader_factory('unknown',
                                concs_feats_infile, 
                                feats_place=feats_place, cor_place=cor_place, *args, **kwds)
#    f1=open('f1','w')    
#    for item in concsfeats_dstg_d_org.items(): 
#        f1.write(item.__repr__()+'\n')
#    pdb.set_trace()
    if (modfy_cntb):
        concsfeats_dstg_d_org = modify_contrb(concsfeats_dstg_d_org)
#    f2=open('f2','w')    
#    for item in concsfeats_dstg_d_org.items(): 
#        f2.write(item.__repr__()+'\n')       
#    pdb.set_trace()
    feats_rep_d_org = read_neurep_file(neurep_infile=feats_neurep_infile)
    nouns_l = open(nouns_infile, 'rU').readlines()
    nouns_l = [noun.rstrip() for noun in nouns_l]    
    ########### representation of nouns ###########
    nouns_rep_d = {}
    file_out = open(nouns_neurep_outfile, 'w')
#CTRL+4===============================================================================
#    pdb.set_trace()        
#===============================================================================
    for curnoun in nouns_l:
        print 'Generating the representation of', curnoun
#        pdb.set_trace()        
        curnoun_neurep_l = [0] * n_units
        curnoun_mxflds_l = [0] * n_units
        if (colored_noise):
            # adding a new list of factors and their contbs., including both hidden and visible factors
            feats_rep_d, concsfeats_dstg_d = add_colorednoise(feats_rep_d_org, 
                                            concsfeats_dstg_d_org, 
                                            n_hf, n_units=n_units, n_states=n_states, sparsity=sparsity_hf) 
            noise_max = 0
        # assigning states to the units of a noun
        for unit_id in range(n_units):
#            pdb.set_trace()
            curnoun_fields_aunit_l = get_flds_stats_aunit(curnoun, unit_id,
                                                       n_states, 
                                                       concsfeats_dstg_d, 
                                                       feats_rep_d)
            # adding white noise to the field
            curnoun_fields_aunit_l = [fld + random.random() * noise_max 
                                      for fld in curnoun_fields_aunit_l]
            # to assure state 0 won't have max field
            curnoun_fields_aunit_l[0] = 0
            # choosing the state with maximum field
            mxfld = max(curnoun_fields_aunit_l)     
            curnoun_mxflds_l[unit_id] = mxfld

            curnoun_neurep_l[unit_id] = random.choice([idx for idx, f in enumerate(curnoun_fields_aunit_l) if f == mxfld])
                    
#            mxfld_idx_l = [idx for idx, f in enumerate(curnoun_fields_aunit_l) if f == mxfld]            
#            curnoun_neurep_l[unit_id] = curnoun_fields_aunit_l.index(mxfld)
#            print curnoun_fields_aunit_l.index(mxfld)
#        pdb.set_trace() 
        curnoun_neurep_l = sparse_neurep_withmxflds(sparsity, curnoun_neurep_l,
                                                    curnoun_mxflds_l, wd=curnoun)
        # saving the neu representation of a noun in a dictionary
        nouns_rep_d[curnoun] = curnoun_neurep_l
        
        # writing the representation of features into a file
        curnoun_neurep_l = [str(state) for state in curnoun_neurep_l]
        neurep_s = ' '.join(curnoun_neurep_l)
        file_out.write(curnoun + '\t' + neurep_s + '\n')
    
    file_out.close()

def flip_actvstates(filename, outfile, n_states=7, sparsity=0.25,proportion=.2):
    d = filereader_factory('neurep', filename)
    for k, v in d.iteritems():
        no_units = len(v)
        no_flips = int(proportion * no_units * sparsity)
        idx_actunt = [idx for idx, item in enumerate(v) if item>0]        
#        print k        
        for i in range(no_flips):
#            pdb.set_trace()
            idx_flipunt = random.choice(idx_actunt)
            oldst = v[idx_flipunt]
            validsts = range(1,n_states+1)
#            print i,oldst,idx_flipunt,validsts
            validsts.remove(oldst)
            newst = random.choice(validsts)
            d[k][idx_flipunt] = newst
            idx_actunt.remove(idx_flipunt)
    filewriter_factory('neurep', outfile, d)

def flip_allstates(filename, outfile, n_states=7, sparsity=0.25,proportion=.2,propinact=.25):
    d = filereader_factory('neurep', filename)
    for k, v in d.iteritems():
        no_units = len(v)
        no_flips = int(proportion * no_units)
        MaxNumAct = int(no_units * sparsity)
        no_flips_inact = min(int(propinact*no_flips),MaxNumAct)
        no_flips_actchgst = min(.5*no_flips, MaxNumAct-no_flips_inact)
#        idx_chosenunts = random.sample(range(no_units), no_flips)
#        idx_inactunts = [idx for idx in idx_chosenunts if v[idx]==0]
#        no_flips = no_flips - no_inact
#        idx_actunts = list(set(idx_chosenunts) - set(idx_inactunts))
#        print k        
#        no_inact = len(idx_inactunts)
        idx_inactunts = [idx for idx,item in  enumerate(v) if item==0]
        idx_actunts = list(set(range(no_units)) - set(idx_inactunts))        
        idx_tobeact = random.sample(idx_inactunts, no_flips_inact)
                 
        for i in idx_tobeact:
            d[k][i] = random.choice(range(1, n_states+1))
        
        idx_tobeinact = random.sample(idx_actunts, no_flips_inact)
        for i in idx_tobeinact:
            d[k][i] = 0
        
        idx_actunts_chgst = list(set(idx_actunts) - set(idx_tobeinact))        
        for i in range(no_flips_actchgst):
#            pdb.set_trace()
            idx_flipunt = random.choice(idx_actunts_chgst)
            oldst = v[idx_flipunt]
            validsts = range(1,n_states+1)
#            print i,oldst,idx_flipunt,validsts
            validsts.remove(oldst)
            newst = random.choice(validsts)
            d[k][idx_flipunt] = newst
            idx_actunts_chgst.remove(idx_flipunt)
    filewriter_factory('neurep', outfile, d)
    
     
def add_colorednoise(vf_neurep_d_org, vf_jprob_d_org, n_hf, *args, **kwds):
    '''    
    returns a new list of factors, including both visible and hidden factors
    returns a new list of factors contbs., including the contb. of hidden factors, beside visible factors               
    '''    
    # visible factors' neurep
    vf_neurep_d = copy.deepcopy(vf_neurep_d_org)
    # visible factors' contb
    vf_jprob_d = copy.deepcopy(vf_jprob_d_org)
    vf_neurep_tup = vf_neurep_d.items()
    # no. of visible factors (vf)
    n_vf = len(vf_neurep_tup) 
    # creating hidden factors whose contb. randomly takes the contb. value of a visible factor  
    for idx in range(n_hf):
        # name of a new hidden factor (hf)
        hf_nm = 'hf' + str(idx)
        # adding the neurep of new hf to the factors list
        vf_neurep_d[hf_nm] = create_aneurep(alg='random',*args, **kwds)
        # selecting randomly a vf whose contb. is going to be the contb. of the new hf.        
        selvf_idx = random.choice(range(n_vf))
        # the name of the selected vf       
        selvf_nm = vf_neurep_tup[selvf_idx][0]
        # the new hf is gonna appear wherever the selected vf have any contb. to other words
        for pair, fq in vf_jprob_d_org.iteritems():
            w1, w2 = pair.split()
            if (selvf_nm == w1):
                new_pair = hf_nm + ' ' + w2
                vf_jprob_d[new_pair] = fq
            elif (selvf_nm == w2):
                new_pair = w1 + ' ' + hf_nm
                vf_jprob_d[new_pair] = fq
    # returns a new list of factors, including both visible and hidden factors
    # returns a new list of factors contbs., including the contb. of hidden factors, beside visible factors               
    return vf_neurep_d, vf_jprob_d
    

def neurep_factors_fncwds(n_units=359, n_states=7, sparsity=.25, 
                          factor_rep_outfile='myFACTORS_FNCWDS_syneurep.csv'):
    '''
    Generate and Store neural representation of all factors of function words
    >>> neurep_mywords.neurep_factors_fncwds(n_units=359, n_states=7, sparsity=.25, factor_rep_outfile='myFACTORS_FNCWDS_syneurep.csv')    
    '''
    allfactors = []
    #lxc/n:lexicalCategory/noun
    lxcn = create_aneurep(n_units, n_states, sparsity, alg='random')
    allfactors.append(('lxc/n', lxcn))
    #lxc/v:lexicalCategory/verb
    lxcv = create_aneurep(n_units, n_states, sparsity, alg='family', 
                        family_mems=[lxcn])
    allfactors.append(('lxc/v', lxcv))
    #lxc/aj:lexicalCategory/adj
    lxcaj = create_aneurep(n_units, n_states, sparsity, alg='family', 
                        family_mems=[lxcn, lxcv])
    allfactors.append(('lxc/aj', lxcaj))
    #lxc/conj:lexicalCategory/conjunction
    lxconj = create_aneurep(n_units, n_states, sparsity, alg='family', 
                        family_mems=[lxcn, lxcv, lxcaj])
    allfactors.append(('lxc/conj', lxconj))
    #lxc/prep:lexicalCategory/preposition
    lxcprep = create_aneurep(n_units, n_states, sparsity, alg='family', 
                        family_mems=[lxcn, lxcv, lxcaj, lxconj])
    allfactors.append(('lxc/prep', lxcprep))
    #lxc/pron:lexicalCategory/pronoun
    lxcpron = create_aneurep(n_units, n_states, sparsity, alg='family', 
                        family_mems=[lxcn, lxcv, lxcaj, lxconj, lxcprep])
    allfactors.append(('lxc/pron', lxcpron))
    #lxc/adv:lexicalCategory/adverb
    lxcadv = create_aneurep(n_units, n_states, sparsity, alg='family', 
                        family_mems=[lxcn, lxcv, lxcaj, lxconj, lxcprep, lxcpron])
    allfactors.append(('lxc/adv', lxcadv))
    
    #num/sg: Number/singular
    numsg = create_aneurep(n_units, n_states, sparsity, alg='random')
    allfactors.append(('num/sg', numsg))
    #num/pl: Number/plural
    numpl = create_aneurep(n_units, n_states, sparsity, alg='family', 
                        family_mems=[numsg])
    allfactors.append(('num/pl', numpl))                 
    
    #neg: Negation
    neg = create_aneurep(n_units, n_states, sparsity, alg='random') 
    allfactors.append(('neg', neg))
    
    #det/indf: Determiner/indefinite
    detindf = create_aneurep(n_units, n_states, sparsity, alg='random')
    allfactors.append(('det/indf', detindf))  
    #det/def: Determiner/definite
    detdef = create_aneurep(n_units, n_states, sparsity, alg='family', 
                        family_mems=[detindf])
    allfactors.append(('det/def', detdef))
    #det/propn: Determiner/ProperNoun
    detpropn = create_aneurep(n_units, n_states, sparsity, alg='family', 
                        family_mems=[detindf, detdef])
    allfactors.append(('det/propn', detpropn))
    
    #loc/close: Location/close
    loclose = create_aneurep(n_units, n_states, sparsity, alg='random')
    allfactors.append(('loc/close', loclose))
    #loc/far: Location/far
    locfar = create_aneurep(n_units, n_states, sparsity, alg='family', 
                        family_mems=[loclose])
    allfactors.append(('loc/far', locfar))
    
    #dir/from: Direction/from
    dirfrom = create_aneurep(n_units, n_states, sparsity, alg='random')
    allfactors.append(('dir/from', dirfrom))
    #g2: Direction/towards
    dirtowards = create_aneurep(n_units, n_states, sparsity, alg='family', 
                        family_mems=[dirfrom])
    allfactors.append(('dir/towards', dirtowards))
    #g3: Direction/sameplace
    dirsameplace = create_aneurep(n_units, n_states, sparsity, alg='family', 
                        family_mems=[dirfrom, dirtowards])
    allfactors.append(('dir/sameplace', dirsameplace))
    #g4: Direction/above
    dirabove = create_aneurep(n_units, n_states, sparsity, alg='family', 
                        family_mems=[dirfrom, dirtowards, dirsameplace])
    allfactors.append(('dir/above', dirabove))
    
    file_out = open(factor_rep_outfile, 'w')
    for f_name, f_rep in allfactors:
        f_rep_l = [str(state) for state in f_rep] 
        neurep_s = ' '.join(f_rep_l)
        file_out.write(f_name + '\t' + neurep_s + '\n') 
    file_out.close()
        
def create_aneurep(n_units=541, n_states=7, sparsity=.25, alg='random', 
              family_mems=None, *args, **kwds):
    '''
    Return a neurepresentation (vector) with the given parameters, 
    choose active units and set their states according to an algorithm
    -------------------------
    >>> neurep_mywords.create_aneurep(n_units=5, n_states=7, sparsity=.8, alg='family', family_mems=[[1,0,3,0,5]])
    -------------------------    
    param4: alg an algorithm according to which active units and their states are chosen
        'random': randomly choose sparsity*n_units units and randomly select their states
        'family': active units must be the same units as in family members, however
        their states must not be the same as the ones of family members
    param5: family_mems a list whose elements are the nurep of a family member 
            family members share the same active units yet with different states
    '''
    family_mems = family_mems or [[]]
    aneurep = [0] * n_units

    if alg == 'random':
        smp_unit_id = random.sample(range(n_units), 
                                    int(round(sparsity*n_units)))
        for idx in smp_unit_id:
            aneurep[idx] = random.randint(1, n_states)
    
    elif alg == 'family':
        family_actunt_l = [idx for idx, item in enumerate(family_mems[0]) 
                           if item]
        for idx in family_actunt_l:
            fml_st_l = [fmlmem_neurep[idx] for fmlmem_neurep in family_mems]
            choice_st_l = list(set(range(1, n_states+1)) - set(fml_st_l))
            if not choice_st_l:
                raise ValueError('Running out of choice for this unit')
            aneurep[idx] = random.choice(choice_st_l)
    else:
        raise TypeError('Correct the name of alg input')
    return aneurep
    
def get_flds_stats_aunit(curfeat, unit_id, n_states, featpairs_ntog_d, 
                      feats_rep_d):
    '''
    Return external fields of states of a certain unit (q-state spin) of a feature, 
    each value of a unit copuled to a distinct external field, induced by other features.
    -------------------------
    param1: curfeat current feature whose unit is of concern
    param2: unit_id index of the concerned unit
    param3: n_state number of states each unit has
    param4: featpairs_ntog_d a dictionary containing the co-occurrences of all pair of features
    param5: feats_rep_d a dictionary containing the representaiton of history features
    return: fields_aunit_l = [0, 20, 30, 0, 4, 5, 6]    
    '''
    # including zero state: n_states + 1
    fields_aunit_l = [0] * (n_states + 1)
    # history features, determining the external fields of the unit    
    hfeats_l = feats_rep_d.keys()
    for hfeat in hfeats_l:
        pair = ' '.join([hfeat, curfeat])
        rev_pair = ' '.join([curfeat, hfeat])
        hfeat_state = get_state_aunit(hfeat, unit_id, feats_rep_d)
        # TEST
        #if pair in featpairs_ntog_d:
        #    print pair, featpairs_ntog_d[pair] 
        #elif rev_pair in featpairs_ntog_d:
        #    print rev_pair, featpairs_ntog_d[rev_pair] 
        # TEST        
        if hfeat_state and (pair in featpairs_ntog_d):
#            print pair,featpairs_ntog_d[pair], str(hfeat_state)
            fields_aunit_l[hfeat_state] += featpairs_ntog_d[pair]
        elif hfeat_state and (rev_pair in featpairs_ntog_d):
#            print rev_pair,featpairs_ntog_d[rev_pair], str(hfeat_state)
            fields_aunit_l[hfeat_state] += featpairs_ntog_d[rev_pair]
    # print fields_aunit_l
    # pdb.set_trace()
    return fields_aunit_l

def get_state_aunit(feat, unit_id, feats_rep_d):
    '''
    Return the state of a certain unit of a feature
    -------------------------
    param1: feat feature whose unit is of concern
    param2: unit_id index of the concerned unit
    return: int
    '''
    return feats_rep_d[feat][unit_id]    

def sparse_neurep_withmxflds(sparsity, neurep_l, neurep_mxflds_l, wd=''):
    '''
    Return sparse neural representation of a word, the units with maximum fields
    are likely to stay active
    -------------------------
    >>> neurep_mywords.sparse_neurep_withmxflds(sparsity=.25, neurep_l=[3, 2, 4, 1], neurep_mxflds_l=[.2, .8, .1, 0]) 
    >>> neurep_l
    [0 2 0 0]
    -------------------------
    param1: sparsity proportion of active units to all units of a rep
    param2: neurep_l neural representation which should be represented sparsely
    param3: neurep_mxflds_l maximum fields of units of a neural representation
    return: neurep_l with sparse representation     
    '''    
    slctd_unts_idx_l = []
    n_units = len(neurep_l)
    zflds = [f for f in neurep_mxflds_l if f == 0]    
    if len(zflds) > (n_units - int(round(sparsity * n_units))):
#        pdb.set_trace()
        raise ValueError(wd + ': Too Many Units (' + str(len(zflds)) + ') out of ' + str(n_units) + ' with ZERO fields!')

    # TODO: solve 'sel' warning
    for sel in range(int(round(sparsity * n_units))):
        mxfld = max(neurep_mxflds_l)
        mxfld_idx_l = [idx for idx, f in enumerate(neurep_mxflds_l) if f == mxfld]
        mxfld_idx = random.choice(mxfld_idx_l)
        slctd_unts_idx_l.append(mxfld_idx)
        neurep_mxflds_l[mxfld_idx] = 0
    for unit_id in range(n_units):
        if unit_id not in slctd_unts_idx_l:
            neurep_l[unit_id] = 0
    return neurep_l
                                                    
def get_corr_pirmoradfeats(feats_neurep_infile='myFEATS_neurep.csv',         
                	   sepfiles=False, feats_neurep_infile2='myFEATS_neurep2.csv',
			   feats_corr_outfile='myFEATS_neurep_corr.csv'): 
    '''
    Return the correlation of neural rep of all pair of features
    -------------------------
    >>> neurep_mywords.get_corr_pirmoradfeats(feats_neurep_infile='myFEATS_neurep.csv', feats_corr_outfile='myFEATS_neurep_corr.csv')
    -------------------------
    >>> neurep_mywords.get_corr_pirmoradfeats(feats_neurep_infile='myNOUNS_neurep.csv', feats_corr_outfile='myNOUNS_neurep_corr.csv')
    -------------------------
    >>> neurep_mywords.get_corr_pirmoradfeats(feats_neurep_infile='myFNCWDS_syneurep.csv', feats_corr_outfile='myFNCWDS_syneurep_corr.csv')
    -------------------------
    >>> neurep_mywords.get_corr_pirmoradfeats(feats_neurep_infile='myNOUNS_neurep.csv', sepfiles=True, feats_neurep_infile2='myVERBS_neurep.csv', feats_corr_outfile='myNOUNS_VERBS_neurep_corr.csv')
    -------------------------
    >>> neurep_mywords.get_corr_pirmoradfeats(feats_neurep_infile='myNOUNS_neurep.csv', sepfiles=True, feats_neurep_infile2='myADJS_neurep.csv', feats_corr_outfile='myNOUNS_ADJS_neurep_corr.csv')
    -------------------------
    >>> neurep_mywords.get_corr_pirmoradfeats(feats_neurep_infile='myVERBS_neurep.csv', sepfiles=True, feats_neurep_infile2='myADJS_neurep.csv', feats_corr_outfile='myVERBS_ADJS_neurep_corr.csv')
    -------------------------
    >>> neurep_mywords.get_corr_pirmoradfeats(feats_neurep_infile='myFNCWDS_syneurep.csv', sepfiles=True, feats_neurep_infile2='myFACTORS_FNCWDS_syneurep.csv', feats_corr_outfile='myFNCWDS_FACTORS_syneurep_corr.csv')
    -------------------------
    >>> neurep_mywords.get_corr_pirmoradfeats(feats_neurep_infile='myNOUNS_syneurep.csv', sepfiles=True, feats_neurep_infile2='myFNCWDS_syneurep.csv', feats_corr_outfile='myNOUNS_FNCWDS_syneurep_corr.csv')
    -------------------------
    >>> neurep_mywords.get_corr_pirmoradfeats(feats_neurep_infile='myVERBS_syneurep.csv', sepfiles=True, feats_neurep_infile2='myFNCWDS_syneurep.csv', feats_corr_outfile='myVERBS_FNCWDS_syneurep_corr.csv')
    -------------------------
    >>> neurep_mywords.get_corr_pirmoradfeats(feats_neurep_infile='myADJS_syneurep.csv', sepfiles=True, feats_neurep_infile2='myFNCWDS_syneurep.csv', feats_corr_outfile='myADJS_FNCWDS_syneurep_corr.csv')
    -------------------------
    >>> neurep_mywords.get_corr_pirmoradfeats(feats_neurep_infile='myFNCWDS_intgd_neurep.csv', sepfiles=True, feats_neurep_infile2='myCONTWDS_neurep.csv', feats_corr_outfile='myFNCWDS_CONTWDS_intgd_neurep_corr.csv')
    -------------------------
    >>> neurep_mywords.get_corr_pirmoradfeats(feats_neurep_infile='myFNCWDS_neurep-vnoise.02.csv', sepfiles=True, feats_neurep_infile2='myCONTWDS_neurep-vnoise.02.csv', feats_corr_outfile='myFNCWDS_CONTWDS_neurep_corr-vnoise.02.csv')
    -------------------------
    >>> neurep_mywords.get_corr_pirmoradfeats(feats_neurep_infile='myFNCWDS_intgd_neurep-vnoise.02.csv', sepfiles=True, feats_neurep_infile2='myCONTWDS_neurep-vnoise.02.csv', feats_corr_outfile='myFNCWDS_CONTWDS_intgd_neurep_corr-vnoise.02.csv')
    -------------------------
    >>> neurep_mywords.get_corr_pirmoradfeats(feats_neurep_infile='myNOUNS_fneurep-noisem.001syn.02.csv', sepfiles=True, feats_neurep_infile2='myVERBS_fneurep-noisem.02syn.08.csv', feats_corr_outfile='myNOUNS_VERBS_fneurep_corr.csv')
    -------------------------
    >>> neurep_mywords.get_corr_pirmoradfeats(feats_neurep_infile='myNOUNS_fneurep-noisem.001syn.02.csv', sepfiles=True, feats_neurep_infile2='myADJS_fneurep-noisem10-6syn.08.csv', feats_corr_outfile='myNOUNS_ADJS_fneurep_corr.csv')
    -------------------------
    >>> neurep_mywords.get_corr_pirmoradfeats(feats_neurep_infile='myNOUNS_fneurep-noisem.001syn.02.csv', sepfiles=True, feats_neurep_infile2='myFNCWDS_intgd_fneurep-noisem.25syn.001.csv', feats_corr_outfile='myNOUNS_FNCWDS_fneurep_corr.csv')
    -------------------------
    >>> neurep_mywords.get_corr_pirmoradfeats(feats_neurep_infile='myVERBS_fneurep-noisem.02syn.08.csv', sepfiles=True, feats_neurep_infile2='myADJS_fneurep-noisem10-6syn.08.csv', feats_corr_outfile='myVERBS_ADJS_fneurep_corr.csv')
    -------------------------
    >>> neurep_mywords.get_corr_pirmoradfeats(feats_neurep_infile='myVERBS_fneurep-noisem.02syn.08.csv', sepfiles=True, feats_neurep_infile2='myFNCWDS_intgd_fneurep-noisem.25syn.001.csv', feats_corr_outfile='myVERBS_FNCWDS_fneurep_corr.csv')
    -------------------------
    >>> neurep_mywords.get_corr_pirmoradfeats(feats_neurep_infile='myADJS_fneurep-noisem10-6syn.08.csv', sepfiles=True, feats_neurep_infile2='myFNCWDS_intgd_fneurep-noisem.25syn.001.csv', feats_corr_outfile='myADJS_FNCWDS_fneurep_corr.csv')
    -------------------------
    param1: feats_neurep_infile -- e.g. 'myFEATS_neurep.csv':
        made_of_metal\t0 3 5 0 1 2...
        ... (up to 190)
    param2: sepfiles False if the correlations between words of one file needed to be calculated
		     True if the correlations between words of two separte files needed to be calculated
    param3: feats_neurep_infile2 second file if sepfiles=True (myVERBS_neurep.csv or myNOUNS_neurep.csv)
    param4: feats_corr_outfile -- e.g. 'myFEATS_neurep_corr.csv':
        feat1 feat2\tn_act_sp n_algnd_sp n_misalgnd_sp
        a_mammal an_animal\t7 7 0 
        ... (up to 18528 pairs comparison)
    return: nothing
    
    '''
#    pdb.set_trace()    
    # reading features file
    feats_rep_d = read_neurep_file(feats_neurep_infile)    
    feats_l = feats_rep_d.keys()
#    featpairs_zip = combinations(feats_l, 2)
    featpairs_zip = [(f1, f2) for f1 in feats_l for f2 in feats_l]

    if sepfiles:
    	feats_rep_d2 = read_neurep_file(feats_neurep_infile2)
	feats_l2 = feats_rep_d2.keys()
	featpairs_zip = [(f1, f2) for f1 in feats_l for f2 in feats_l2]
    	for feat, feat_rep in feats_rep_d2.iteritems():
		feats_rep_d[feat] = feat_rep
 
    out_file = open(feats_corr_outfile, 'w')
    # comparing the neural reps of all pairs of feats
    #TODO: for feat_pair in itertools.combinations(feats_l, 2):
    for feat_pair in featpairs_zip:
        feat1 = feat_pair[0]
        f1_rep = feats_rep_d[feat1]
        
        feat2 = feat_pair[1]
        f2_rep = feats_rep_d[feat2]
        
        n_act_sp, n_algnd_sp = get_corr_apair_feats(f1_rep, f2_rep)
        n_misalgnd_sp = n_act_sp - n_algnd_sp
        
        out_file.write(feat1 + ' ' + feat2 + '\t' + str(n_act_sp) + ' ' + 
                       str(n_algnd_sp) + ' ' + str(n_misalgnd_sp) + '\n')
    
    out_file.close()

def read_neurep_file(neurep_infile='myFEATS_neurep.csv'):
    '''
    Return a dictionary corresponding to the neural rep of features or words, 
    the dictionary key is feature name or word and the dict value is its 
    neural rep
    -------------------------
    param1: neurep_infile -- e.g. 'myFEATS_neurep.csv':
        made_of_metal\t0 3 5 0 1 2...
        ... (up to 190)
    return: feats_rep_d = {'made_of_metal':[0, 3, 5, ...], ...}
    '''    
    #pdb.set_trace()
    feats_rep_d = {}
    file_in = open(neurep_infile, 'rU')
    for line in file_in:
        line_sp = line.split('\t')
        feat = line_sp[0]
        neurep_l = line_sp[1].split()
        neurep_l = [int(sp) for sp in neurep_l]
        feats_rep_d[feat] = neurep_l
    return feats_rep_d

def get_corr_apair_feats(feat1_rep, feat2_rep):
    '''
    Return the correlation of neural rep of a pair of features
    -------------------------
    param1: feat1_rep a list which is neural representation of feature1
    param2: feat2_rep a list which is neural representation of feature2
    '''
    mlt_f1_f2 = [sp_f1 * sp_f2 for sp_f1, sp_f2 in zip(feat1_rep, feat2_rep)]
    act_idxs_f12 = [idx for idx, item in enumerate(mlt_f1_f2) if item]
    
    act_sp_f1 = [feat1_rep.__getitem__(idx) for idx in act_idxs_f12]
    act_sp_f2 = [feat2_rep.__getitem__(idx) for idx in act_idxs_f12]
    
    act_sp_f12_zip = zip(act_sp_f1, act_sp_f2)
    n_act_sp = len(act_idxs_f12)
    n_algnd_sp = 0
    for sp_f1, sp_f2 in act_sp_f12_zip:
        if sp_f1 == sp_f2:
            n_algnd_sp += 1    

    return n_act_sp, n_algnd_sp
    


def get_valeqvar_pirmorad_mcrae_d(pirmorad_d, mcrae_d, 
                        categories_file='BLISS_nouns_sg.txt', root_flg=False):
    '''
    Return values (correlations) of equivalent variables (feature pairs) 
    in both pirmorad and mcrae dict 
    -------------------------
    param1: pirmorad_d -- e.g. 
            pirmorad_d['feat1 feat2'] = n_algnd_sp
            pirmorad_d['a_mammal an_animal'] = 7
            ... (up to 562 keys)                          
    param2: mcrae_d -- e.g. 
            mcrae_d['a_mammal an_animal'] = 38    
            ... (up to 18528 keys)
    return1: featpairs_l: ['a_mammal an_animal', 'made_of_metal an_animal', ...]
    return2: corr_pirmorad_l [7, ...]
    return3: corr_mcrae_l [38, ...]     
    '''
    cat_d = categorize_corrs_d(pirmorad_d, categories_file)    
    if root_flg:
        print 'HINT: ALL PLURALS WERE REPLACED BY SIGNULARS!'
        pirmorad_d = pu.get_deformed_dict(pirmorad_d)
        mcrae_d = pu.get_deformed_dict(mcrae_d)
    featpairs_l = pirmorad_d.keys()
    corr_pirmorad_l = []
    corr_mcrae_l = []
    cat_l = []
    for featpair in featpairs_l:
        #print featpair        
        # pdb.set_trace()
        # the correlation of featpair in pirmorad
        corr_pirmorad_l.append(pirmorad_d[featpair])
        # the correlation of featpair in mcrae
        fp_sp = featpair.split()
        featpair_rev = ' '.join([fp_sp[1], fp_sp[0]])
        if featpair in cat_d:
            cat_l.append(cat_d[featpair])
        elif featpair_rev in cat_d:
            cat_l.append(cat_d[featpair_rev])
            
        if featpair in mcrae_d:
            corr_mcrae_l.append(mcrae_d[featpair])
        elif featpair_rev in mcrae_d:
            corr_mcrae_l.append(mcrae_d[featpair_rev])
        else:
            corr_mcrae_l.append(0)
            
    return featpairs_l, corr_pirmorad_l, corr_mcrae_l, cat_l

   
def read_corr_pirmoradfile(corr_infile='myFEATS_neurep_corr.csv', 
                           corr_id_pirmorad=1, *args, **kwds):
    '''
    Return a dictionary whose keys are pair of features or words and whose 
    values are the correlation (N_act or N_act_same or N_act_diff) of every pair
    in pirmorad neural representation
    -------------------------
    >>> d = neurep_mywords.read_corr_pirmoradfile()
    >>> d['a_mammal an_animal']
    7    
    >>> len(d)
    18528
    -------------------------
    param1: corr_infile -- e.g. 'myFEATS_neurep_corr.csv':
        feat1 feat2\tn_act_sp n_algnd_sp n_misalgnd_sp
        a_mammal an_animal\t7 7 0 
        ... (up to 562 pairs comparison)
    param2: corr_id_pirmorad index of interest in the correlation of pirmorad
                             pair of features:
                                 0: N_activeunits i.e 7 in above example
                                 1: N_activesame i.e. 7 in above example
                                 2: N_activediff i.e 0 in above example
    return: corrs_d = {'a_mammal an_animal':7, ...}    
    '''
    corrs_d = {}
    f_in = open(corr_infile, 'rU')
    for line in f_in.readlines():
        line_sp = line.split('\t')
        corr_measure = line_sp[1].split()
        corrs_d[line_sp[0]] = int(corr_measure[corr_id_pirmorad])
        
    return corrs_d
    
def read_concscorr_mcraefile(corr_infile='mycos_matrix_brm_IFR.csv', *args, **kwds):
    '''
    Return a dictionary whose keys are pair of concepts and whose values are
    the correlation of every pair in the Mcrae database.
    -------------------------
    >>> d = neurep_mywords.read_concscorr_mcraefile(corr_infile='mycos_matrix_brm_IFR.csv')
    >>> d['calf calf']
    1
    >>> d['calf dagger']
    0.052
    -------------------------
    >>> d = neurep_mywords.read_concscorr_mcraefile(corr_infile='myfncwds_factors_matrix.csv')
    >>> d['this a1']
    1
    -------------------------
    param1: corr_infile -- e.g. 'mycos_matrix_brm_IFR.csv':
        CONCEPT calf church crown dagger ...\n
        calf 1.000 0.000 0.000 0.052 ...\n
        ... (up to 18 concepts)
    return: concscorr_d = {'calf calf':1, 'calf church':0,
                           'calf dagger':0.052, ...}    
    '''
#    pdb.set_trace()
    file_in = open(corr_infile, 'rU')
    concs_l = file_in.readline().split()
    # the word 'CONCEPT' is removed
    concs_l.pop(0)
    concscorr_d = dict()
    line_sp = file_in.readline().split()
    while line_sp:    
        # the first word which is a concept name, not corr, is removed
        conc_row = line_sp.pop(0)
        conc_row = conc_row.lower()
        for conc in concs_l:
            conc = conc.lower()
            pair = conc_row + ' ' + conc
            concscorr_d[pair] = float(line_sp.pop(0))
        
        line_sp = file_in.readline().split()
            
    file_in.close()
    return concscorr_d

def get_corr_pirmorad_mcrae(corr_infile_pirmorad='myFEATS_neurep_corr.csv',
                          corr_infile_mcrae='myCorrelatedPairs_avgshifted.csv',
                         corr_id_pirmorad=1, categories_file='BLISS_nouns_sg.txt', 
                         FLG_mod_contb=1, *args, **kwds):
    '''
    Return the correlation of all feature pairs in both mcrae database and 
    pirmorad database which contains neural rep of features
    -------------------------
    >>> neurep_mywords.get_corr_pirmorad_mcrae(corr_infile_pirmorad='myFEATS_neurep_corr.csv', corr_infile_mcrae='myCorrelatedPairs_avgshifted.csv', corr_id_pirmorad=1)    
        
    param1: corr_infile_pirmorad -- e.g. 'myFEATS_neurep_corr.csv':
            feat1 feat2\tn_act_sp n_algnd_sp n_misalgnd_sp
            a_mammal an_animal\t7 7 0 
            ... (up to 18528 pairs comparison)                          
    param2: corr_infile_mcrae -- e.g. 'myCorrelatedPairs_avgshifted.csv':                      
            0.65\t41.91\tan_animal\ttaxonomic\tsuperordinate\ta_mammal\ttaxonomic
            \tsuperordinate\t38
            ... (up to 562 pairs comparison)
    param3: corr_id_pirmorad index of interest in the correlation of pirmorad
                             pair of features:
                                 0: N_activeunits i.e 7 in above example
                                 1: N_activesame i.e. 7 in above example
                                 2: N_activediff i.e 0 in above example
    return1: featpairs_l: ['a_mammal an_animal', 'made_of_metal an_animal', ...]
    return2: corr_pirmorad_l [7, ...]
    return3: corr_mcrae_l [38, ...]
    -------------------------
    >>> neurep_mywords.get_corr_pirmorad_mcrae(corr_infile_pirmorad='myNOUNS_neurep_corr.csv', corr_infile_mcrae='mycos_matrix_brm_IFR.csv', corr_id_pirmorad=1)                    
    param1: corr_infile_pirmorad -- e.g. 'myNOUNS_neurep_corr.csv':
            noun1 noun2\tn_act_sp n_algnd_sp n_misalgnd_sp
            dagger calf\t32 6 26            
            dog calf\t67 54 13 
            ... (up to 153 pairs comparison)                          
    param2: corr_infile_mcrae -- e.g. 'mycos_matrix_brm_IFR.csv':                      
            CONCEPT calf church crown dagger ...\n
            calf 1.000 0.000 0.000 0.052 ...\n
            ... (up to 18 concepts)
    param3: corr_id_pirmorad 
    return1: featpairs_l: ['dagger calf', ...]
    return2: corr_pirmorad_l [6, ...]
    return3: corr_mcrae_l [0.052, ...]                         
        
    '''
    corrs_pirmorad_d = read_corr_pirmoradfile(corr_infile_pirmorad, 
                                              corr_id_pirmorad)
    #print corr_infile_mcrae
    #pdb.set_trace()
    corrs_mcrae_d = filereader_factory('unknown', corr_infile_mcrae, *args, **kwds)
#    corrs_mcrae_d = bu.UtilDict().normalize_d_values(corrs_mcrae_d)
    if ((not xlog_flg) and FLG_mod_contb):    
        corrs_mcrae_d = modify_contrb(corrs_mcrae_d)
    featpairs_l, corr_pirmorad_l, corr_mcrae_l, cat_l = get_valeqvar_pirmorad_mcrae_d(
                                            corrs_pirmorad_d, corrs_mcrae_d,
                                            categories_file)
    return featpairs_l, corr_pirmorad_l, corr_mcrae_l, cat_l

def create_fullneurep_file(semneurep_file='myNOUNS_neurep.csv', 
                           syneurep_file='myNOUNS_syneurep.csv', 
                           fneurep_file='myNOUNS_fneurep.csv'):
    '''
    Creates a file containing the full representation (semantic+syntactic) of words
    -------------------------
    >>> neurep_mywords.create_fullneurep_file(semneurep_file='myNOUNS_neurep.csv', syneurep_file='myNOUNSG_syneurep-noise0.25.csv', fneurep_file='myNOUNSG_fneurep-noisem.001syn.25.csv')
    >>> neurep_mywords.create_fullneurep_file(semneurep_file='myNOUNPL_neurep.csv', syneurep_file='myNOUNPL_syneurep-noise0.25.csv', fneurep_file='myNOUNPL_fneurep-noisem.001syn.25.csv')
    >>> neurep_mywords.create_fullneurep_file(semneurep_file='myVERBS_neurep-noise.02.csv', syneurep_file='myVERBPL_syneurep-noise0.75.csv', fneurep_file='myVERBPL_fneurep-noisem.02syn.75.csv')
    >>> neurep_mywords.create_fullneurep_file(semneurep_file='myVERBSG_neurep-noise.02.csv', syneurep_file='myVERBSG_syneurep-noise0.25.csv', fneurep_file='myVERBSG_fneurep-noisem.02syn.25.csv')
    >>> neurep_mywords.create_fullneurep_file(semneurep_file='myADJS_neurep.csv', syneurep_file='myADJS_syneurep-noise0.25.csv', fneurep_file='myADJS_fneurep-noisem10-6syn.25.csv')
    >>> neurep_mywords.create_fullneurep_file(semneurep_file='myFNCWDS_intgd_neurep-vnoise.02-noise.25.csv', syneurep_file='myFNCWDS_syneurep.csv', fneurep_file='myFNCWDS_intgd_fneurep-noisem.25syn.001.csv')
    >>> neurep_mywords.create_fullneurep_file(semneurep_file='myPROPNSG_intgd_neurep-noise0.75.csv', syneurep_file='myPROPNSG_syneurep-noise0.25.csv', fneurep_file='myPROPNSG_intgd_fneurep-noisem.75syn.25.csv')
    >>> neurep_mywords.create_fullneurep_file(semneurep_file='myPROPNPL_intgd_neurep-noise0.75.csv', syneurep_file='myPROPNPL_syneurep-noise0.25.csv', fneurep_file='myPROPNPL_intgd_fneurep-noisem.75syn.25.csv')
    '''
#    pdb.set_trace()
    semneurep_d = read_neurep_file(semneurep_file)
    syneurep_d = read_neurep_file(syneurep_file)
    file_out = open(fneurep_file, 'w')
    fneurep_d = {}
    
    for word, rep in semneurep_d.iteritems():
        fneurep_d[word] = rep + syneurep_d[word]
    
    for word, rep in fneurep_d.iteritems():
        rep = [str(st) for st in rep] 
        neurep_s = ' '.join(rep)
        file_out.write(word + '\t' + neurep_s + '\n')

    file_out.close()   


def create_fullneurep_nowlabfile(f1,f2,fout):
    
    f1_neu = filereader_factory('nowlab', f1)
    f2_neu = filereader_factory('nowlab', f2)
    file_out = open(fout, 'w')
    no_p = len(f1_neu)
    fneurep = []
    for i in range(no_p):
        fneurep.append(list(f1_neu[i]) + list(f2_neu[i]))
    
    for i in range(no_p):
#        fneurep[i] = [int(fneurep[i][n]) for n in fneurep[i]]
#        pdb.set_trace()
        fneurep_s = [str(int(f)) for f in fneurep[i]]
        neurep_s = ' '.join(fneurep_s)
        file_out.write(neurep_s + '\n')
    file_out.close()
    
    corfile = fout + '_corr'
#    get_avgNas_elefiles(f=fout, path='', form='scatter',save_fig=0,new_fig=1)    
    
def scatter_corr_pirmorad_mcrae(
                         corr_infile_pirmorad='myFEATS_neurep_corr.csv',
                         corr_infile_mcrae='myCorrelatedPairs_avgshifted.csv',
                         
                         corr_id_pirmorad=1, xlab='N_tog in McRae db', 
                         ylab='N_as in Pirmorad db', 
                         title='Feature Correlations', color='b', x_log=False, 
                         y_log=False, color_cats=True, 
                         categories_file='BLISS_nouns_sg.txt', FLG_mod_contb=1, *args, **kwds):
    '''
    Return scatter of the correlation of all feature pairs in mcrae vs pirmorad
    -------------------------
    >>> neurep_mywords.scatter_corr_pirmorad_mcrae()
    >>> neurep_mywords.scatter_corr_pirmorad_mcrae(corr_id_pirmorad=2,color='r',ylab='N_ad in Pirmorad db')
    -------------------------
    >>> from neurep_mywords import *    
    >>> neurep_mywords.scatter_corr_pirmorad_mcrae(corr_infile_pirmorad='myNOUNS_neurep_corr.csv', corr_infile_mcrae='mycos_matrix_brm_IFR.csv', corr_id_pirmorad=1, xlab='Corr in McRae db', ylab='N_as in Pirmorad db', title='Nouns Correlations')
    >>> neurep_mywords.scatter_corr_pirmorad_mcrae(corr_infile_pirmorad='myNOUNS_neurep_corr.csv', corr_infile_mcrae='mycos_matrix_brm_IFR.csv',  corr_id_pirmorad=2, xlab='Corr in McRae db', ylab='N_ad in Pirmorad db', title='Nouns Correlations', color='r')    
    -------------------------
    >>> neurep_mywords.scatter_corr_pirmorad_mcrae(corr_infile_pirmorad='myNOUNS_VERBS_neurep_corr.csv', corr_infile_mcrae='nvb_sbjwd_5Nov_10m.txt',  corr_id_pirmorad=1, xlab='Joint Prob in a Sbj-Verb Corpus', ylab='N_as in Pirmorad db (neurep)', title='Corr. bet. Nouns and Verbs', feats_place=[0, 1], cor_place=2)
    >>> neurep_mywords.scatter_corr_pirmorad_mcrae(corr_infile_pirmorad='myNOUNS_VERBS_neurep_corr.csv', corr_infile_mcrae='nvb_sbjwd_5Nov_10m.txt',  corr_id_pirmorad=2, xlab='Joint Prob in a Sbj-Verb Corpus', ylab='N_ad in Pirmorad db (neurep)', title='Corr. bet. Nouns and Verbs', color='r', feats_place=[0, 1], cor_place=2)
    -------------------------
    >>> neurep_mywords.scatter_corr_pirmorad_mcrae(corr_infile_pirmorad='myNOUNS_ADJS_neurep_corr.csv', corr_infile_mcrae='nadj_sbjwd_5Nov_10m.txt',  corr_id_pirmorad=1, xlab='Joint Prob in a Sbj-Verb Corpus', ylab='N_as in Pirmorad db (neurep)', title='Corr. bet. Nouns and Adjectives', feats_place=[0, 1], cor_place=2)
    >>> neurep_mywords.scatter_corr_pirmorad_mcrae(corr_infile_pirmorad='myNOUNS_ADJS_neurep_corr.csv', corr_infile_mcrae='nadj_sbjwd_5Nov_10m.txt',  corr_id_pirmorad=2, xlab='Joint Prob in a Sbj-Verb Corpus', ylab='N_ad in Pirmorad db (neurep)', title='Corr. bet. Nouns and Adjectives', color='r', feats_place=[0, 1], cor_place=2)
    -------------------------
    >>> neurep_mywords.scatter_corr_pirmorad_mcrae(corr_infile_pirmorad='myFNCWDS_FACTORS_syneurep_corr.csv', corr_infile_mcrae='myfncwds_factors_matrix.csv',  corr_id_pirmorad=1, xlab='Relevance scored by humans', ylab='N_as in Pirmorad db (neurep)', title='Relevance between Syntactic factors and function words', categories_file='BLISS_fncwds.txt')
    >>> neurep_mywords.scatter_corr_pirmorad_mcrae(corr_infile_pirmorad='myFNCWDS_FACTORS_syneurep_corr.csv', corr_infile_mcrae='myfncwds_factors_matrix.csv',  corr_id_pirmorad=2, xlab='Relevance scored by humans', ylab='N_ad in Pirmorad db (neurep)', title='Relevance between Syntactic factors and function words', categories_file='BLISS_fncwds.txt')
    -------------------------
    >>> neurep_mywords.scatter_corr_pirmorad_mcrae(corr_infile_pirmorad='myNOUNS_FNCWDS_syneurep_corr.csv', corr_infile_mcrae='fncn_sbjwd_5Nov_10m.txt',  corr_id_pirmorad=1, xlab='Joint Prob in a Sbj-Verb Corpus (log)', ylab='N_as in Pirmorad db (neurep)', title='Corr. bet. syntactic units of nouns and function words', categories_file='BLISS_fncwds.txt', x_log=True, feats_place=[0, 1], cor_place=2)
    >>> neurep_mywords.scatter_corr_pirmorad_mcrae(corr_infile_pirmorad='myNOUNS_FNCWDS_syneurep_corr.csv', corr_infile_mcrae='fncn_sbjwd_5Nov_10m.txt',  corr_id_pirmorad=2, xlab='Joint Prob in a Sbj-Verb Corpus (log)', ylab='N_ad in Pirmorad db (neurep)', title='Corr. bet. syntactic units of nouns and function words', categories_file='BLISS_fncwds.txt', x_log=True, feats_place=[0, 1], cor_place=2)
    -------------------------
    >>> neurep_mywords.scatter_corr_pirmorad_mcrae(corr_infile_pirmorad='myVERBS_FNCWDS_syneurep_corr.csv', corr_infile_mcrae='fncvb_sbjwd_5Nov_10m.txt',  corr_id_pirmorad=1, xlab='Joint Prob in a Sbj-Verb Corpus (log)', ylab='N_as in Pirmorad db (neurep)', title='Corr. bet. syntactic units of verbs and function words', categories_file='BLISS_fncwds.txt', x_log=True, feats_place=[0, 1], cor_place=2)    
    >>> neurep_mywords.scatter_corr_pirmorad_mcrae(corr_infile_pirmorad='myVERBS_FNCWDS_syneurep_corr.csv', corr_infile_mcrae='fncvb_sbjwd_5Nov_10m.txt',  corr_id_pirmorad=2, xlab='Joint Prob in a Sbj-Verb Corpus (log)', ylab='N_ad in Pirmorad db (neurep)', title='Corr. bet. syntactic units of verbs and function words', categories_file='BLISS_fncwds.txt', x_log=True, feats_place=[0, 1], cor_place=2)
    -------------------------
    >>> neurep_mywords.scatter_corr_pirmorad_mcrae(corr_infile_pirmorad='myADJS_FNCWDS_syneurep_corr.csv', corr_infile_mcrae='fncadj_sbjwd_5Nov_10m.txt',  corr_id_pirmorad=1, xlab='Joint Prob in a Sbj-Verb Corpus (log)', ylab='N_as in Pirmorad db (neurep)', title='Corr. bet. syntactic units of adjs and function words', categories_file='BLISS_fncwds.txt', x_log=True, feats_place=[0, 1], cor_place=2)    
    >>> neurep_mywords.scatter_corr_pirmorad_mcrae(corr_infile_pirmorad='myADJS_FNCWDS_syneurep_corr.csv', corr_infile_mcrae='fncadj_sbjwd_5Nov_10m.txt',  corr_id_pirmorad=2, xlab='Joint Prob in a Sbj-Verb Corpus (log)', ylab='N_as in Pirmorad db (neurep)', title='Corr. bet. syntactic units of adjs and function words', categories_file='BLISS_fncwds.txt', x_log=True, feats_place=[0, 1], cor_place=2)    
    -------------------------
    >>> neurep_mywords.scatter_corr_pirmorad_mcrae(corr_infile_pirmorad='myFNCWDS_CONTWDS_intgd_neurep_corr-vnoise.02.csv', corr_infile_mcrae='fncont_intgd_sbjwd_5Nov_10m.txt',  corr_id_pirmorad=1, xlab='Joint Prob in a Sbj-Verb Corpus (log)', ylab='N_as in Pirmorad db (neurep)', title='Corr. bet. semantic units of content and function words', categories_file='BLISS_fncwds.txt', x_log=True, feats_place=[0, 1], cor_place=2)    
    >>> neurep_mywords.scatter_corr_pirmorad_mcrae(corr_infile_pirmorad='myFNCWDS_CONTWDS_intgd_neurep_corr-vnoise.02.csv', corr_infile_mcrae='fncont_intgd_sbjwd_5Nov_10m.txt',  corr_id_pirmorad=2, xlab='Joint Prob in a Sbj-Verb Corpus (log)', ylab='N_ad in Pirmorad db (neurep)', title='Corr. bet. semantic units of content and function words', categories_file='BLISS_fncwds.txt', x_log=True, feats_place=[0, 1], cor_place=2)    
    -------------------------
    >>> neurep_mywords.scatter_corr_pirmorad_mcrae(corr_infile_pirmorad='myNOUNS_VERBS_fneurep_corr.csv', corr_infile_mcrae='nvb_sbjwd_5Nov_10m.txt',  corr_id_pirmorad=1, xlab='Joint Prob in a Sbj-Verb Corpus(log)', ylab='N_as in Pirmorad db (neurep)', title='Corr. bet. Full rep of nouns and verbs', categories_file='BLISS_nouns_sg.txt', x_log=True, feats_place=[0, 1], cor_place=2)
    >>> neurep_mywords.scatter_corr_pirmorad_mcrae(corr_infile_pirmorad='myNOUNS_ADJS_fneurep_corr.csv', corr_infile_mcrae='nadj_sbjwd_5Nov_10m.txt',  corr_id_pirmorad=1, xlab='Joint Prob in a Sbj-Verb Corpus(log)', ylab='N_as in Pirmorad db (neurep)', title='Corr. bet. semantic units of content and function words', categories_file='BLISS_fncwds.txt', x_log=True, feats_place=[0, 1], cor_place=2)    
    >>> neurep_mywords.scatter_corr_pirmorad_mcrae(corr_infile_pirmorad='myNOUNS_FNCWDS_fneurep_corr.csv', corr_infile_mcrae='fncn_sbjwd_5Nov_10m.txt',  corr_id_pirmorad=1, xlab='Joint Prob in a Sbj-Verb Corpus(log)', ylab='N_as in Pirmorad db (neurep)', title='Corr. bet. semantic units of content and function words', categories_file='BLISS_fncwds.txt', x_log=True, feats_place=[0, 1], cor_place=2)    
    >>> neurep_mywords.scatter_corr_pirmorad_mcrae(corr_infile_pirmorad='myVERBS_ADJS_fneurep_corr.csv', corr_infile_mcrae='vbadj_sbjwd_5Nov_10m.txt',  corr_id_pirmorad=1, xlab='Joint Prob in a Sbj-Verb Corpus(log)', ylab='N_as in Pirmorad db (neurep)', title='Corr. bet. semantic units of content and function words', categories_file='BLISS_fncwds.txt', x_log=True, feats_place=[0, 1], cor_place=2)    
    >>> neurep_mywords.scatter_corr_pirmorad_mcrae(corr_infile_pirmorad='myVERBS_FNCWDS_fneurep_corr.csv', corr_infile_mcrae='fncvb_sbjwd_5Nov_10m.txt',  corr_id_pirmorad=1, xlab='Joint Prob in a Sbj-Verb Corpus(log)', ylab='N_as in Pirmorad db (neurep)', title='Corr. bet. semantic units of content and function words', categories_file='BLISS_fncwds.txt', x_log=True, feats_place=[0, 1], cor_place=2)    
    >>> neurep_mywords.scatter_corr_pirmorad_mcrae(corr_infile_pirmorad='myADJS_FNCWDS_fneurep_corr.csv', corr_infile_mcrae='fncadj_sbjwd_5Nov_10m.txt',  corr_id_pirmorad=1, xlab='Joint Prob in a Sbj-Verb Corpus(log)', ylab='N_as in Pirmorad db (neurep)', title='Corr. bet. semantic units of content and function words', categories_file='BLISS_fncwds.txt', x_log=True, feats_place=[0, 1], cor_place=2)    
    -------------------------
    param: see get_corr_pirmorad_mcrae() 
    ******************************************
    How to assign colors to function words
    fncwds=open('BLISS_fncwds.txt','r').readlines()
    fncwds = [i.rstrip() for i in fncwds]
    num = len(fncwds)
    plt.scatter(range(num), [0]*num, c=range(num), s=30, cmap='jet', linewidth=.5)
    plt.xticks(arange(num),fncwds, rotation=45)    
    ******************************************
    '''
#    pdb.set_trace()
    # TODO: solve 'fp_l' warning
    plt.figure()
    fp_l, corr_pir_l, corr_mc_l, cat_l = get_corr_pirmorad_mcrae(
                                    corr_infile_pirmorad, corr_infile_mcrae,
                                    corr_id_pirmorad, categories_file, 
                                    FLG_mod_contb=FLG_mod_contb, *args, **kwds)
    # log-log
    corr_mc_l_log = []
    corr_pir_l_log = []
    # mc corr    
    for corr in sorted(corr_mc_l):
        if corr:
            minnz_corr = corr
            break
#    pdb.set_trace()
       
    # pir corr
    for corr in sorted(corr_pir_l):
        if corr:
            minnz_corr = corr
            break
    if x_log:
        for corr in corr_mc_l:
            if corr:
                corr_mc_l_log.append(math.log(corr, 2))
            else:
                corr_mc_l_log.append(math.log(minnz_corr, 2))
        corr_mc_l = corr_mc_l_log
    if y_log:
        for corr in corr_pir_l:
            if corr:
                corr_pir_l_log.append(math.log(corr, 2))
            else:
                corr_pir_l_log.append(math.log(minnz_corr, 2))
        corr_pir_l = corr_pir_l_log

    if color_cats:
        plt.scatter(corr_mc_l, corr_pir_l, c=cat_l, s=30, cmap='jet', linewidth=.5)
    else:
        plt.scatter(corr_mc_l, corr_pir_l, c=color)
    plt.rcParams.update({'xtick.labelsize': 20})
    plt.xlabel(xlab, fontsize=30)
    plt.ylabel(ylab, fontsize=30)
    plt.title(title, fontsize=14)
    plt.yticks(range(0,41,10),fontsize=20)
    xtkmx=max(corr_mc_l)+.1
#    pdb.set_trace()
    plt.xticks(np.arange(0,xtkmx,.1),fontsize=20)
    plt.subplots_adjust(bottom=0.14)
    plt.tight_layout()
    #plt.show()
    #plt.savefig('Nas_nounverbs.png')

def combinations(iterable, r):
    # combinations('ABCD', 2) --> AB AC AD BC BD CD
    # combinations(range(4), 3) --> 012 013 023 123
    pool = tuple(iterable)
    n = len(pool)
    if r > n:
        return
    indices = range(r)
    yield tuple(pool[i] for i in indices)
    while True:
        for i in reversed(range(r)):
            if indices[i] != i + n - r:
                break
        else:
            return
        indices[i] += 1
        for j in range(i+1, r):
            indices[j] = indices[j-1] + 1
        yield tuple(pool[i] for i in indices)

def categorize_corrs_d(corr_dict, categories_file = 'BLISS_nouns_sg.txt'):
    ''' 
    returns a dictionary whose keys are pairs and values are a number corresponding to a category indicated in categories_file    
    '''
    # pdb.set_trace()
    cat_l = open(categories_file, 'rU').readlines()
    cat_l = [cat.rstrip() for cat in cat_l]    
    cat_num_d = {}
    num = 0    
    for cat in cat_l:
        cat_num_d[cat] = num
        num += 1
    cat_d = {}
    for pair, corr in corr_dict.iteritems():
        pair_l = pair.split()
        for cat in cat_l:
            if cat in pair_l:
                cat_d[pair] = cat_num_d[cat]
                break
    return cat_d
def write2matrixf(outfile, d, colwds, rowds):
    out_f = open(outfile, 'w')
    out_f.write('word' + '\t' + '\t'.join(colwds) + '\n')
    for wr in rowds:
        fq_l = []
        for wc in colwds:
            p = wr + ' ' + wc
            revp = wc + ' ' + wr
            if p in d:
                fq_l.append(str(d[p]))
            elif revp in d:
                fq_l.append(str(d[revp]))
        out_f.write(wr + '\t' + '\t'.join(fq_l) + '\n')
    out_f.close()
def write2jntpbf(outfile, d):
    outf = open(outfile, 'w')
    for rndv, pb in d.iteritems():
        words = rndv.split()
        wst = '\t'.join(words)
        outf.write(wst + '\t' + str(pb) + '\n')
    outf.close()

def write2neurep(outfile, l_tup):
    outf = open(outfile, 'w')
    if type(l_tup) is dict:
        l_tup = l_tup.items()
    for k, v in l_tup:
        v = [str(state) for state in v] 
        neurep_s = ' '.join(v)
        outf.write(k + '\t' + neurep_s + '\n')
    outf.close()

def read_nowlab_file(filename):
    '''
    read pattern file
    >>> data = read_nowlab_file('pattern.dat')
    '''
    f_in = open(filename, 'rU')
    lines = f_in.readlines()
    no_p = len(lines)
    no_units = len(lines[0].split())
    data = np.zeros([no_p, no_units])
    for idx, line in enumerate(lines):
        line = line.rstrip()        
        line_sp = line.split()
        line_sp = [int(i) for i in line_sp]
#        pdb.set_trace()
        data[idx,:] = line_sp
    return data
    
def filereader_factory(type_, filename, *args, **kwds):
    ''' 
    returns the relevant reader function of filename    
    ''' 
    # pdb.set_trace()
    if (type_ == 'unknown'):
        type_ = get_filetype(filename)
    if (type_ == 'jntpb'):
#    if (filename == 'myCorrelatedPairs_avgshifted.csv' or 
#        filename == 'myCONCS_FEATS_concstats_brm.csv' or 
#        re.match('nvb\w*', filename) or 
#        re.match('nadj_\w*', filename) or 
#        re.match('vbadj_\w*', filename) or
#        re.match('fnc\w*_\w*', filename) or
#        re.match('propn\w*_\w*', filename) or
#        re.match('mycontwds_\w*synfactors\w*.txt', filename) or
#        re.match('my\w*_synfactors\+fnc.txt', filename)):
        return read_featcorr_mcraefile(filename, *args, **kwds)
    if (type_ == 'jntpb_fq0'):
        return bu.FileReader().read_words_d_file(filename) 
    if (type_ == 'cor'):
#    elif re.match('\w*_corr*\w*', filename):
        return read_corr_pirmoradfile(filename, *args, **kwds)
    if (type_ == 'matrix'):
#    elif re.match('my\w*_matrix\w*.csv', filename):
        return read_concscorr_mcraefile(filename, *args, **kwds)
    if (type_ == 'readl'): # readlines
#    elif (re.match('BLISS_\w*.txt', filename) or
#        re.match('BLISS_\w*\+\w*.txt', filename) or
#        re.match('pats_\w*', filename)):
        return [w.rstrip() for w in open(filename, 'rU').readlines()]    
    if (type_ == 'neurep'):
#    elif (re.match('\w*neurep\w*', filename)):
        return read_neurep_file(filename)

    if (type_ == 'nowlab'):
        return read_nowlab_file(filename)    
    else:
        raise ValueError('I do not know how to read your file type: ' + type_)

def get_filetype(filename):
    if (filename == 'myCorrelatedPairs_avgshifted.csv' or 
        filename == 'myCONCS_FEATS_concstats_brm.csv' or 
        re.match('nvb\w*', filename) or 
        re.match('nadj_\w*', filename) or 
        re.match('vbadj_\w*', filename) or
        re.match('fnc\w*_\w*', filename) or
        re.match('propn\w*_\w*', filename) or
        re.match('mycontwds_\w*synfactors\w*.txt', filename) or
        re.match('my\w*_synfactors\+fnc.txt', filename)):
        return 'jntpb'
    elif re.match('\w*_corr*\w*', filename):
        return 'cor'
    elif re.match('my\w*_matrix\w*.csv', filename):
        return 'matrix'
    elif (re.match('BLISS_\w*.txt', filename) or
        re.match('BLISS_\w*\+\w*.txt', filename) or
        re.match('pats_\w*', filename)):
        return 'readl'
    elif (re.match('\w*neurep\w*', filename)):
        return 'neurep'
    else:
        raise ValueError('I do not know how to read your file: ' + filename)

def filewriter_factory(type_, filename, *args, **kwds):
    if(type_ == 'matrix'):
        write2matrixf(filename, *args, **kwds)
    elif(type_ == 'jntpb'):
        write2jntpbf(filename, *args, **kwds)
    elif(type_ == 'neurep'):
        write2neurep(filename, *args, **kwds)
    else:
        raise ValueError('I do not recognize the type of your file: ' + type)

def getjntpbfile_factory(filename, *args, **kwds):
    if re.match('(myNOUN\w*_VERB\w*|myVERB\w*_NOUN\w*)', filename):
        return 'nvb_normalized_sbjwd_5Nov_10m.txt'
    elif re.match('(myNOUN\w*_ADJ\w*|myADJ\w*_NOUN\w*)', filename):
        return 'nadj_normalized_sbjwd_5Nov_10m.txt'
    elif re.match('(myNOUN\w*_FNCWD\w*|myFNCWD\w*_NOUN\w*)', filename):
        return 'fncn_normalized_sbjwd_5Nov_10m.txt'
    elif re.match('(myNOUN\w*_PROPN\w*|myPROPN\w*_NOUN\w*)', filename):
        return 'propncont_intgd_sbjwd_5Nov_10m.txt'
    elif re.match('(myVERB\w*_ADJ\w*|myADJ\w*_VERB\w*)', filename):
        return 'vbadj_normalized_sbjwd_5Nov_10m.txt'
    elif re.match('(myVERB\w*_FNCWD\w*|myFNCWD\w*_VERB\w*)', filename):
        return 'fncvb_normalized_sbjwd_5Nov_10m.txt'
    elif re.match('(myVERB\w*_PROPN\w*|myPROPN\w*_VERB\w*)', filename):
        return 'propncont_intgd_sbjwd_5Nov_10m.txt'
    elif re.match('(myPROPN\w*_FNCWD\w*|myFNCWD\w*_PROPN\w*)', filename):
        return 'fncpropn_normalized_sbjwd_5Nov_10m.txt'
    elif re.match('(myPROPN\w*_ADJ\w*|myADJ\w*_PROPN\w*)', filename):
        return 'propncont_intgd_sbjwd_5Nov_10m.txt'
    elif re.match('(myFNCWD\w*_ADJ\w*|myADJ\w*_FNCWD\w*)', filename):
        return 'fncadj_normalized_sbjwd_5Nov_10m.txt'
    else:
        print 'WARNING: No joint prob was found for the file '+filename
        return ''

def normalize_jointdict_perword(infile='fncn_sbjwd_5Nov_10m.txt', 
                                outfile='fncn_normalized_sbjwd_5Nov_10m.txt',
                                refrncfile='BLISS_fncwds+factorsyn.txt',feats_place=[0,1], cor_place=2,sp='\t'):
    '''
    Normalizes joint probs in the infile per word (verb or noun or adj)
    >>> neurep_mywords.normalize_jointdict_perword(infile='fncn_sbjwd_5Nov_10m.txt', outfile='fncn_normalized_sbjwd_5Nov_10m.txt')
    >>> neurep_mywords.normalize_jointdict_perword(infile='fncadj_sbjwd_5Nov_10m.txt', outfile='fncadj_normalized_sbjwd_5Nov_10m.txt')
    >>> neurep_mywords.normalize_jointdict_perword(infile='fncvb_sbjwd_5Nov_10m_ordered.txt', outfile='fncvb_normalized_sbjwd_5Nov_10m.txt')
    >>> neurep_mywords.normalize_jointdict_perword(infile='fncpropn_sbjwd_5Nov_10m.txt', outfile='fncpropn_normalized_sbjwd_5Nov_10m.txt')
    >>> neurep_mywords.normalize_jointdict_perword(infile='propncont_intgd_sbjwd_5Nov_10m.txt', outfile='propncont_intgd_normalized_sbjwd_5Nov_10m.txt')    
    >>> neurep_mywords.normalize_jointdict_perword(infile='nvb_sbjwd_5Nov_10m.txt', outfile='nvb_normalized_sbjwd_5Nov_10m.txt')
    >>> neurep_mywords.normalize_jointdict_perword(infile='nadj_sbjwd_5Nov_10m.txt', outfile='nadj_normalized_sbjwd_5Nov_10m.txt')
    >>> neurep_mywords.normalize_jointdict_perword(infile='vbadj_sbjwd_5Nov_10m.txt', outfile='vbadj_normalized_sbjwd_5Nov_10m.txt')
    '''
    ref_wds = filereader_factory('readl',refrncfile)
#    pdb.set_trace()
    jointprob_d = read_featcorr_mcraefile(infile, feats_place=feats_place, cor_place=cor_place,sp=sp)
    sumpb_d = {}
    outf = open(outfile, 'w')
    for pair, pb in jointprob_d.iteritems():
        w1, w2 = pair.split()
        if w1 in ref_wds:
            sumpb_d[w1] = sumpb_d.get(w1, 0) + pb
        elif w2 in ref_wds:
            sumpb_d[w2] = sumpb_d.get(w2, 0) + pb
            
    for pair, pb in jointprob_d.iteritems():
        w1, w2 = pair.split()
        normzd_pb = 0
        if w1 in ref_wds:
            if sumpb_d[w1]:
                normzd_pb = float(pb) / sumpb_d[w1]
        elif w2 in ref_wds:
            if sumpb_d[w2]:
                normzd_pb = float(pb) / sumpb_d[w2]
            
        outf.write(w1 + '\t' + w2 + '\t' + str(normzd_pb) + '\n')
    outf.close()

def get_path_prj():
    return MAINPATH
def get_path_result():
    return get_path_prj() + 'result/'
def get_path_fullfn(path,fullfn):
    return path + fullfn    
def get_fullfn(fn):
    fullfn = get_path_prj() + fn
    return fullfn

def modify_contrb(concsfeats_dstg_d_org):    
    #introducing anti-correlation
    nouns_verbs_l = filereader_factory('readl','BLISS_nouns_verbs.txt')
    nouns_l = filereader_factory('readl','BLISS_nouns_sg.txt')

    factsyn_l = filereader_factory('readl', 'BLISS_factorsyn.txt')
    fncwdfctsyn_l = filereader_factory('readl','BLISS_fncwds+factorsyn.txt')
    fncwd_l = filereader_factory('readl','BLISS_fncwds.txt')
    #pdb.set_trace()
    h=open('f1','w')
    sub_jntpbs_d = {}    
    for k,v in concsfeats_dstg_d_org.iteritems():
        w1, w2 = k.split()
        if ((w1 in nouns_verbs_l) or (w2 in nouns_verbs_l)):
            concsfeats_dstg_d_org[k] = concsfeats_dstg_d_org[k] * nv_contb_mltp
        
        if ((w1 in factsyn_l) or (w2 in factsyn_l)):
            concsfeats_dstg_d_org[k] = concsfeats_dstg_d_org[k] * fsyn_mltp

        if ((w1 in fncwd_l) or (w2 in fncwd_l)):
            concsfeats_dstg_d_org[k] = concsfeats_dstg_d_org[k] * fncwd_mltp
        
        if (w1 in fncwdfctsyn_l):
            sub_jntpbs_d[w1] = sub_jntpbs_d.get(w1,[]) + [v]
        elif (w2 in fncwdfctsyn_l):
            sub_jntpbs_d[w2] = sub_jntpbs_d.get(w2,[]) + [v]
        h.write(k + '\t' + str(v) +'\n')
    h.close()

    #    concsfeats_dstg_d_org = bu.UtilDict().normalize_d_values(concsfeats_dstg_d_org)   
    # regularize the effect of nouns for producing other words
    avg_jntpbs = np.mean(concsfeats_dstg_d_org.values())

    if popnoun_flg:
        popnouns_d = {}        
        for k,v in concsfeats_dstg_d_org.iteritems():
            w1,w2 = k.split()        
            if w1 in nouns_l:
                if v > avg_jntpbs:
                    popnouns_d[w1] = popnouns_d.get(w1, 0) + 1                   
            elif w2 in nouns_l:
                if v > avg_jntpbs:            
                    popnouns_d[w2] = popnouns_d.get(w2, 0) + 1                   
        for k,v in concsfeats_dstg_d_org.iteritems():
                w1,w2 = k.split()        
                if w1 in popnouns_d:
                        concsfeats_dstg_d_org[k] = v * 1./popnouns_d[w1]                   
                elif w2 in popnouns_d:
                        concsfeats_dstg_d_org[k] = v * 1./popnouns_d[w2]                   
        avg_jntpbs = np.mean(concsfeats_dstg_d_org.values())
        
#    for k,v in popnouns_d.iteritems():
#        print k,v
    
    h1=open('f2','w')
    if avg_over_tot:
        for k,v in concsfeats_dstg_d_org.iteritems():
#            w1,w2 = k.split()            
#            if w1 in fncwd_l:
#                concsfeats_dstg_d_org[k] = concsfeats_dstg_d_org[k] * fncwd_mltp
#            elif w2 in fncwd_l:
#                concsfeats_dstg_d_org[k] = concsfeats_dstg_d_org[k] * fncwd_mltp
            concsfeats_dstg_d_org[k] = concsfeats_dstg_d_org[k]-(avg_jntpbs*ngt_mltp)
            
            h1.write(k + '\t' + str(concsfeats_dstg_d_org[k]) +'\n')
    else:
        for k,v in concsfeats_dstg_d_org.iteritems():
            w1,w2 = k.split()
            if w1 in fncwdfctsyn_l:
                concsfeats_dstg_d_org[k] = concsfeats_dstg_d_org[k]-(np.mean(sub_jntpbs_d[w1]) * ngt_mltp)                
            elif w2 in fncwdfctsyn_l:
                concsfeats_dstg_d_org[k] = concsfeats_dstg_d_org[k]-(np.mean(sub_jntpbs_d[w2]) * ngt_mltp)
            h1.write(k + '\t' + str(concsfeats_dstg_d_org[k]) +'\n')
    h1.close()
#    pdb.set_trace()
    open(AVG_FILE,'a').write('mlt of negt. contb:' + str(ngt_mltp) + '\t' + 'mlt of n/vb:' + str(nv_contb_mltp) + '\t' + 'mlt of synfact:' + str(fsyn_mltp) + '\n')
#    print 'mlt of negt. contb:' + str(ngt_mltp) + '\t' + 'mlt of n/vb:' + str(nv_contb_mltp) + '\t' + 'mlt of synfact:' + str(fsyn_mltp)
    return concsfeats_dstg_d_org

def get_drn(neu='semrep', manual='', ngt_mltp = ngt_mltp, nv_contb_mltp = nv_contb_mltp,
            fsyn_mltp = fsyn_mltp, fncwd_mltp = fncwd_mltp, sparsity_hf=sparsity_hf,popnoun_flg =popnoun_flg,
            flip_states_flg =flip_states_flg, flip_states_prop =flip_states_prop, flip_states_prop_inact =flip_states_prop_inact,
            xlog_flg = xlog_flg, avg_over_tot = avg_over_tot):
    # get directory name
#    pdb.set_trace()
    path = 'result/'
    dirEntries = os.listdir(path)
#    pdb.set_trace()
    dr = ''
    if manual:
        dr = manual
    elif (neu == 'semrep'):
        dn1 = "%s-ngt%.1f,fsyn%.1f,nvcb%.1f,avgt%d,fwd%.1f,popn%d,flp%d,fprp%.2f,fiact%.2f,sphf%.2f*" % (neu, ngt_mltp, fsyn_mltp, nv_contb_mltp, avg_over_tot, fncwd_mltp, popnoun_flg, flip_states_flg, flip_states_prop, flip_states_prop_inact, sparsity_hf)
#        dn1 = "%s-ngt%.1f,fsyn%.1f,nvcb%.1f,avgt%d,fwd%.1f,popn%d,flp%d,fprp%.2f*" % (neu, ngt_mltp, fsyn_mltp, nv_contb_mltp, avg_over_tot, fncwd_mltp, popnoun_flg, flip_states_flg, flip_states_prop)
        for entry in dirEntries:
            if re.match(dn1, entry):
                dr = entry
        if not dr:
            raise ValueError('I do not know the name of your directory: '+dn1)        
        dr = dr + '/'
    dr = path + dr
#    print 'Semantic Path: ', dr, '\n'
    return dr

def normalize_manyjointdict_perword():
    lxc = 'fnc'
    normalize_jointdict_perword('fncn_sbjwd_5Nov_40m.txt', 'fncn_normlzd_sbjwd_5Nov_40m.txt', 'BLISS_fncwds.txt')
    normalize_jointdict_perword('fncvb_sbjwd_5Nov_40m.txt', 'fncvb_normlzd_sbjwd_5Nov_40m.txt', 'BLISS_fncwds.txt')
    normalize_jointdict_perword('fncadj_sbjwd_5Nov_40m.txt', 'fncadj_normlzd_sbjwd_5Nov_40m.txt', 'BLISS_fncwds.txt')
    pu.concatfiles('fncont_intgd_sbjwd_5Nov_40m.txt', 'fncn_normlzd_sbjwd_5Nov_40m.txt', 'fncvb_normlzd_sbjwd_5Nov_40m.txt', 'fncadj_normlzd_sbjwd_5Nov_40m.txt')
    normalize_jointdict_perword('fncont_intgd_sbjwd_5Nov_40m.txt', 'fncont_intgd_normlzd_sbjwd_5Nov_40m.txt', 'BLISS_fncwds.txt')
    
if __name__ == "__main__":
    SEMPATH = get_drn(neu='semrep', manual='randcor-1800-27Sep12/') 
    print 'SEMPATH: ', SEMPATH, '\n'

    create_fullneurep_nowlabfile(SEMPATH+'pattern_zeta0.00apf0.00.txt_modfwdsp',
                                 pu.get_fn_neurep('n2v2afp', 'nhf20', SYNPATH, 'syneurep')['rpc'],
                                 FPATH + 'myWORDS_fneurep_nhf0nhf20_07rplcd.csv')
