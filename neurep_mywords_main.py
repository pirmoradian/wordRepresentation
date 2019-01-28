# -*- coding: utf-8 -*-
"""
Created on Mon Aug 22 13:11:13 2011

@author: Sahar
"""
import neurep_mywords as nw
import prjUtil as pu
from neurep_mywords import get_path_fullfn
import KLdiv
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
import math
import pdb
import numpy as np
import blissplot
import blissutil as bu
import re
import os
#import Image
from cnst import *
import sys
import random
#===============================================================================
# RUN FUNCTIONS
#===============================================================================
def run_gen_neurep_cldns(p='result/synrep-test,code,chng-6Feb12/',neu='syneurep', nhf_l=[20]):
    '''
    nws.run_gen_neurep_cldns()
    '''
    open(AVG_FILE,'a').write('***************\n' + pu.get_date() + '\n' +
                                '$' + p + '\n')
    if (neu == 'syneurep'):
        wcat_l = ['fwd', 'npl', 'nsg', 'vpl', 'vsg', 'adj', 'psg', 'ppl', 'n2v2afp'] #'n2v2ap',
    elif (neu == 'neurep'):
        wcat_l = ['vpl', 'adj','fwd', 'psg', 'ppl', 'n2v2afp'] 
    for wcat in wcat_l:
        print '\n\n*** Generating representation of category', wcat, '***'
        gen_neurep_colorednoise(wcat, p,
                           neurep_flg=True, corr_flg=True, ent_flg=False, corplt_flg=True, neu=neu, nhf_l=nhf_l)
#        gen_neurep_colorednoise(wcat, p,
#                           neurep_flg=False, corr_flg=False, ent_flg=False, corplt_flg=True, neu=neu, nhf_l=nhf_l)
    
def gen_neurep_colorednoise(wcat='adj', path='result/synrep-cldns-18Jan12/',
                              neurep_flg = True, corr_flg = True, ent_flg = True, corplt_flg = True,
                              neu='syneurep',nhf_l=[20]):
    '''
    >>> syneurep_mywords_main.gen_neurep_colorednoise()
    '''
    mainpath = MAINPATH 
    sepfile = True
    f_out = open(AVG_FILE,'a')
    modfy_cntb = True
    #print "Modify contribution flag:", modfy_cntb
    for nhf in nhf_l:
        sfx = 'nhf' + str(nhf)
        f_out.write('*' + sfx + '\n')
        print '\nno_hf:' + str(nhf)
        d = pu.get_fn_neurep(wcat,sfx,path,neu=neu)
#        for item in d.items(): 
#            print item.__repr__()+'\n'
        if re.match('n2v2\w*', wcat):
            neurep_flg = False
            corplt_flg = False
            corr_flg = True
            ent_flg = True
            sepfile = False
            wcat_l = d['wcat_l']
            in_files = tuple([pu.get_fn_neurep(wc,sfx,path,neu=neu)['fn'] for wc in wcat_l])
            pu.concatfiles(d['fn'], *in_files)
        if (re.match('nsg', wcat) and re.match('neurep', neu)):
            neurep_flg = False
            corr_flg = False
            ent_flg = True
        if (re.match('fwd', wcat) and re.match('syneurep', neu)):        
            modfy_cntb = False
        if (neurep_flg):
            # pdb.set_trace()
            print '\nRepresentations are being saved into:\n', d['fn'], '\n'
            nw.neurep_nouns(n_units=d['nu'], n_states=d['ns'], sparsity=d['sp'],
                        nouns_infile=d['wsofcat'], 
                        concs_feats_infile=d['fct_contb'], 
                        feats_neurep_infile=d['fct_neurep'],  
                        feats_place=[0,1], cor_place=2, noise_max=0.0, 
                        n_hf=nhf, colored_noise=True,sparsity_hf=sparsity_hf,
                        nouns_neurep_outfile=d['fn'], modfy_cntb=modfy_cntb)
            if (flip_states_flg == 1):
                nw.flip_actvstates(d['fn'], d['fn'], n_states=d['ns'], sparsity=d['sp'], proportion=flip_states_prop)
            elif(flip_states_flg == 2):
#                pdb.set_trace()                
                nw.flip_allstates(d['fn'], d['fn'], n_states=d['ns'], sparsity=d['sp'], proportion=flip_states_prop, propinact=flip_states_prop_inact)
        if (ent_flg):
            # Entropy
            reposition_wdsneurep(d['fn'], d['repos'], refpos=d['wsofcat'])
            rm_wordlab_neurepfile(infile=d['repos'],outfile=d['nolab'])
            ent_u_l = get_entropy_units(d['nolab'], entunit_file=d['entunit'])
            plot_entropy_units(d['nolab'], '',entunit_file=d['entunit'])
            print 'avg(ent):' + str(np.mean(ent_u_l))
            print 'std(ent):' + str(np.std(ent_u_l))
            f_out.write(wcat + ': '+ 'avg(ent):' + str(np.mean(ent_u_l)) +'  std:' + str(np.std(ent_u_l)) + '\n')            
            if (wcat == 'n2v2afp'):            
                replc_n1_n2_neurepfile(d['nolab'], d['rpc'])
        if (corr_flg):
            # Computes correlation
            print '\nComputing correlations and plotting and storing Nas histograms into:\n' + d['neucor'] + '_Nas.png'
            nw.get_corr_pirmoradfeats(d['fn'], sepfile, d['fct_neurep'], d['neucor'])
            get_avgNas(corr_infile_pirmorad=d['neucor'], corr_id_pirmorad=1)    
            plot_Nas_dist(d['neucor'], del_max=d['max_actunt'])#pu.get_fn_neurep('n2v2afp', 'nhf100', SEMPATH, 'neurep')['neucor'])
            #    print 'mean:', m, ' std:',s
            nw.get_corr_pirmoradfeats(d['fn'], False, d['fct_neurep'], d['selfneucor'])
            get_avgNas(corr_infile_pirmorad=d['selfneucor'], corr_id_pirmorad=1)    
            plot_Nas_dist(d['selfneucor'], del_max=d['max_actunt'])

            if (wcat == 'n2v2afp'):
                fout = d['neucor'] + '_adp2pats'
                replc_w_n_jntfile(fn=d['neucor'], type_='cor', fout=fout)

        if (corplt_flg):
            print '\nPlotting relation between join probability of words and the Nas of their neural representation, and saving into:\n' + d['neucor'] + '.png'
#            nw.scatter_corr_pirmorad_mcrae(corr_infile_pirmorad=d['neucor'], corr_infile_mcrae=d['fct_contb'],  corr_id_pirmorad=1, xlab='Joint Prob in a Sbj-Verb Corpus', ylab='N_as in Pirmorad db (neurep)', title='Corr. bet. '+wcat+' and their facts '+sfx, feats_place=[0, 1], cor_place=2,color_cats=False,x_log=xlog_flg)
            nw.scatter_corr_pirmorad_mcrae(corr_infile_pirmorad=d['neucor'], corr_infile_mcrae=d['fct_contb'],  corr_id_pirmorad=1, xlab='Joint Prob. in BLISS', ylab='Nas', title='', feats_place=[0, 1], cor_place=2,color_cats=True,categories_file=d['wsofcat'], x_log=xlog_flg, FLG_mod_contb=0)
            plt.savefig(mainpath + d['neucor'] + '.png')
        
        if (wcat == 'fwd' and neu == 'syneurep'):
            pu.concatfiles(pu.get_fn_neurep('adj',sfx,path,neu=neu)['fct_neurep'], 
                        d['fct_neurep'], d['fn'])
        if (wcat == 'n2v2afp' and neu == 'syneurep'):            
            plot_Nas_pairoflxcats(sfx=sfx, path=path, neu=neu)   
#            filize_lxcfcts()    
#            filize_fncwds()
#            plot_Nas_lxcatswithlxcfcts(sfx=sfx, path=path, neu=neu)    
        if (wcat == 'adj' and neu == 'neurep'):
            pu.concatfiles(pu.get_fn_neurep('fwd',sfx,path,neu=neu)['fct_neurep'], 
                        d['fct_neurep'], pu.get_fn_neurep('vpl',sfx,path,neu=neu)['fn'],
                        d['fn'])
        if (wcat == 'vpl' and neu == 'neurep'):
            convt_base2sform(infile=d['fn'], 
                 outfile=pu.get_fn_neurep('vsg',sfx,path,neu=neu)['fn'], type_='neurep')                
    f_out.close()    

#===============================================================================
# FILE MANIPULATIONS
#===============================================================================
def create_fullneurep_files(semsfx='nhf60', synsfx='nhf20',sempath = 'result/semrep-cldns-ngtcont1.2-2forprop-7Jan12/',
                            synpath = 'result/synrep-ngt0.0,fsyn1.0,nvcb1.0,avgt1,fwd1.0-12:20-11Mar12/',
                            fpath = 'result/fullrep-12:31-11Mar12/'):
    '''
    nws.create_fullneurep_files()
    '''
        
    lxcats = ['nsg', 'vsg', 'adj', 'fwd', 'npl', 'vpl', 'psg', 'ppl']    
    # pdb.set_trace()    
    for i in range(len(lxcats)):
        fn1 = pu.get_fn_neurep(lxcats[i],semsfx,sempath, 'neurep')['fn']
        fn2 = pu.get_fn_neurep(lxcats[i],synsfx,synpath, 'syneurep')['fn']
        fn3 = pu.get_fn_neurep(lxcats[i],semsfx+synsfx,fpath, 'fneurep')['fn']
        nw.create_fullneurep_file(semneurep_file=fn1, syneurep_file=fn2, fneurep_file=fn3)
    
    concatfiles_allwds(semsfx=semsfx, synsfx=synsfx, fpath=fpath)

def concatfiles_allwds(semsfx='nhf60', synsfx='nhf20', 
                       fpath='result/fullrep-12:31-11Mar12/', lxcats=None, 
                       txtapnd='all'):
    print 'Concatenating semantic and syntactic files...'
    lxcats = lxcats or ['nsg', 'vsg', 'adj', 'fwd', 'npl', 'vpl', 'psg', 'ppl']    
    fn_l = []
    d = pu.get_fn_neurep(txtapnd,semsfx+synsfx,fpath, 'fneurep')
    dst = d['fn']
    # pdb.set_trace()
    for i in range(len(lxcats)):
        fn_l.append(pu.get_fn_neurep(lxcats[i],semsfx+synsfx,fpath, 'fneurep')['fn'])
    in_files = tuple(fn_l)
#    pdb.set_trace()
    if not os.path.exists(dst):
        pu.concatfiles(dst, *in_files)
    else:
        print 'This file already exists:', dst
    reposition_wdsneurep(dst, d['repos'],
                         refpos=d['wsofcat'])

    rm_wordlab_neurepfile(infile=d['repos']
                         ,outfile=d['nolab'])
                         
    replc_n1_n2_neurepfile(d['nolab'],
                           d['rpc'])
        
    ent_u_l = get_entropy_units(d['nolab'], entunit_file=d['entunit'])
    plot_entropy_units(d['nolab'], '',entunit_file=d['entunit'])    
    print 'avg(ent):' + str(np.mean(ent_u_l))
    print 'std:' + str(np.std(ent_u_l))
    
    nw.get_corr_pirmoradfeats(d['fn'], False, 'test', d['neucor'])
    get_avgNas(corr_infile_pirmorad=d['neucor'], corr_id_pirmorad=1)  
    plot_Nas_dist(d['neucor'], del_max=225)
        
def reposition_wdsneurep(infile,outfile,refpos=PATS_FILE):
    d = nw.filereader_factory('neurep', infile)
    refp = nw.filereader_factory('readl', refpos)
    rpos_l = []    
    for w in refp:
        rpos_l.append((w, d[w]))
    nw.filewriter_factory('neurep', outfile, rpos_l)
    
def replc_n1_n2_neurepfile(fn = 'nw', fout = 'out', n1=0, n2=7):
    '''
    >>> nws.replc_n1_n2_neurepfile()
    '''
    # pdb.set_trace()
    text = open(fn, 'rU').read()
    temp = 8
    if (temp==n1 or temp==n2):
        raise ValueError('Conflict of Value between temp and your numbers!')
    text = text.replace(str(n1), str(temp))
    text = text.replace(str(n2), str(n1))
    text = text.replace(str(temp), str(n2))
    open(fout, 'w').write(text)

def replc_w_n_jntfile(type_='jntpb_fq0', fn='jntf', fout='out', pat_file='pats_149.txt'):
    '''
    >>> nws.replc_n1_n2_neurepfile()
    '''
    # pdb.set_trace()
    d1 = nw.filereader_factory(type_, fn)    
    pats = nw.filereader_factory('readl', pat_file)    
    d = {}    
    for k, v in d1.iteritems():
        w1, w2 = k.split()
        npr = str(pats.index(w1)) + ' ' + str(pats.index(w2))
        d[npr] = v
        
    bu.UtilDict().writetofile(d, fout)    
    
def rm_wordlab_neurepfile(infile='myNOUNS_neurep.csv',
                          outfile='myNOUNS_neurep_nowlab.csv'):
    '''
    >>> nws.rm_wordlab_neurepfile(infile='myNOUNS_neurep_fsp1.csv',outfile='myNOUNS_neurep_fsp1_nowlab.csv')
    '''
    
    f_in = open(infile, 'rU')
    f_out = open(outfile, 'w')
    for line in f_in.readlines():
        w, neurep = line.split('\t')
        f_out.write(neurep)
    f_in.close()
    f_out.close()

def put_wordlab_neurepfile(infile='myNOUNS_neurep_nowlab.csv',
                          outfile='myNOUNS_neurep_wlab.csv'):
    '''
    >>> nws.put_wordlab_neurepfile(infile='myNOUNS_neurep_fsp1.csv',outfile='myNOUNS_neurep_fsp1_nowlab.csv')
    '''
    
    f_in = open(infile, 'rU')
    f_out = open(outfile, 'w')
    counter = 0
    for line in f_in.readlines():
        f_out.write(str(counter) + '\t' + line)
        counter += 1
    f_in.close()
    f_out.close()
    
def convt_base2sform(infile='fncn_normalized_sbjwd_5Nov_10m.txt', 
                     outfile='fncn_sform_normalized_sbjwd_5Nov_10m.txt', type_='jntpb',*args, **kwds):
    '''
    >>> nws.convt_base2sform(infile='fncn_normalized_sbjwd_5Nov_10m.txt', outfile='fncn_sform_normalized_sbjwd_5Nov_10m.txt', feats_place=[0,1], cor_place=2)
    >>> nws.convt_base2sform(infile='fncvb_normalized_sbjwd_5Nov_10m.txt', outfile='fncvb_sform_normalized_sbjwd_5Nov_10m.txt', feats_place=[0,1], cor_place=2)
    '''
    root_d=blissplot.get_wbaseform_d()
    sform = root_d.keys()
    bform = root_d.values()
    basefile_d = nw.filereader_factory(type_, infile, *args, **kwds)
    outfile = open(outfile, 'w')
    for k, v in basefile_d.iteritems():
        l = ''
        ws = k.split()
        for w in ws:
            if w in bform:
                w = sform[bform.index(w)]
            l = l + w + '\t'
        if type(v) is list:
            v = [str(i) for i in v]
            stv = ' '.join(v)
        else:
            stv = str(v)
        outfile.write(l + stv + '\n')
    
def convt_matrx_jointpb(f1='mycontwds_synfactors_matrix.csv',
                        f2='mycontwds_synfactors.txt'):
    '''
    >>> nws.convt_matrx_jointpb(f1='mynouns_synfactors_matrix.csv', f2='mynouns_synfactors.txt')
    '''
    d = nw.read_concscorr_mcraefile(f1)
    fout = open(f2, 'w')
    for cont_synf, w in d.iteritems():
        cont, synf = cont_synf.split()
        fout.write(cont + '\t' + synf + '\t' + str(w) + '\n')
    fout.close()
    
#===============================================================================
# Units Measures
#===============================================================================
def get_avgNas_perword(corr_infile, 
                       avgNas_ofile, corr_id_pirmorad=1, pat_file=PATS_FILE):
    corrs_d = nw.read_corr_pirmoradfile(corr_infile, corr_id_pirmorad)
    cor_wd_d = {}
    f_o = open(avgNas_ofile, 'w') 
    print 'Warning in get_avgNas_perwd(): Take care about your input file'
#    pdb.set_trace()
    for k, v in corrs_d.iteritems():
        wds = k.split()
        #for w in wds:
        w = wds[0]
        cor_wd_d[w] = cor_wd_d.get(w, 0) + v
#    pdb.set_trace()
    no_pats = float(len(cor_wd_d))
    for k, v in cor_wd_d.iteritems():
        f_o.write(k + '\t' + str(v/no_pats) + '\n')        
    f_o.close()
    
def get_avgNas_perword_withalf(corr_infile, 
                       avgNas_ofile, corr_id_pirmorad=1, pat_file=PATS_FILE):
    corrs_d = nw.read_corr_pirmoradfile(corr_infile, corr_id_pirmorad)
    cor_wd_d = {}
    f_o = open(avgNas_ofile, 'w')    
    for k, v in corrs_d.iteritems():
        wds = k.split()
        for w in wds:
            cor_wd_d[w] = cor_wd_d.get(w, []) + [v]
    for k, v in cor_wd_d.iteritems():
        m0 = np.mean(v)
        srt_v = sorted(v)
        len_v = len(srt_v)
        half_v = srt_v[len_v/2:]
        m1 = np.mean(half_v)
        del(half_v[-1])
        m2 = np.mean(half_v)
        f_o.write(k + '\t' + str(m0) + '\t' + str(m1) + '\t' + str(m2) + '\n')        
    f_o.close()
    
def get_avgNas(corr_infile_pirmorad='myFEATS_neurep_corr.csv', corr_id_pirmorad=1):
    '''
    nws.get_avgNas()
    '''
    f_out = open(AVG_FILE,'a')
#    f_out.write('#############\n' + corr_infile_pirmorad + '\n')
   
    corrs_pirmorad_d = nw.read_corr_pirmoradfile(corr_infile_pirmorad, 
                                              corr_id_pirmorad)
#    print  'avg(Nas):' + str(np.mean(corrs_pirmorad_d.values()))
#    print  'std(Nas):' + str(np.std(corrs_pirmorad_d.values()))
#    f_out.write('Nas: '+ 'avg:' + str(np.mean(corrs_pirmorad_d.values()) )+
#                '  std:' + str(np.std(corrs_pirmorad_d.values())) + '\n')

    return  np.mean(corrs_pirmorad_d.values())

def get_avgNas_elefiles(f='pattern.dat', corr_id_pirmorad=1, path='result/randcor-16:00-28May12/', form='scatter',save_fig=1,new_fig=1):
    fn = path+f
#    pdb.set_trace()
    data = nw.filereader_factory('nowlab', fn)
    Nas_l = []
    Na_l = []
    Nad_l = []
    no_pats = np.shape(data)[0]
    no_units = np.shape(data)[1]
    for n_p1 in range(no_pats):
        p1 = list(data[n_p1, :])
        for n_p2 in range(n_p1+1, no_pats):
            p2 = list(data[n_p2, :])
            nas = 0
            na = 0
            for n_u in range(no_units):
                if ((p1[n_u] != 7) and (p2[n_u] != 7)):
                    na = na + 1
                    if (p1[n_u] == p2[n_u]):
                        nas = nas + 1
            Nas_l.append(nas)
            Nad_l.append(na-nas)
            Na_l.append(na)
    print  'avg(Nas):' + str(np.mean(Nas_l))
    print  'std(Nas):' + str(np.std(Nas_l))
    print  'avg(Nad):' + str(np.mean(Nad_l))
    print  'std(Nad):' + str(np.std(Nad_l))
    print  'avg(Na):' + str(np.mean(Na_l))
    print  'std(Na):' + str(np.std(Na_l))
    
    sfx = ''
    dist_l = []
    if corr_id_pirmorad == 0:
        dist_l = Na_l
        sfx = 'Na'
    elif corr_id_pirmorad == 1:
        dist_l = Nas_l
        sfx = 'Nas'
    elif corr_id_pirmorad == 2:
        dist_l = Nad_l
        sfx = 'Nad'
    
    mean, std = plot_freq_dist(dist_l, form=form, xlab=sfx,save_fig=save_fig,new_fig=new_fig)
#    plt.savefig(path + 'Nas_' + f +'.png')
    
    # fit an exponential function
#    logNas_fq_srt = [np.log(nas) for nas in Nas_fq_srt]
#    z= polyfit(Nas_srt, logNas_fq_srt, 1)
#    print 'exp values:', z
#    z2=[(np.exp(z[1])*np.exp(z[0]*i)) for i in Nas_srt]    
#    plt.figure();plt.plot(Nas_srt, z2)
#    mean, std = plot_freq_dist(Na_l, title='Na',save_fig=save_fig,new_fig=new_fig)
    if save_fig:    
        plt.savefig(path + 'Nas_' + f +'.png')
    return mean, std
    
def get_entropy_units(filename, entunit_file, actunt=False, inactst=7):
    '''
    >>> ent_u_l = get_entropy_units('pattern.dat')
    '''
#    pdb.set_trace()
    data = nw.filereader_factory('nowlab', filename)
    ent_u_l = []
    no_units = np.shape(data)[1]
    #SAHAR:
    f_test = open(entunit_file, 'w')
    entunitd_file = entunit_file + '_d'
    f_test2 = open(entunitd_file, 'w')
    u_allz_no = 0
    u_allz_l = []
    for n_u in range(no_units):
        st_u = list(data[:,n_u])
        f_test.write(str(st_u)+'\n')
        hist_st_d = dict([(x, st_u.count(x)) for x in set(st_u)])
        f_test2.write(str(hist_st_d) + '\n')
        if actunt:
            if inactst in hist_st_d:
                del[hist_st_d[inactst]]
#            else:
#                print 'No ', inactst, 'for unit ', n_u, ' in ', hist_st_d
        if not hist_st_d:
            u_allz_no += 1
            u_allz_l.append(n_u)
        ent_u = KLdiv.shannon_entropy(hist_st_d)
        ent_u_l.append(ent_u)
    print 'no of units whose states are all inactive: ', u_allz_no
#    print 'units no: ', u_allz_l
    f_test.close()
    return ent_u_l

def get_corr_wdl(corfile, wdl):
    
    cor_na_d = nw.filereader_factory('cor', corfile, corr_id_pirmorad=0)
    cor_nas_d = nw.filereader_factory('cor', corfile, corr_id_pirmorad=1)
    f = open(WDL_FILE, 'a')
    f.write(corfile + '\n')
    f.write('word1 word2 Na Nas\n')
    for w in nw.combinations(wdl, 2):
        pair = ' '.join([w[0], w[1]])
        revpair = ' '.join([w[1], w[0]])
        if pair in cor_na_d:
            na = cor_na_d[pair]
            nas = cor_nas_d[pair]
        else:
            na = cor_na_d[revpair]
            nas = cor_nas_d[revpair]
        f.write(pair + ' ' + str(na) + ' ' + str(nas) + '\n')
    f.write('***********\n')
    f.close()
#===============================================================================
# PLOT FUNCTIONS    
#===============================================================================
def plot_feat_cpf_dist(feats_infile='myFEATS_brm.csv'):
    '''
    >>> nws.plot_feat_cpf_dist()
    '''
    f_in = open(feats_infile, 'rU')
    cpf_l = []
    for line in f_in.readlines():
        line_sp = line.split('\t')
        cpf_l.append(int(line_sp[1]))
    plot_freq_dist(cpf_l, title='Frequency of Concepts per Feature', 
                   xlab='No. of Concepts per Feature', 
                   ylab='Frequency', figname='feat_cpf_dist')

def plot_Nas_pairoflxcats(sfx='nhf20', path='result/synrep-fsynmlt.1-27Jan12/', neu='syneurep'):
    '''
    nws.plot_Nas_pairoflxcats(sfx='nhf20', path='result/synrep-fsynmlt.1-27Jan12/')
    '''
    lxcats = WD_CATS 
    lxcats_nm = lxcats
    hline_thsh = [3.2] * (len(lxcats)-1) + [6.4]
    if (neu=='neurep'):
        lxcats = ['nsg', 'vsg', 'adj', 'psgpl', 'fwd']
        lxcats_nm = ['Nouns', 'Verbs', 'Adjs', 'Propns', 'Fwds']
        hline_thsh = [4.8] * 4 + [1.5]
    lxcats_nm_captlz = [i.title() for i in lxcats_nm]
#    pdb.set_trace()    
    fn_lxcats = get_fn_lxcats(path, sfx, neu, lxcats=lxcats)
    h = open('Nas', 'a')    
    h.write('\n**** ' +sfx + ' ' + path + '\n')
    for i in range(len(lxcats)):
        colornames = ['b'] * len(lxcats)
        colornames[i] = 'g'
        plt.figure()
        avgnas = []
        fn1 = fn_lxcats[i]
        for j in range(len(lxcats)):
            fn2 = fn_lxcats[j]
            print lxcats[i] + ", " + lxcats[j]
            fncor = path + lxcats[i] + lxcats[j] + sfx + '_corr.csv'
            #pdb.set_trace()            
            nw.get_corr_pirmoradfeats(fn1, True,fn2, fncor)
            avgnas.append(get_avgNas(fncor, 1))
        avgnas_s = [str(int(round(av))) for av in avgnas]
#        pdb.set_trace()
        h.write(lxcats[i] + ': ' + '  '.join(avgnas_s)+'\n')
        plt.bar(range(len(lxcats)), avgnas, color=colornames)
        plt.xticks(np.arange(len(lxcats))+.4, lxcats_nm_captlz,fontsize=20)
        if (neu=='neurep'):# or i < No_CONTCATS):   
#            plt.ylim([0, 90])
            plt.yticks(range(0,41,10),fontsize=20)
        else:                    
#            plt.ylim([0, 180])
            plt.yticks(range(0,91,20),fontsize=20)
        plt.hlines([hline_thsh[i]],0,len(lxcats),colors='k', linestyles='dashed', lw=1.8);
        #plt.ylim([0,40])
        plt.ylim([0,90])
        plt.ylabel('<Nas>',fontsize=20)
        
#        plt.title('avg Nas between ' + lxcats[i] + ' and other lxcats (' + sfx + ')')
        plt.savefig(path + 'avgNas_' + lxcats[i] + '_' + sfx +'.png', dpi = 200)
    h.close()

def plot_Nas_lxcatswithlxcfcts(sfx='nhf20', path='result/synrep-fsynmlt.1-27Jan12/', neu='syneurep'):
    '''
    nws.plot_Nas_lxcatswithlxcfcts(sfx='nhf20', path='result/synrep-fsynmlt.1-27Jan12/')
    '''
    lxcats = WD_CATS    
    lxcats_nm = lxcats
    lxcats_nm_captlz = [i.title() for i in lxcats_nm]
    fn_lxcats = get_fn_lxcats(path, sfx, neu)

    lxcfct1 = nw.filereader_factory('readl', F_SYNFCT) #['lxc/n', 'lxc/v', 'lxc/aj', 'lxc/conj', 'lxc/prep', 'lxc/pron', 'lxc/adv'] 
    lxcfct = [re.sub("/", "",f) for f in lxcfct1]
    # pdb.set_trace()    
    h = open('Nas', 'a')    
    h.write('\n**** ' +sfx + ' ' + path + '\n')
    for j in [6]: #range(len(lxcfct)):
        fn2 = path + lxcfct[j] + '.csv'
        plt.figure()
        avgnas = []
#        colornames = ['r', 'r', 'b', 'b', 'b', 'r', 'r', 'b']
#        colornames = ['b', 'b', 'r', 'r', 'b', 'b', 'b', [.1,.5,1]]
#        colornames = ['b', 'b', 'b', 'b', 'r', 'b', 'b', 'c']
#        colornames = ['b', 'b', 'b', 'b', 'b', 'b', 'b', [0.4,.6,.9]]
#        colornames = ['b', 'b', 'b', 'b', 'b', 'b', 'b', 'c']
#        colornames = ['b', 'b', 'b', 'b', 'b', 'b', 'b', [.1,.5,1]]
        colornames = ['b', 'b', 'b', 'b', [.5,.8,.9], 'b', 'b', [0.4,.6,.9]]
        for i in range(len(lxcats)):
            fn1 = fn_lxcats[i] #pu.get_fn_neurep(lxcats[i],sfx,path,neu=neu)['fn']
            print lxcfct[j] + ", " + lxcats[i]
            fncor = path + lxcats[i] + '_' + lxcfct[j] + sfx +'_corr.csv'
            #pdb.set_trace()            
            nw.get_corr_pirmoradfeats(fn1, True, fn2, fncor)
            avgnas.append(get_avgNas(fncor, 1))
        avgnas_s = [str(int(round(av))) for av in avgnas]
#        pdb.set_trace()
        h.write(lxcats[i] + ': ' + '  '.join(avgnas_s)+'\n')
        plt.bar(range(len(lxcats)), avgnas, color=colornames)
        plt.xticks(np.arange(len(lxcats))+.4, lxcats_nm_captlz,fontsize=26)
        #plt.title('avg Nas between ' + lxcfct1[j] + ' and lxcats (' + sfx + ')')
        plt.yticks(range(0,91,10),fontsize=20)
        #plt.ylim([0,90])
        plt.ylabel('<Nas>',fontsize=30)        
        plt.savefig(path + 'avgNas_' + lxcfct[j] + '_' + sfx +'.png')
    h.close()
            
def get_fn_lxcats(path=SYNPATH, sfx='nhf20', neu='syneurep', lxcats=WD_CATS):    
    fn_lxcats = []
    if neu=='syneurep':
#        for i in range(No_CONTCATS):
#            fn_lxcats.append(pu.get_fn_neurep(lxcats[i],sfx,path,neu=neu)['fn'])
#        for i in range(No_CONTCATS, NO_CATS):
#            fn_lxcats.append(path + lxcats[i] + '.csv')
        for i in range(NO_CATS):
            fn_lxcats.append(pu.get_fn_neurep(lxcats[i],sfx,path,neu=neu)['fn'])
    
    elif neu=='neurep':
        for i in range(len(lxcats)):
            fn_lxcats.append(pu.get_fn_neurep(lxcats[i],sfx,path,neu=neu)['fn'])
    return fn_lxcats 
      
def plot_Nas_dist(corr_infile_pirmorad='myFEATS_neurep_corr.csv', corr_id_pirmorad=1, del_max=135,save_fig=1,new_fig=1):
    '''
    nws.plot_Nas_dist()
    '''
    t = corr_infile_pirmorad.split('/')[-1]
    f_out = open(AVG_FILE,'a')
    f_out.write('#############\n' + corr_infile_pirmorad + '\n')
   
#    plt.figure()    
    corrs_pirmorad_d = nw.read_corr_pirmoradfile(corr_infile_pirmorad, 
                                              corr_id_pirmorad)
#    f_out.write('Nas: '+ 'avg:' + str(np.mean(corrs_pirmorad_d.values()) )+
#                '  std:' + str(np.std(corrs_pirmorad_d.values())) + '\n')
    for k, v in corrs_pirmorad_d.items():
        w1, w2 = k.split()
        if w1 == w2:
            del corrs_pirmorad_d[k]
            #print 'The words with exactly the same name were deleted, regardless of their correlation'
        
    if (corr_id_pirmorad==0):
        sfx='Na'
    elif(corr_id_pirmorad==1):
        sfx='Nas'
    elif(corr_id_pirmorad==2):
        sfx='Nad'
    fullfgn = corr_infile_pirmorad + '_' + sfx + '.png'
    mean,std = plot_freq_dist(corrs_pirmorad_d.values(), title=sfx+': '+t, 
                   xlab=sfx, 
                   ylab='Frequency', figname=fullfgn, del_max=del_max,save_fig=save_fig, new_fig=new_fig)    
    return mean, std
    
def plot_Nas_dist_aword(corr_infile_pirmorad='myFEATS_neurep_corr.csv', word='house', corr_id_pirmorad=1):
    '''
    nws.plot_Nas_dist()
    '''
    f_out = open(AVG_FILE,'a')
    f_out.write('#############\n' + corr_infile_pirmorad + '\n')
   
    corrs_d = nw.read_corr_pirmoradfile(corr_infile_pirmorad, 
                                              corr_id_pirmorad)
    Nas_l = []
#    pdb.set_trace()    
    for k, v in corrs_d.iteritems():
        wds = k.split()
        for w in wds:
            if (w==word):
                Nas_l.append(v)
 
    print  'avg:' + str(np.mean(Nas_l))
    print  'std:' + str(np.std(Nas_l))
    f_out.write('Nas: '+ 'avg:' + str(np.mean(Nas_l))+
                '  std:' + str(np.std(Nas_l)) + '\n')
    plot_freq_dist(Nas_l, title='Distribution of Nas: '+word, 
                   xlab='Nas', 
                   ylab='Frequency', figname='Nasdist_'+word)    

    
def plot_freq_dist(cpf_l, form='scatter',title='', xlab='Nas', 
                   ylab='Frequency', figname='freq_dist', ymin=0, xmin=0, del_max=135,save_fig=1,new_fig=1):   
    if new_fig:
        plt.figure()                       
    d=dict([(cpf, cpf_l.count(cpf)) for cpf in set(cpf_l)])
    if del_max in d:
        cpf_l = [value for value in cpf_l if value != del_max]
        del d[del_max]
        print "I removed the Nas with ", str(del_max)
    x = d.keys()
    y = d.values()

    #y = [cpf / float(sum(y)) for cpf in y]
    if (form=='scatter'):
        plt.scatter(x, y)
    elif (form == 'bar'):
        plt.hist(cpf_l, bins=50)
    
    #plt.title('Distribution of Concepts per Feature')
    mean = np.mean(cpf_l)
    std = np.std(cpf_l)
    mean_st = "%.1f" % mean
    std_st = "%.1f" % std
    title_=''
    plt.title(title_)
    xlab=xlab
    plt.xlabel(xlab, fontsize=30)    
    #plt.ylabel('Probability')
#    pdb.set_trace()
    plt.ylabel(ylab, fontsize=30)
    plt.yticks(range(0,2001,1000),fontsize=20)
    plt.xticks(range(0,41,10),fontsize=20)
    plt.text(30,1700, 'mean: '+mean_st, fontsize=20) #30  3000 40 100 2000
    plt.text(30,1500, 'std: ' + std_st, fontsize=20)
    plt.ylim(ymin=ymin)
    plt.xlim(xmin=xmin)
    #    plt.ylim(ymax=1000)
#    plt.xlim(xmax=50)  
#    plt.vlines(mean,0,4000,colors='k', linestyles='dashed', lw=1.8);
    plt.subplots_adjust(bottom=0.12,left=0.16)
    plt.tight_layout()
    if save_fig:
        plt.savefig(figname)
    return mean, std

def plot_entropy_units(filename, sfx='.dat',entunit_file='entunits.txt', actunt=False, *args, **kwds):
    '''
    >>> plot_entropy_units('pattern.dat')
    '''
    #pdb.set_trace()
    plt.figure()
    ent_u_l = get_entropy_units(filename, entunit_file, actunt=actunt, *args, **kwds)
#    print 'avg:' + str(np.mean(ent_u_l))
#    print 'std:' + str(np.std(ent_u_l))
    plt.scatter(range(len(ent_u_l)), ent_u_l)
    plt.ylim([0,3])
    plt.title(filename)
    plt.xlabel('Units')
    plt.ylabel('Entropy')
    fullfgn = entunit_file+'actu' + str(actunt) +'.png'
    plt.savefig(fullfgn) 
    

def assign_colorcodes(filename='BLISS_fncwds+factorsyn.txt', cmap='jet'):
    plt.figure()
    fncwds=nw.filereader_factory('readl', filename)
    num = len(fncwds)
    plt.scatter(range(num), [0]*num, c=range(num), s=30, cmap=cmap, linewidth=.5)
    plt.xticks(np.arange(num),fncwds, rotation=45)  
    
        
def visualize_neurep(neurep_file, wd, path):
    f = plt.figure()
    colors=['w', 'b', 'g', 'r', 'c', 'm', 'y', 'k']
    d = filereader_factory('neurep', neurep_file)
    rep_l = d[wd]
#    rep_l =[1,2,3,4,5,6,7,1,2]
    mtxsz = int(sqrt(len(rep_l)))
    plt.xlim([0,mtxsz])
    plt.ylim([0,mtxsz])
    axis('off')
    f.patch.set_facecolor('w')
    for idx, st in enumerate(rep_l):
        col = idx % mtxsz
        row = mtxsz - int(idx / mtxsz)
        plt.text(col, row, '*', color=colors[st])
    plt.savefig(path + 'vis_' + wd + '.png')

def filize_lxcfcts():

    lxcfcts_reps = nw.filereader_factory('neurep', pu.get_fn_neurep('n2v2afp', 'nhf20', SYNPATH, 'syneurep')['fct_neurep'])
    lxcfcts = nw.filereader_factory('readl', F_SYNFCT) #['lxc/n', 'lxc/v', 'lxc/aj', 'lxc/conj', 'lxc/prep', 'lxc/pron', 'lxc/adv']  
    for i in range(len(lxcfcts)):        
        lxcf =  re.sub("/", "",lxcfcts[i])
        outfile= SYNPATH+ lxcf + '.csv'
        d = {lxcfcts[i]: lxcfcts_reps[lxcfcts[i]]}
        nw.filewriter_factory('neurep', outfile, d)

def filize_fncwds():

    fn_neurep = pu.get_fn_neurep('fwd', 'nhf20', SYNPATH, 'syneurep')['fn']
    files =     FWD_CATS
    wordlists = [FWD_CNJ, FWD_PREP, FWD_AUX, FWD_ART, FWD_DEM]
    for i in range(len(files)):
        fout= SYNPATH+ files[i] + '.csv'
        extract_neurep_wordlist(fn_neurep, wordlists[i], fout)

def extract_neurep_wordlist(fn_neurep, wordlist, fout):
#    pdb.set_trace()
    allwd_reps = nw.filereader_factory('neurep', fn_neurep)
    d = {}    
    for w in wordlist:        
        d[w] = allwd_reps[w]
    nw.filewriter_factory('neurep', fout, d)

    
def regen_synfactors():
#    pdb.set_trace()

    #regenerating syntactic factors
    nw.neurep_factors_fncwds(n_units=359, n_states=7, sparsity=.50, factor_rep_outfile=F1)
    
    #combining factors + contentwords
    pu.itemize_lxcpbmtx(inmtxfn=F2, outmtxfn=F3)
    convt_matrx_jointpb(F3, F4)
    pu.rescale_dictf(F4, F4_rsc, 'jntpb', feats_place=[0,1], cor_place=2)    
    pu.concatfiles(F6, F4_rsc, F5_rsc)
    pu.cnvt_wdsform_jpbfile(F6, F7, 'BLISS_nouns_verbs_bs.txt', form='sform',  fn_cndwds='BLISS_fncwds.txt', cnd_flg='True')
    pu.concatfiles(F8, F6, F7)
    oscomd = 'sort ' + F8 + ' | uniq > ' + F9
    os.system(oscomd)

def plot_avgNas_elefiles(save_fig=0,new_fig=0):
    zeta=[0.00]#, 0.10, 0.20]
    apf = [0.00]#,0.01,0.03,0.05,0.08,0.10]#, 0.50, 0.90]
    supt = ''#randomly correlated'
#    apf = [18, 37, 15, 36, 55]    
    fig=plt.figure()
    figno = 1
    for z in zeta:
        z = "%.2f" % z
        for a in apf:
            a = "%.2f" % a
            par = 'zeta'+ z + 'apf' + a
#            a = "%d" % a
#            par = 'p' + a
            elefile='pattern_' + par + '.txt' + '_modfwdsp'
            path= 'result/randcor-18:00-27Sep12/'
#            path="/home/user/Documents/network/Eleprj/"
            fig.add_subplot(2, 3, figno)
            mean, std = get_avgNas_elefiles(elefile, path, form='scatter',save_fig=save_fig,new_fig=new_fig)#bar
            figno += 1
            plt.title('zeta='+ z + ' apf=' + a,  fontsize=12)
#            plt.title('#patterns = '+ a,  fontsize=12)
            plt.xlabel('')
            plt.ylabel('')
            mean = "%.1f" % mean
            std = "%.1f" % std
            text1 = 'mean:'+mean
            text2 = 'std:'+std
            plt.xticks(range(0, 51,10), fontsize=8)#121 41, 10 61
            plt.yticks(range(0,1001,200), fontsize=8)# 2501 281, 40 2501
            plt.text(40,800, text1, fontsize=8) #30  3000 40 100 2000
            plt.text(40,700, text2, fontsize=8) #30 1800 2500 40 80
#            plt.xlim([0,120])
#            plt.ylim([0,4000])
    plt.suptitle(supt)
    plt.show()

def plot_avgNas_words_withcor(path=''):
#    for i in range(len(lxc)):
#        lxc1 = lxc[i]
#        for j in range(i,len(lxc)):
#            lxc2 = lxc[j]
#            f = path + lxc1+lxc2+'_corr.csv'
#            plot_Nas_dist(f)

    v1 = ['nsg', 'vsg', 'fwd', 'nsg', 'nsg']
    v2 = ['nsg', 'vsg', 'fwd', 'adj', 'vsg']    
    fig=plt.figure()
    figno = 1
    for i in range(len(v1)):
            d_mx = 135
            z = v1[i]
            a = v2[i]
            if (a=='fwd' or z =='fwd'):
                d_mx = 45
#            par = 'v1'+ z + 'v2' + a
#            a = "%d" % a
            f = path + z+a+'_corr.csv'
            fig.add_subplot(3, 2, figno)
#            pdb.set_trace()
            mean, std = plot_Nas_dist(f, del_max=d_mx,save_fig=0,new_fig=0)#bar
            figno += 1
            plt.title(z + ' - ' + a,  fontsize=12)
#            plt.title('#patterns = '+ a,  fontsize=12)
            plt.xlabel('')
            plt.ylabel('')
            mean = "%.1f" % mean
            std = "%.1f" % std
            text1 = 'mean:'+mean
            text2 = 'std:'+std
            plt.xticks(range(0, 91, 10), fontsize=8)#121 61,20
            plt.yticks(range(0,161, 40), fontsize=8)# 2501 2501,500
            plt.text(70, 100, text1, fontsize=8)  #2000 3000 40
            plt.text(70, 80, text2, fontsize=8) #1800 2500 40
#            plt.xlim([0,120])
#            plt.ylim([0,4000])
    fullfgn = path + 'Nas_lxcats.png'
    plt.savefig(fullfgn)
    plt.show()
    
def plot_avgNas_hiddenfactors():
    #v1s=['semrep-ngt1.0,fsyn1.0,nvcb1.0,avgt1,fwd1.0,popn0,flp2,fprp0.2,fiact0.25,sphf0.00-12:04-05Oct12','semrep-ngt1.0,fsyn1.0,nvcb1.0,avgt1,fwd1.0,popn0,flp2,fprp0.5,fiact0.25,sphf0.00-12:14-05Oct12','semrep-ngt1.0,fsyn1.0,nvcb1.0,avgt1,fwd1.0,popn0,flp2,fprp0.8,fiact0.25,sphf0.00-12:27-05Oct12','semrep-ngt1.0,fsyn1.0,nvcb1.0,avgt1,fwd1.0,popn0,flp2,fprp1.0,fiact0.25,sphf0.00-12:28-05Oct12'] #
#    v1s = ['semrep-ngt1.0,fsyn1.0,nvcb1.0,avgt1,fwd1.0,popn0,flp0,fprp0.0,fiact0.00,sphf0.25-15:25-05Oct12','semrep-ngt1.0,fsyn1.0,nvcb1.0,avgt1,fwd1.0,popn1,flp0,fprp0.0,fiact0.00,sphf0.25-15:48-05Oct12', 'semrep-ngt1.0,fsyn1.0,nvcb1.0,avgt1,fwd1.0,popn0,flp0,fprp0.0,sphf0.75-11:32-05Oct12','semrep-ngt1.0,fsyn1.0,nvcb1.0,avgt1,fwd1.0,popn1,flp0,fprp0.0,sphf0.75-11:33-05Oct12'] 
#    v1s = ['semrep-ngt1.0,fsyn1.0,nvcb1.0,avgt1,fwd1.0,popn0,flp1,fprp0.2,fiact0.00,sphf0.00-15:57-05Oct12', 'semrep-ngt1.0,fsyn1.0,nvcb1.0,avgt1,fwd1.0,popn0,flp1,fprp0.5-20:07-04Oct12', 'semrep-ngt1.0,fsyn1.0,nvcb1.0,avgt1,fwd1.0,popn0,flp1,fprp0.8-20:09-04Oct12', 'semrep-ngt1.0,fsyn1.0,nvcb1.0,avgt1,fwd1.0,popn0,flp1,fprp1.0-20:10-04Oct12']    
#    v1s=['semrep-ngt1.0,fsyn1.0,nvcb1.0,avgt1,fwd1.0,popn0,flp2,fprp0.2,fiact0.25,sphf0.00-19:36-05Oct12','semrep-ngt1.0,fsyn1.0,nvcb1.0,avgt1,fwd1.0,popn0,flp2,fprp0.5,fiact0.25,sphf0.00-19:37-05Oct12','semrep-ngt1.0,fsyn1.0,nvcb1.0,avgt1,fwd1.0,popn0,flp2,fprp0.8,fiact0.25,sphf0.00-19:37-05Oct12','semrep-ngt1.0,fsyn1.0,nvcb1.0,avgt1,fwd1.0,popn0,flp2,fprp1.0,fiact0.25,sphf0.00-19:38-05Oct12']    
    v1s =[]    
    popn = [0]; flp=[0]; fprp=[.0]; fiact=[0.]; sphf=[.25,.5,1.0]; nhfs=[400,400,200];
    for popni in popn:
        for flpi in flp:
            for fprpi in fprp:
                for fiacti in fiact:
                    for sphfi in sphf:
                        drn = nw.get_drn(sparsity_hf=sphfi,popnoun_flg =popni, 
                                flip_states_flg =flpi,flip_states_prop =fprpi, 
                                flip_states_prop_inact =fiacti)
                        v1s.append(drn)
#    v2s = ['myVERBPL_neurep_nhf60_corr.csv_Nas.png','myNOUNS_VERBPL_nhf60_corr.txt_Nas.png'] #['myVERBPL_neurep_nhf60_corr.csv_Nas.png','myNOUNS_VERBPL_nhf60_corr.txt_Nas.png'] #['my_N2V2AFP_nhf0_corr.txt_Nas.png']#
#    v2s = ['myVERBPL_neurep_nhf'+str(i)+'_corr.csv_Nas.png' for i in nhfs]
#    v2s = ['myNOUNS_VERBPL_nhf'+str(i)+'_corr.txt_Nas.png' for i in nhfs]
#    x1=[0,0,1,1]
#    x1=[0,0,1,1]
#    x2=[.25,.75,.25,.75]    
    params = nhfs 
    subt = '#hidden_factors='#'Flip_proportion=' #'hf_sparsity=' 
    sfx = '(sparsity='+str(sphf[0])+')'#'(Flipping both active and inactive units)'  #'(considering popularity of nouns)' # '(Changing hidden factor sparsity)' 
    supts = ['Corr bet Nouns and Verbs '+sfx]#['Corr bet all words '+sfx] #['Corr bet Verbs and Verbs '+sfx,'Corr bet Nouns and Verbs ' + sfx] #, 'Corr bet Nouns and Verbs (considering popularity of nouns)']#
    fignm = ['hf_sp'+str(sphf[0])+'_nv.png']#'flip_allunits_words_.25to.3.png']#['hf_nounpop_vv.png', 'hf_nounpop_nv.png']# ['hf_sp.5to1.0_vv.png','hf_sp.5to1.0_nv.png']#['flip_actunits_vv.png','flip_actunits_nv.png']#['flip_allunits_vv_.1to.4.png', 'flip_allunits_nv_.1to.4.png']   
    figpath = 'result/semrep-noise-Oct12/'    
    
    formats = ["%s", "%s", "%.2f"]
    path = ''
    figno = 1
    plt.figure()
    for i in range(len(v2s)):
#        plt.figure()#figsize=(6*3.13,4*3.13))
        v2 = v2s[i]
        v2_st = formats[0] % v2   #"%.4f" % v1 #  
        for j in range(len(v1s)):
            v1 = v1s[j]
            param = formats[2] % params[i]
            v1_st = formats[1] % v1 #"%.6f" % v2 #
#            fgn = fgn_template % (hr, v1,v1,v2, v2)
            fgn = v1_st + '/' + v2_st
            
            fullfgn = path + fgn    
#            plt.subplot(len(v1s),len(v2s),figno)
            plt.subplot(2,2,figno)
#            plt.title(varnames[0]+'='+ v1_st + ' '+ varnames[1] +'=' + v2_st,  fontsize=12)
            plt.title(subt + param,  fontsize=12)
#            plt.title('NOUNS_POPULARITY_FLG=' + str(x1[j])+',sp='+param,  fontsize=12)
            
            plt.axis('off')
            im = Image.open(fullfgn)
            plt.imshow(im, origin='lower') 
            figno += 1
#    plt.suptitle('zeta=0.00, a_pf=0.01')
    plt.suptitle(supts[0])
    plt.savefig(figpath+fignm[0], dpi = 200)
    figno = 1
    plt.show()
    
def avgsparsity(filename):
    d = nw.filereader_factory('neurep', filename)
    sparsity = []    
    no_wds = len(d)    
    for w, neu in d.iteritems():
        no_units = float(len(neu))
        neu_arr = np.array(neu)
        no_active_units = len(neu_arr[neu_arr>0])
        sp = no_active_units/no_units
#        print w, ':', sp
        sparsity.append(sp)
#    print sparsity
#    print sort(sparsity)
    return np.mean(sparsity)        

def regen_avgNas(path,hf='nhf0',corr_id=1,save_fig=1):
    wcat_l = ['vpl', 'adj','fwd', 'psg', 'ppl', 'n2v2afp']
    for wcat in wcat_l:
        d = pu.get_fn_neurep(wcat,hf,path,neu='neurep')    
#        get_avgNas(corr_infile_pirmorad=d['neucor'], corr_id_pirmorad=corr_id)    
        plot_Nas_dist(d['neucor'], del_max=d['max_actunt'],corr_id_pirmorad=corr_id, save_fig=save_fig)
        #    print 'mean:', m, ' std:',s
#        get_avgNas(corr_infile_pirmorad=d['selfneucor'], corr_id_pirmorad=corr_id)    
        plot_Nas_dist(d['selfneucor'], del_max=d['max_actunt'],corr_id_pirmorad=corr_id, save_fig=save_fig)

def create_jntpbfile(fn_pb):
    f1 = 'BLISS_adjs.txt'
    f2 = 'BLISS_nouns_sg.txt'
    f3 = 'BLISS_verbs_sg.txt'
    z = [(f1,f2), (f2,f3), (f3,f1), (f3,f2)]
    fpb = open(fn_pb, 'w')
    for files in z:
        file1 = nw.filereader_factory('readl', files[0])
        file2 = nw.filereader_factory('readl', files[1])
        pb = 1./(len(file2))
        if (file1 =='BLISS_verbs_sg.txt'):
            pb = 1./36
        for wd1 in file1:
            for wd2 in file2:
                fpb.write(wd1 + '\t' + wd2 + '\t' + str(pb) +'\n')
    
    fpb.close()
    fout = fn_pb+'_adp2pats'            
    replc_w_n_jntfile(fn=fn_pb, type_='jntpb', fout=fout)

def modify_sparsity_fwds(fn, fn_out):
    fwds = nw.filereader_factory('readl', 'BLISS_fncwds.txt')
    pats = nw.filereader_factory('readl', 'pats_149.txt')
    data = nw.filereader_factory('nowlab', fn)
    fout = open(fn_out, 'w')
    no_flips = 90 # 135 - 45
    idx_l = []
    for idx, item in enumerate(pats):
        if item in fwds:
            idx_l.append(idx)  
    for p_no in range(len(pats)):
        p = list(data[p_no, :])        
        if p_no in idx_l:
#            pdb.set_trace()
            idx_actv_units = [idx for idx,item in enumerate(p) if item!=7]
            idx_chosenunts = random.sample(idx_actv_units, no_flips)
            for i in idx_chosenunts:
                p[i] = 7
        p_st = [str(int(i)) for i in p]
        p_st = ' '.join(p_st)
        fout.write(p_st + '\n')
    fout.close()
    
def plot_nouncor_lxcats(filename, path=''):
    catnames = ['animals', 'buildings', 'objects']
    for i in range(len(catnames)):
        colornames = ['b'] * len(catnames)
        colornames[i] = 'g'
        plt.figure()
        catnm1 = catnames[i]
        fn1 = 'myNOUNS_' + catnm1 + '_neurep_fadt33mlt3cpf.csv'
        avgnas = []
        for j in range(len(catnames)):
            catnm2 = catnames[j]
            fn2 = 'myNOUNS_' + catnm2 + '_neurep_fadt33mlt3cpf.csv'
            print catnm1 + ", " + catnm2
            fncor = path + 'myNOUNS' + catnm1 + catnm2 + '_neurep_fadt33mlt3cpf_corr.csv'
            #pdb.set_trace()            
            nw.get_corr_pirmoradfeats(fn1, True,fn2, fncor)
            avgnas.append(get_avgNas(fncor, 1))
#        avgnas_s = [str(int(round(av))) for av in avgnas]
#        pdb.set_trace()
        plt.bar(range(len(catnames)), avgnas, color=colornames)
        plt.xticks(np.arange(len(catnames))+.42, catnames,fontsize=30)
        plt.yticks(range(0,41,10),fontsize=20)
        plt.hlines([4.8],0,2.9,colors='k', linestyles='dashed', lw=1.8);
        plt.ylim([0,40])
        plt.ylabel('<Nas>',fontsize=30)        
#        plt.title('avg Nas between ' +   catnm1 + ' and other word cats')
        plt.savefig(path + 'avgNas_' + catnm1 +'.png',dpi = 200)

if __name__ == "__main__":
    
#    pu.mknewdir(neu=neu)

    if NEU=='neurep':
      # nhf_l: Number of hidden factors for generating the neural representation
      nhf_l = [400]
      run_gen_neurep_cldns(SEMPATH, neu=NEU, nhf_l=nhf_l)
      
    elif NEU=='syneurep':
      nhf_l = [20]
      regen_synfactors()
      run_gen_neurep_cldns(SYNPATH, neu=NEU, nhf_l=nhf_l)
      
    elif NEU=='fneurep':
      create_fullneurep_files(semsfx='nhf400', synsfx='nhf20', sempath = SEMPATH, synpath = SYNPATH, fpath = FPATH)
      

    
#    regen_avgNas(SEMPATH, 'nhf200',corr_id=0)
        
    # testing plot_Nas_pairoflxcats
#    plot_Nas_pairoflxcats(sfx='nhf20', path=SYNPATH, neu='syneurep')   
#    pdb.set_trace()

#    filize_lxcfcts()    
#    filize_fncwds()
#    plot_Nas_lxcatswithlxcfcts(sfx='nhf20', path=SYNPATH, neu='syneurep')    
     
#    create_jntpbfile('pFreq_2rulevt.txt')     
     
#    plot_avgNas_hiddenfactors()
    
    # testing convt_matrx_jointpb
#    convt_matrx_jointpb('mycontwds_synfactors_matrix_192Feb12.csv', 
#                        'mycontwds_synfactors_192Feb12.txt')

#    create_fullneurep_files(semsfx='nhf1000', synsfx='nhf20', sempath = SEMPATH, synpath = SYNPATH, fpath = FPATH)
                      
#    pdb.set_trace()
#    wdl = ['proves', 'kills', 'needs', 'banishes', 'hope', 'prove', 'kill', 'need', 'get', 'banish', 'die']
#    for wd in ['proves', 'kills', 'needs', 'banishes', 'hope', 'prove', 'kill', 'need', 'get', 'banish', 'die']:
#        visualize_neurep(pu.get_fn_neurep('all', 'nhf60nhf20', FPATH, 'fneurep')['fn'], wd,FPATH)
#    get_corr_wdl(pu.get_fn_neurep('n2v2afp', 'nhf20', SYNPATH, 'syneurep')['neucor'], wdl)
#    ofile = SEMPATH + 'avgNas_myN2V2AFP_nhf60_perwd.txt'
#    get_avgNas_perword(pu.get_fn_neurep('n2v2afp', 'nhf60', SEMPATH, 'neurep')['neucor'], ofile)    
#    ofile = SEMPATH + 'avgNas_myN2V2AFP_nhf60_perwdwithalf.txt'
#    get_avgNas_perword_withalf(pu.get_fn_neurep('n2v2afp', 'nhf60', SEMPATH, 'neurep')['neucor'], ofile)    

#    ns='nhf0'; p=SEMPATH; neu='neurep';
#    fout = pu.get_fn_neurep('n2v2afp', ns, p, neu)['neucor'] + '_adp2pats'
#    pdb.set_trace()    
#    replc_w_n_jntfile(fn='/home/user/Documents/python/pFreq_sbj_1rulevi_nmz.txt', type_='jntpb',
#                      fout='/home/user/Documents/python/result/pFreq_sbj_1rulevi_adp2pats.txt')

#    plot_avgNas_elefiles()

#    get_avgNas_elefiles(f='pattern_zeta0.00apf0.00.txt_modfwdsp', corr_id_pirmorad=0, path='result/randcor-18:00-27Sep12/', form='scatter',save_fig=1,new_fig=1)

#    plot_avgNas_words_withcor(SEMPATH)

#    plot_Nas_dist_aword(pu.get_fn_neurep('n2v2afp', 'nhf60', SEMPATH, 'neurep')['neucor'], 'house')
#    m,s = plot_Nas_dist(pu.get_fn_neurep('n2v2afp', 'nhf400', SEMPATH, 'neurep')['neucor'])
#    print 'mean:', m, ' std:',s

#    m,s = plot_Nas_dist(SEMPATH + 'myNVAFP_neurep_nhf400_corr.csv',corr_id_pirmorad=0)
#    print 'mean:', m, ' std:',s

#    avg = avgsparsity('myFEATS_neurep_adt33mlt3cpf.csv')    
#    print str(avg)
#    pass
#    pdb.set_trace()
#    lxcats = [['nsg', 'npl'], ['vsg', 'vpl'], ['adj'], ['fwd'], ['psg', 'ppl']]    
#    txtapnds=['nouns', 'verbs', 'adj', 'fwd', 'propns']
#    for i in range(len(lxcats)):
#        concatfiles_allwds(semsfx='nhf60', synsfx='nhf20', 
#                       fpath=FPATH, lxcats=lxcats[i], txtapnd=txtapnds[i])
#    fn = SEMPATH+'pattern_zeta0.00apf0.00.txt'
#    modify_sparsity_fwds(fn , fn+'_modfwdsp')

#    plot_nouncor_lxcats('myNOUNS_neurep_fadt33mlt3cpf.csv')
