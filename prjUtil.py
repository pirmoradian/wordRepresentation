# -*- coding: utf-8 -*-
"""
Created on Fri Jan 27 16:13:44 2012

@author: Sahar
"""
import pdb
import shutil
from cnst import *
import neurep_mywords as nw
from time import strftime
import os
import time
import blissplot as bp
import blissutil as bu
import copy

def get_fn_neurep(wcat,sfx,path,neu='syneurep'):
        # wcat must be 'nsg' 'vsg' 'adj'
        # sfx can take on values: 'nhf0', 'nhf10', 'nhf20'
        # neu can take on values: 'syneurep', 'neurep', 'fneurep', 
        replaceTxt = '_07rplcd'   
        wcat_l = []
        lxcor = ''
        fct_neurep = ''
        fct_name = ''
#===============================================================================
# FOR (ALMOST) ALL NEURAL REPRESENTATION
#===============================================================================
        if (wcat == 'nsg'):
            txtapnd = 'NOUNSG'
            wsofcat = 'BLISS_nouns_sg.txt'
            fct_contb3 = 'fncn_sbjwd_5Nov_10m.txt'
        elif (wcat == 'npl'):
            txtapnd = 'NOUNPL'
            wsofcat = 'BLISS_nouns_pl.txt'
            fct_contb3 = 'fncn_sform_sbjwd_5Nov_10m.txt' 
        elif (wcat == 'vsg'):
            txtapnd = 'VERBSG'
            wsofcat = 'BLISS_verbs_sg.txt'
            fct_contb3 = 'fncvb_sform_sbjwd_5Nov_10m.txt' 
        elif (wcat == 'vpl'):
            txtapnd = 'VERBPL'
            wsofcat = 'BLISS_verbs_bs.txt'
            fct_contb3 = 'fncvb_sbjwd_5Nov_10m.txt' 
        elif wcat == 'fwd':
            txtapnd = 'FNCWDS'
            wsofcat = 'BLISS_fncwds.txt'
            fct_contb3 = 'myfncwds_factors_matrix.csv'   
            fct_contb = fct_contb3        
        if wcat == 'adj':
            txtapnd = 'ADJS'
            wsofcat = 'BLISS_adjs.txt'
            fct_contb3 = 'fncadj_sbjwd_5Nov_10m.txt' 
        if wcat == 'psg':
            txtapnd = 'PROPNSG'
            wsofcat = 'BLISS_propnouns_sg.txt'
            fct_contb3 = 'fncpropnsg_sbjwd_5Nov_10m.txt' 
        if wcat == 'ppl':
            txtapnd = 'PROPNPL'
            wsofcat = 'BLISS_propnouns_pl.txt'
            fct_contb3 = 'fncpropnpl_sbjwd_5Nov_10m.txt'
        if wcat == 'psgpl':
            txtapnd = 'PROPNSGPL'
            wsofcat = 'BLISS_propnouns.txt'
            fct_contb3 = 'TODO'
        if (wcat == 'nv'):
            txtapnd = 'NV'
            wsofcat = 'TODO'
            fct_contb3 = 'TODO'
            wcat_l = ['nsg', 'vsg']
        if (wcat == 'n2v2afp'):
            txtapnd = 'N2V2AFP'
            wsofcat = PATS_FILE
            fct_contb3 = 'TODO'
            wcat_l = ['nsg', 'vsg', 'adj', 'fwd', 'npl', 'vpl', 'psg', 'ppl']
        if (wcat == 'n2v2ap'):
            txtapnd = 'N2V2AP'
            wsofcat = 'TODO'
            fct_contb3 = 'TODO'
            wcat_l = ['nsg', 'vsg', 'adj', 'npl', 'vpl', 'psg', 'ppl']
        if (wcat == 'n2v2a'):
            txtapnd = 'N2V2A'
            wsofcat = 'TODO'
            fct_contb3 = 'TODO'
            wcat_l = ['nsg', 'vsg', 'adj', 'npl', 'vpl']
        if (wcat == 'all'):
            txtapnd = 'WORDS'
            wsofcat = PATS_FILE
            fct_contb3 = 'TODO'
        if (wcat == 'nouns'):
            txtapnd = 'NOUNS'
            wsofcat = 'BLISS_nouns_sgpl.txt'
            fct_contb3 = 'TODO'
        if (wcat == 'verbs'):
            txtapnd = 'VERBS'
            wsofcat = 'BLISS_verbs_sgpl.txt'
            fct_contb3 = 'TODO'
        if (wcat == 'propns'):
            txtapnd = 'PROPNS'
            wsofcat = 'BLISS_verbs_sgpl.txt'
            fct_contb3 = 'TODO'
#===============================================================================
# ONLY SYNTACTIC UNITS        
#===============================================================================
        if (neu == 'syneurep'):
            n_units=359; n_states=7; sparsity=0.25
            fct_contb = F9 #'mycontwds_allsynfactors_fin_21Feb12.txt'
            fct_neurep = os.path.join(path, 'myFACTORS+FNCWDS_syneurep_' + sfx + '.csv') #TODO: check for other cases (other than syneyrep!)
            lxcor = 'FNCWDS'
            if wcat == 'fwd':
                fct_contb = F10 #'myfncwds_factors_matrix.csv'
                sparsity= 0.5
            if wcat == 'n2v2afp':
                lxcor = ''
#===============================================================================
# ONLY SEMANTIC UNITS        
#===============================================================================
        if (neu == 'neurep'):
            n_units=541; n_states=7; sparsity=0.25
            fct_contb = 'TODO'
            fct_neurep = 'myNOUNSG_neurep_fadt33mlt3cpf.csv'
            if (wcat=='vsg' or wcat=='vpl'):
                fct_contb = 'nvb_normlzd_sbjwd_5Nov_40m.txt' 
                lxcor = 'NOUNS' 
                fct_name = 'BLISS_nouns_sg.txt'
            elif wcat == 'adj':
                fct_contb = 'nadj_normlzd_sbjwd_5Nov_40m.txt' 
                lxcor = 'NOUNS'
                fct_name= 'BLISS_nouns_sg.txt'
            elif wcat == 'fwd':
                sparsity = 1./12
                lxcor = 'CONTWDS'           
                fct_contb = 'fncont_intgd_normlzd_sbjwd_5Nov_40m.txt'
                fct_neurep = os.path.join(path, 'myNVA_' + neu + '_' + sfx + '.csv')            
                fct_name = 'BLISS_contwds.txt'
            elif (wcat == 'psg' or wcat == 'ppl'):
                lxcor = 'CONTWDS'           
                fct_contb = 'propncont_intgd_normlzd_sbjwd_5Nov_40m.txt' 
                fct_neurep = os.path.join(path, 'myNVA_' + neu + '_' + sfx + '.csv')
                fct_name = 'BLISS_contwds.txt'
#===============================================================================
# FULL UNITS
#===============================================================================
        if (neu == 'fneurep'):
            n_units=900; n_states=7; sparsity=0.25; 
            fct_contb = 'TODO'    
            


        #pdb.set_trace()
        fn = os.path.join(path, 'my' + txtapnd + '_' + neu + '_' + sfx + '.csv')
        nolab = os.path.join(path, 'my' + txtapnd + '_' + neu + '_' + sfx + '_nowlab.csv')
        entunit = os.path.join(path, 'entunit_' + str.lower(txtapnd) + '_' + sfx + '.txt')
        neucor = os.path.join(path, 'my' + txtapnd + '_'+ lxcor +'_' + sfx + '_corr.txt')
        avgNas_pwd = os.path.join(path, 'my' + txtapnd + '_'+ lxcor +'_' + sfx + '_avgNas.txt')
        selfneucor = os.path.join(path, 'my' + txtapnd + '_' + neu + '_' + sfx + '_corr.csv')        
        #fct_contb2 = 'my' + str.lower(txtapnd) + '_synfactors+fnc.txt'
        rpc = os.path.join(path, 'my' + txtapnd + '_' + neu + '_' + sfx + replaceTxt + '.csv')
        repos = os.path.join(path, 'my' + txtapnd + '_' + neu + '_' + sfx + '_r' + '.csv')
        
#===============================================================================
# EXCEPTIONS
#===============================================================================
        if (neu == 'neurep'):
            neucor = os.path.join(path, 'my' + lxcor +'_' + txtapnd + '_'+ sfx + '_corr.txt')
            if (wcat=='nsg' or wcat=='npl'):
                fn = 'my' + txtapnd + '_neurep_fadt33mlt3cpf.csv'
                fct_neurep = 'myFEATS_neurep_adt33mlt3cpf.csv'
        if (neu == 'syneurep' and (wcat == 'fwd')):
            neucor = os.path.join(path, 'myFACTORS_FNCWDS_' + sfx + '_corr.txt')
            fct_neurep = F1  #'myFACTORS_FNCWDS_syneurep.csv'
            
        # returing a dict of filenames    
        d = dict()
        d['fn'] = fn
        d['nolab']= nolab
        d['entunit'] = entunit
        d['neucor'] = neucor
        d['selfneucor'] = selfneucor
        d['fct_contb'] = fct_contb
        #d['fct_contb2'] = fct_contb2
        d['fct_contb3'] = fct_contb3
        d['fct_name'] = fct_name
        d['rpc'] = rpc
        d['fct_neurep'] = fct_neurep
        d['wsofcat'] = wsofcat
        d['nu'] = n_units
        d['ns'] = n_states
        d['sp'] = sparsity
        d['sp_hf'] = sparsity_hf
        d['repos'] = repos
        d['wcat_l'] = wcat_l
        d['max_actunt'] = int(n_units*sparsity)
        return d
# pertaining to TIME
def get_date():
    return time.asctime( time.localtime(time.time()) )

# pertaining to Dictionary    
def get_deformed_dict(d, form='root'):
    if (form == 'root'):
        form_d=bp.get_wbaseform_d()
    elif(form == 'sform'):
        form_d=bp.get_wsform_d()            
    d_cp = copy.deepcopy(d)
    for k in d:
        del d_cp[k]
        words = k.split()
        words_root = []
        for w in words:
            if w in form_d:
                w = form_d[w]
            words_root.append(w)
        d_cp[' '.join(words_root)] = d[k]
    return d_cp 

# pertaining to File & Directory
def concatfiles(dst, *args):
    destination = open(dst,'w')
    for f in args:
        shutil.copyfileobj(open(f,'rU'), destination)
    destination.close()


def mknewdir(neu='syneurep', dscp=''):
    if (neu == 'syneurep'):
        dn = "%s%s-ngt%.1f,fsyn%.1f,nvcb%.1f,avgt%d,fwd%.1f-%s%s" % (nw.get_path_result(), neu, ngt_mltp, fsyn_mltp, nv_contb_mltp, avg_over_tot, fncwd_mltp, strftime("%H%M-%d%b%y"),dscp)
    elif (neu == 'neurep'):
        dn = "%s%s-ngt%.1f,fsyn%.1f,nvcb%.1f,avgt%d,fwd%.1f,popn%d,flp%d,fprp%.2f,fiact%.2f,sphf%.2f-%s%s" % (nw.get_path_result(), neu, ngt_mltp, fsyn_mltp, nv_contb_mltp, avg_over_tot, fncwd_mltp, popnoun_flg, flip_states_flg, flip_states_prop, flip_states_prop_inact, sparsity_hf, strftime("%H%M-%d%b%y"),dscp)
    elif (neu == 'fneurep'):
        dn = "%s%s-%s%s" % (nw.get_path_result(), neu, strftime("%H%M-%d%b%y"),dscp)
    else:
        raise ValueError('I do not know the type of your directory: ' + neu)

    if not os.path.exists(dn): 
        os.makedirs(dn)
        print 'new dir created: ' + dn
#TODO: def get_path_storedrep(neu, ):
            
def cnvt_wdsform_jpbfile(fn_in, fn_o, fn_wds, form='root', 
                         fn_cndwds='BLISS_fncwds.txt', cnd_flg='False'):
    '''

    '''    
    # reading the file with words that have to be coverted to different form ('root' or 'sform')
    d = nw.filereader_factory('unknown', fn_in, feats_place=[0,1], cor_place=2)
    # reading the list of words that need to be converted to a different form    
    ref_wds = nw.filereader_factory('unknown', fn_wds)
    # reading the list of conditional words that if appear together with our main words, the conversion will happen
    ref_cndwds = nw.filereader_factory('unknown', fn_cndwds)
    # getting the list of all roots and sforms of words
    if (form=='root'):  
        form_d=bp.get_wbaseform_d()
    elif (form=='sform'):
        form_d=bp.get_wsform_d() 
    out_d = {}
    outf = open(fn_o, 'w')
    #pdb.set_trace()
    # converting to different form all words in the dictionary that appear together with cndwds 
    for pair, fq in d.iteritems():
        w1, w2 = pair.split()
        if ((w1 in ref_wds) and ((not cnd_flg) or (w2 in ref_cndwds))):
                w1 = form_d[w1]
        elif ((w2 in ref_wds) and ((not cnd_flg) or (w1 in ref_cndwds))):
                w2 = form_d[w2]
        newds = [w1, w2]
        out_d['\t'.join(newds)] = fq
        outf.write('\t'.join(newds+[str(fq)]) + '\n')
    outf.close()

def itemize_lxcpbmtx(inmtxfn='mylxcs_synfactors_matrix.csv', outmtxfn='mylxcitems_synfactors_matrix.csv'):
#    pdb.set_trace()    
    mtx_d = nw.filereader_factory('matrix', inmtxfn)
    lxcs = ['nsg', 'npl', 'vsg', 'vpl', 'psg', 'ppl', 'adj']
    item_pb_d = {}
    rowds = []
    for wcat in lxcs:
        for pair in mtx_d.keys():
            fq = mtx_d[pair]
            w1, w2 = pair.split()
            if (w1 == wcat):
                wcol = w2
            elif (w2 == wcat):
                wcol = w1
            else:
                continue
            del mtx_d[pair]
            wsofcat = nw.filereader_factory('readl', get_fn_neurep(wcat,'','',neu='syneurep')['wsofcat'])
#            pdb.set_trace()
            for w in wsofcat:
                item_pb_d[' '.join([wcol, w])] = fq
        rowds = rowds + wsofcat        
    colwds = nw.filereader_factory('readl', F_SYNFCT)
    nw.filewriter_factory('matrix', outmtxfn, item_pb_d, colwds, rowds)

def rescale_dictf(filename, outfile, type_='jntpb', *args, **kwds):
    d = nw.filereader_factory('unknown', filename, *args, **kwds)
    d = bu.UtilDict().rescale_d(d)
    nw.filewriter_factory(type_, outfile, d)
    
if __name__ == "__main__":
    # testing get_fn_neurep()
    path = MAINPATH
#    h = open('test_get_fn_neurep.txt', 'w');
#    for neu in ['syneurep']:
#        h.write('\n' + neu+'\n')
#        wcat_l = ['fwd', 'npl', 'nsg', 'vpl', 'vsg', 'adj', 'psg', 'ppl', 'n2v2afp']
#        for wcat in wcat_l:
#            h.write('\n')
#            for sfx in ['nhf0']:
#                dfn = get_fn_neurep(wcat,sfx,path,neu)
#                for item in dfn.items(): 
#                    h.write(item.__repr__()+'\n')
#    h.close()

    # testing cnvt_wdsform_jpbfile()
#    cnvt_wdsform_jpbfile('mycontwds_allsynfactors_test.txt', 'testo_cnvt_wdsform', 'BLISS_nouns_verbs_bs.txt', form='sform',  fn_cndwds='BLISS_fncwds.txt', cnd_flg='True')
#    cnvt_wdsform_jpbfile('mycontwds_allsynfactors_21Feb12.txt', 'mycontwds_allsynfactors_sform_21Feb12.txt', 'BLISS_nouns_verbs_bs.txt', form='sform',  fn_cndwds='BLISS_fncwds.txt', cnd_flg='True')
    
    # testing mknewdir
    mknewdir(neu='fullrep')
    
    # testing itemize_lxcpbmtx
#    itemize_lxcpbmtx(inmtxfn='mylxcs_synfactors_matrix_192Feb12.csv', outmtxfn='mycontwds_synfactors_matrix_192Feb12.csv')

    # testing concatfiles
#    concatfiles('mycontwds_allsynfactors+sform_21Feb12.txt', 'mycontwds_allsynfactors_21Feb12.txt', 'mycontwds_allsynfactors_sform_21Feb12.txt')

    # testing rescale_dictf
#    rescale_dictf('mycontwds_synfactors_192Feb12.txt', 'mycontwds_synfactors_192Feb12_rscl.txt', 'jntpb', feats_place=[0,1], cor_place=2)
