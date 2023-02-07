#This is the original pmtools2

import numpy as np
from collections import Counter


def get_clusters(eidx_to_clabel, k):
    #eidx_to_clabel is a list containing the cluster labels of each entry
    #k is the number of clusters
    #Returns a list of lists: clabel_to_indexes, where clabel_to_indexes[clabel] has
    #the indexes of the entries of cluster with clabel
    
    clabel_to_indexes = []
    
    for clabel in range(k):

        entries_indexes = [entry_idx for entry_idx,label in enumerate(list(eidx_to_clabel)) if label == clabel]

        #Get entries of the cluster
        entries_indexes_np = np.array(entries_indexes)
        
        #if clabel == 0: #For debugging
            #print('Cluster label', clabel, ':' , entries_indexes)
        
        clabel_to_indexes.append(list(entries_indexes_np))
    
    return clabel_to_indexes


def get_entries_freqs(entries):
    #entries is a list of dataset entries
    #Returns a list of collcounters: attidx_to_collcounter #TODO: como es la mejor forma de explicarlo
    
    num_atts = len(entries[0])
    attidx_to_collcounter = []
    entries_np = np.array(entries)
    
    for attidx in range(num_atts):
        attidx_to_collcounter.append(Counter(entries_np[:,attidx]))
    
    return attidx_to_collcounter


def get_clusters_freqs(entries, clabel_to_indexes):
    #entries is a list of dataset entries
    #clabel_to_indexes: list of lists. Where clabel_to_indexes[clabel] has
    #the indexes of the entries of cluster with clabel
    #Returns a list of dictionaries: attidx_clabel_to_collcounter #TODO: como es la mejor forma de explicarlo
    
    num_atts = len(entries[0])
    entries_np = np.array(entries)
    attidx_clabel_to_collcounter = [dict() for attidx in range(num_atts)]

    for clabel,indexes in enumerate(clabel_to_indexes):

        #Get entries of the cluster
        indexes_np = np.array(indexes)
        c_entries_np = entries_np[indexes_np]

        for attidx in range(num_atts):
            attidx_clabel_to_collcounter[attidx][clabel] = Counter(c_entries_np[:,attidx])
            
    return attidx_clabel_to_collcounter


def create_policy(entries, eidx_to_clabel, centroids, k, th):
    #entries is a list of dataset entries
    #eidx_to_clabel is a list containing the cluster labels of each entry
    #centroids is the list of centroids. Note: centroids[i] corresponds to clabel_i
    #k is the number of clusters
    #th is the threshold to consider effective attributes
    #Returns policy: a list of rules, where each rule is a dict of (attidx,attval) key-value's
    
    num_atts = len(entries[0])
    clabel_to_indexes = get_clusters(eidx_to_clabel, k)
    attidx_to_collcounter = get_entries_freqs(entries)
    attidx_clabel_to_collcounter = get_clusters_freqs(entries, clabel_to_indexes)
    
    l_size = len(entries) #dataset size
    
    policy = []
    
    for clabel,cen in enumerate(centroids):
        
        c_size = len(clabel_to_indexes[clabel]) #cluster size
        rule = {} #Each rule is a dictionary
        
        for attidx in range(num_atts):
            
            attval = cen[attidx]
            freq_in_l = (attidx_to_collcounter[attidx])[attval]
            freq_in_c = (attidx_clabel_to_collcounter[attidx][clabel])[attval]
        
            if freq_in_c/c_size - freq_in_l/l_size > th:
                rule[attidx] = attval
                
        if len(rule) > 0:
            policy.append(rule)
    
    return policy


def get_false_neg(pos_entries, policy):
    #TODO revisar
    false_negs = []
    
    for entry in pos_entries:
        
        #Check access for each rule
        denies_count = 0
        
        for rule in policy:
            
            res = True
            for att_idx in rule.keys():

                if entry[att_idx] != rule[att_idx]:
                    res = False
                    break
            
            if res == False:
                denies_count += 1
                
        if denies_count == len(policy):
            false_negs.append(entry)
            
    return false_negs
    
    
def get_false_pos(neg_entries, policy):
    #TODO revisar
    false_pos = []
    
    for entry in neg_entries:
        
        #Check access for each rule
        denies_count = 0
        
        for rule in policy:
            
            res = True
            for att_idx in rule.keys():

                if entry[att_idx] != rule[att_idx]:
                    res = False
                    break
            
            if res == False:
                denies_count += 1
                
        if denies_count < len(policy):
            false_pos.append(entry)
            
    return false_pos


def evaluate_policy(policy, pos_entries, neg_entries):
    #TODO revisar
    false_negs = get_false_neg(pos_entries, policy)
    FN = len(false_negs)
    TP = len(pos_entries) - FN
    false_pos = get_false_pos(neg_entries, policy)
    FP = len(false_pos)
    TN = len(neg_entries) - FP
    
    precision = TP / (TP + FP)
    
    recall = TP / (TP + FN)
    
    fscore = 2*(precision*recall)/(precision+recall)
    
    return false_negs,false_pos,precision,recall,fscore


def compute_wsc(policy):
    return sum([len(rule) for rule in policy])



#Hay que mejorarla
def num_activations(pos_entries, policy):
    #TODO revisar
    activations = []
    
    for entry in pos_entries:
        
        #Check access for each rule
        denies_count = 0
        
        for rule in policy:
            
            res = True
            for att_idx in rule.keys():

                if entry[att_idx] != rule[att_idx]:
                    res = False
                    break
            
            if res == False:
                denies_count += 1
                
        activations.append(len(policy)-denies_count)

            
    return activations