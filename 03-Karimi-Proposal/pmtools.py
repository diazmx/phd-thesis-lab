
def load_entries(filename):
    log_entries = []
    f = open(filename,'r')
    for line in f.readlines():
        if len(line) > 1:
            arr = line.split(',')
            log_entries.append([int(x) for x in arr])
    f.close()
    return log_entries
    
    
def print_entries(log_entries, attnames_list, val_labels_dict):
        
    num_att = len(attnames_list)

    for i in range(len(log_entries)):
        entry = log_entries[i]
        for j in range(num_att):
            attname = attnames_list[j]
            val = entry[j]
            if val < len(val_labels_dict[attname]):
                vallabel = val_labels_dict[attname][val]
            else:
                vallabel = attname+str(val)
            str_output = attname + ':' + vallabel
            if j < num_att-1:
                str_output += ', '
            print(str_output, end='')
        print()
        
        
def get_freqs(log_entries, entry_idxs=None):
    
    num_att = len(log_entries[0])
    freqs = [dict() for i in range(num_att)]

    if entry_idxs == None:
        entries = log_entries
    else:
        entries = [log_entries[i] for i in entry_idxs]

    for entry in entries:
        for att_idx in range(num_att):
            val = entry[att_idx]
            
            if val in freqs[att_idx]:
                freqs[att_idx][val] += 1
            else:
                freqs[att_idx][val] = 1
            
    return freqs


def print_freqs(total_freqs, attnames_list, val_labels_dict):

    num_att = len(attnames_list)
    for i in range(num_att):
        attname = attnames_list[i]
        print('---'+attname+'---')
        for val in total_freqs[i]:
            
            if val < len(val_labels_dict[attname]):
                vallabel = val_labels_dict[attname][val]
            else:
                vallabel = attname+str(val)
            
            val_frec = total_freqs[i][val]
            print(vallabel,':',val_frec,end=', ')
        print()
        


def extract_rules(log_entries, threshold, cluster_labels, centroids, verbose=True, att_names=None, val_labels_dict=None):
    
    num_entries = len(log_entries)
    num_att = len(log_entries[0])
    total_freqs = get_freqs(log_entries, range(num_entries))
    
    num_clusters = len(centroids)
    label_to_indexes = [list() for i in range(num_clusters)]
    
    for i in range(num_entries):
        label = cluster_labels[i]
        label_to_indexes[label].append(i)
        
        
    rules = []
    cluster_idx = 0
    for entry_idxs in label_to_indexes: #For each cluster
        
        if verbose:
            print('Cluster:', cluster_idx, 'Effective attributes:')
        
        freqs_cluster = get_freqs(log_entries, entry_idxs)

        rule = dict()
        
        for att_idx in range(num_att):
        
            attvals_to_freq = freqs_cluster[att_idx]
        
            for attval in attvals_to_freq.keys():
                
                level = attvals_to_freq[attval]/len(entry_idxs) - total_freqs[att_idx][attval]/len(log_entries)
                
                if level >= threshold: #Check just with greater than
                    
                    if att_idx in rule:
                        rule[att_idx].add(attval)
                    else:
                        rule[att_idx] = {attval}
                    
                    if verbose:
                        attname = att_names[att_idx]
                        if attval < len(val_labels_dict[attname]):
                            vallabel = val_labels_dict[attname][attval]
                        else:
                            vallabel = attname+str(attval)
                        print('---',attname, '=', vallabel, 'count =', attvals_to_freq[attval])
                        print('level =', level)
                    
        cluster_idx += 1
        rules.append(rule)
        
        if verbose:
            print()
            
    return rules
    

def print_centroids(centroids, attnames_list, val_labels_dict):
    
    for centroid in centroids:
        print('[', end='')
        
        for att_idx in range(len(centroid)):
            attval = centroid[att_idx]
            attname = attnames_list[att_idx]
            
            if attval < len(val_labels_dict[attname]):
                vallabel = val_labels_dict[attname][attval]
            else:
                vallabel = attname+str(attval)
            
            str_output = attname + "=" + vallabel
            
            if att_idx < len(centroid)-1:
                print(str_output, end=', ')
            else:
                print(str_output, end='')
                 
        print(']')


def print_policy(policy, attnames_list, val_labels_dict):
    
    rule_id=0
    for rule in policy:
    
        print('Rule'+str(rule_id+1))
        for att in rule.keys():
            
            attname = attnames_list[att]
            print(attname, end=' = ')
            
            for attval in rule[att]:
                if attval < len(val_labels_dict[attname]):
                    vallabel = val_labels_dict[attname][attval]
                else:
                    vallabel = attname+str(attval)
                print(vallabel, end=' ')
                
        print()
        rule_id += 1


  
def compute_wsc(policy):
    return sum([len(rule) for rule in policy])
    
    
    
def get_false_neg(msg_logs, policy):
    
    false_negs = []
    
    for entry in msg_logs:
        
        #Check access for each rule
        denies_count = 0
        
        for rule in policy:
            
            res = True
            for att_idx in rule.keys():

                if entry[att_idx] not in rule[att_idx]:
                    res = False
                    break
            
            if res == False:
                denies_count += 1
                
        if denies_count == len(policy):
            false_negs.append(entry)
            
    return false_negs
    
    
    
def get_false_pos(neg_msg_logs, policy):
    
    false_pos = []
    
    for entry in neg_msg_logs:
        
        #Check access for each rule
        denies_count = 0
        
        for rule in policy:
            
            res = True
            for att_idx in rule.keys():

                if entry[att_idx] not in rule[att_idx]:
                    res = False
                    break
            
            if res == False:
                denies_count += 1
                
        if denies_count < len(policy):
            false_pos.append(entry)
            
    return false_pos



def evaluate_policy(policy, pos_entries, neg_entries):

    false_negs = get_false_neg(pos_entries, policy)
    FN = len(false_negs)
    TP = len(pos_entries) - FN
    FNR = FN/len(pos_entries)
    
    false_pos = get_false_pos(neg_entries, policy)
    FP = len(false_pos)
    TN = len(neg_entries) - FP
    FPR = FP/len(neg_entries)
    
    accuracy = (TP + TN)/(len(pos_entries) + len(neg_entries))
    
    return false_negs,false_pos,FNR,FPR,accuracy
    

    
def jaccard_similarity(rule1, rule2, num_att):
    inter_card = 0
    union_card = 0
    for att in range(num_att):
        
        set1 = rule1[att] if (att in rule1) else set()
        set2 = rule2[att] if (att in rule2) else set()
        inter_card += len(set1&set2)
        union_card += len(set1|set2)
        
    return inter_card/union_card


def centroids_to_rules(centroids):
    num_att = len(centroids[0])
    policy = []
    for centroid in centroids:
        rule = dict()
        for att in range(num_att):
            if att not in rule:
                rule[att] = {centroid[att]}
            else:
                rule[att].add(centroid[att])
        policy.append(rule)
        
    return policy    