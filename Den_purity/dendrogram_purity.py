# implement dengram purity for ete.Tree structure
from ete3 import Tree
from itertools import combinations

import random

def lca(tree, i, j):
    node_i = tree.search_nodes(name = str(i))[0]
    node_j = tree.search_nodes(name = str(j))[0]
    node_i_parent_list = node_i.get_ancestors()
    node_j_parent_list = node_j.get_ancestors()
    pointer_i = len(node_i_parent_list) - 1
    pointer_j = len(node_j_parent_list) - 1
    while(pointer_i >= 0 and pointer_j >= 0):
        if(node_i_parent_list[pointer_i] != node_j_parent_list[pointer_j]):
            break
        pointer_i -= 1
        pointer_j -= 1
    common_parent = node_i_parent_list[pointer_i+1]
    leaves = []
    for i in common_parent.get_leaves():
        leaves.append((i.name))
    return leaves

def set_purity(A, B):
    numerator = len(set(A).intersection(B))
    denominator = len(A)
    return numerator*1.0/denominator


def dendrogram_purity(tree, leaf_partition):
    purity = 0
    cnt = 0
    cluser_num=0
    for Ck in leaf_partition:
        print ("Begin calculate in he",str(cluser_num),'cluser')
        cluser_num+=1
        for i in range(len(Ck)):
            for j in range(i+1, len(Ck)):
                
                index_i = Ck[i]
                index_j = Ck[j]
                cnt += 1
                lca_set = lca(tree, index_i, index_j)
                purity += set_purity(lca_set, Ck)
                
    purity /= cnt
    return purity


# def dendrogram_purity_approx(approxy_num,tree, leaf_partition):
#     purity = 0
#     cnt = 0
#     cluser_num=0
#     for Ck in leaf_partition:
#         print ("Begin calculate in he",str(cluser_num),'cluser')
#         cluser_num+=1
#         com=list(combinations(range(len(Ck)),2))
#         print ('Totally combination num',len(com))
#         for i,j in random.sample(com,approxy_num):              
#             index_i = Ck[i]
#             index_j = Ck[j]
#             cnt += 1
#             lca_set = lca(tree, index_i, index_j)
#             purity += set_purity(lca_set, Ck)
                
#     purity /= cnt
#     return purity



def dendrogram_purity_approx(approxy_num,tree, leaf_partition):
    purity = 0
    cnt = 0
    cluser_num=0
    for Ck in leaf_partition:
        print ("Begin calculate in he",str(cluser_num),'cluser')
        cluser_num+=1
        com=list(combinations(range(len(Ck)),2))
        print ('Totally combination num',len(com))
        if len(com)<approxy_num:
            approxy_num=len(com)
        for i,j in random.sample(com,approxy_num):              
            index_i = Ck[i]
            index_j = Ck[j]
            cnt += 1
            lca_set = lca(tree, index_i, index_j)
            purity += set_purity(lca_set, Ck)
                
    purity /= cnt
    return purity
    
def dendrogram_purity_approx2(approxy_num,tree, leaf_partition):
    purity = 0
    cnt = 0
    cluser_num=0
    for Ck in leaf_partition:
        print ("Begin calculate in he",str(cluser_num),'cluser')
        cluser_num+=1

        for _ in range(approxy_num):
            index_i,index_j =random.sample(Ck,2)             
            cnt += 1
            lca_set = lca(tree, index_i, index_j)
            purity += set_purity(lca_set, Ck)
                
    purity /= cnt
    return purity




def get_3_layer(loop_3):
    t3='('
    for loop_2 in loop_3:
        t2='('
        for loop_1 in loop_2:  
            t1='('
            for loop_0 in loop_1:  
                t0='('
                for sample in loop_0:
                    t0+=(str(sample)+',')
                t0=t0[:-1]
                t0+=')'
                t1+=t0+','        
            t1=t1[:-1]   
            t1+=')'
            t2+=t1+','
        t2=t2[:-1]   
        t2+=')'
        t3+=t2+','
    t3=t3[:-1]   
    t3+=')'
    t3+=';'
    return Tree(t3)

def get_2_layer(loop_2):   
    t2='('
    for loop_1 in loop_2:  
        t1='('
        for loop_0 in loop_1:  
            t0='('
            for sample in loop_0:
                t0+=(str(sample)+',')
            t0=t0[:-1]
            t0+=')'
            t1+=t0+','        
        t1=t1[:-1]   
        t1+=')'
        t2+=t1+','
    t2=t2[:-1]   
    t2+=')'
    t2+=';'
    return Tree(t2)

def get_1_layer(loop_1):  
    t1='('
    for loop_0 in loop_1:  
        t0='('
        for sample in loop_0:
            t0+=(str(sample)+',')
        t0=t0[:-1]
        t0+=')'
        t1+=t0+','        
    t1=t1[:-1]   
    t1+=')'
    t1+=';'
    return Tree(t1)