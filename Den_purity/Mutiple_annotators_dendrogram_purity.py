import json
import numpy as np
import os
import re
from sklearn.metrics.cluster import normalized_mutual_info_score as NMI
from collections import defaultdict
from ete3 import Tree
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
from itertools import combinations
import json
import numpy as np
import os
import re
from sklearn.metrics.cluster import normalized_mutual_info_score as NMI
from collections import defaultdict
from ete3 import Tree
from dendrogram_purity import *
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
from itertools import combinations
import sys
sys.path.append("..") 
from config import conf


def _distance(pt_1, pt_2):
    pt_1 = np.array(pt_1)
    pt_2 = np.array(pt_2)
    return np.linalg.norm(pt_1-pt_2)

def _pdist(pts):
    com_pts=(list(combinations(pts,2)))
    all_pts_dis=[]
    for pts_pairs in com_pts:
        pts_dis=_distance(pts_pairs[0],pts_pairs[1])
        all_pts_dis.append(pts_dis)
    return all_pts_dis
    
def get_sub_labels(sub_names,labels,names):
    sub_labels=[]
    for i in sub_names:
        sub_labels.append(labels[names.index(i)])
    return sub_labels

def get_real_labels(number_labels,class_names):
    real_labels=[]
    for i in number_labels:
        real_labels.append(class_names[i])
    return real_labels

def get_H_lables(number_labels,class_names,standard):
   
    real_lables=get_real_labels(number_labels,class_names)
    
    H_labels=[]

    for i in real_lables:
        for s in range(len(standard)):
            if i in standard[s][0]:
                H_labels.append(standard[s][1])
    return H_labels

def check_singel_cluser(cluser,labels,names,class_names):
    
    sub_labels=get_sub_labels(cluser,labels,names)
    sub_real_labels=get_real_labels(sub_labels,class_names)

    for sub_real_label in sorted(list(set(sub_real_labels))):
        i=0
        for every_sample in sub_real_labels:
            if every_sample==sub_real_label:
                i+=1
        print (str(sub_real_label),'num',str(i))
        
        
def check_mutiple_clusers(all_clusers,labels,names,class_names):
    
    layer_num=0
    for layer_clusters in all_clusers:
        print ('\n',"In layer",str(layer_num),"there is",len(layer_clusters),"clusers")
        
        cluser_num=0
        for cluser in layer_clusters:
            print ('\n',"In layer",str(layer_num),"cluser",str(cluser_num))
            cluser_num+=1
            
            sub_labels=get_sub_labels(cluser,labels,names)
            sub_real_labels=get_real_labels(sub_labels,class_names)

            for sub_real_label in sorted(list(set(sub_real_labels))):
                i=0
                for every_sample in sub_real_labels:
                    if every_sample==sub_real_label:
                        i+=1
                print (str(sub_real_label),'num',str(i))    
                
                
        layer_num+=1

def get_cluser_lists(pre_lable,names):
    sortedClusters={}
    sortedClusters = defaultdict(list)
    for i in range(len(pre_lable)):
        c = pre_lable[i]
        n = names[i]
        sortedClusters[str(c)].append(n)

    cluser_lists=[]
    for i in range(len(sortedClusters)):
        cluser_lists.append((sortedClusters[str(i)]))
        
    return cluser_lists

def get_cluster_lists_distance(names,embeddings,max_cluser_layer,s_score_shrehold):
    s_scores=[]
    labels=[]
    centers=[]
    
    
    
    if len(embeddings)< max_cluser_layer+1:
        max_cluser_layer=len(embeddings)-1
    if len(embeddings)==2:
        print ("not spilit because only 2 samples",'\n')
        return 1,0,_distance(embeddings[0], embeddings[1]),names
    if len(embeddings)==1:
        print ("not spilit because only 1 samples",'\n')
        return 1,0,0,names
    for n_cluser in range(2,max_cluser_layer+1):
        kmeans = KMeans(n_cluser, random_state=0).fit(embeddings)
        s_score=silhouette_score(embeddings, kmeans.labels_)
        
        print (str(n_cluser),'s_score',str(s_score))
        
#         every_dis=np.mean(_pdist(kmeans.cluster_centers_))
#         print ("This center distance is ",every_dis)
        
        s_scores.append(s_score)

        labels.append(kmeans.labels_)
        centers.append(kmeans.cluster_centers_) 
    max_s_score=max(s_scores)
    index=s_scores.index(max_s_score)
#     index=0
    cluser_num=index+2
    if max_s_score > s_score_shrehold:
        print ("max_s_score",max_s_score, "split into",str(cluser_num),'clusters')
    else:
        print ("max_s_score",max_s_score, " wont split into",str(cluser_num),'clusters','coz s_score is less than',str(s_score_shrehold))
    


    center=centers[index]
    dis=np.mean(_pdist(center))
    print ("Max_s_score Center distance is ",dis,'\n')
    
    
    pre_lable=labels[index]	
    cluser_lists=get_cluser_lists(pre_lable,names)
    
    
    return cluser_num,max_s_score,dis,cluser_lists



def manually_get_cluster_lists_distance(names,embeddings,max_cluser_layer,human_design_cluser):
    s_scores=[]
    labels=[]
    centers=[]
    
    
    if len(embeddings)==2:
        print ("not spilit because only 2 samples",'\n')
        return 1,0,_distance(embeddings[0], embeddings[1]),names
    if len(embeddings)==1 :
        print ("not spilit because only 1 samples",'\n')
        return 1,0,0,names
    
    if  human_design_cluser==1:
        print ("not spilit because we dont want it split anymore",'\n')
        return 1,0,0,names
    

    kmeans = KMeans(human_design_cluser, random_state=0).fit(embeddings)
    cluser_num=human_design_cluser

    center=kmeans.cluster_centers_
    dis=np.mean(_pdist(center))

    pre_lable=kmeans.labels_
    cluser_lists=get_cluser_lists(pre_lable,names)
        
    print ("Human designed split into",str(cluser_num),'clusters')
    print ("Max_s_score Center distance is ",dis,'\n')
    max_s_score=1
        
    return cluser_num,max_s_score,dis,cluser_lists
        
        




def get_all_clusers_distances(names,embeddings,max_cluser_layer,cluster_depth,s_score_shrehold,human_design,human_design_cluser):
    max_allow_clusers=30
    layer_clusers=names
    pick_clusers_distances=[]
    all_clusers_distances=[]
    remin_clusers=[]
    
    
    
    for _ in range (cluster_depth):
        
        print (str(_),'layer started','\n')

        layer_clusers_distances=[]
        
        not_split_clusers=[]
        split_clusers=[]
        
        new_layer_clusers=[]
        

        if type(layer_clusers[0]) != list:
            layer_clusers=[layer_clusers]

        not_split_indexs=[]
        layer_cluser_index=0
        
        for layer_cluser in layer_clusers :
        
            sub_names=layer_cluser
            sub_embeddings=[]
            
            for sub_name in sub_names:
                sub_embeddings.append(embeddings[names.index(sub_name)])

            if human_design:
                cluser_num,max_s_score,dis,cluser_lists=manually_get_cluster_lists_distance(sub_names,sub_embeddings,max_cluser_layer,human_design_cluser[_][layer_cluser_index])
            else:
                cluser_num,max_s_score,dis,cluser_lists=get_cluster_lists_distance(sub_names,sub_embeddings,max_cluser_layer,s_score_shrehold)
            
            layer_clusers_distances.append((sub_names,dis))
            
            if max_s_score > s_score_shrehold:
                new_layer_clusers.extend(cluser_lists)
                split_clusers.append((sub_names,dis))
            else :
                not_split_clusers.append((sub_names,dis))
                not_split_indexs.append(layer_cluser_index)
            
            layer_cluser_index+=1
            
            
        all_clusers_distances.append(layer_clusers_distances+remin_clusers)
        pick_clusers_distances.append(layer_clusers_distances) 
        
        print (len(layer_clusers_distances),'to be split',"clusters ")
        print (len(not_split_clusers),'not split',"clusters ")
        if len(not_split_clusers)!=0:
            print ("index is",not_split_indexs)
        print (len(split_clusers),'split',"clusters ")
        print ('split into',len(new_layer_clusers)," clusters ")
        print (len(layer_clusers_distances+remin_clusers),'all',"clusters ",'\n')

        
        remin_clusers.extend(not_split_clusers)
        layer_clusers = new_layer_clusers
        
        if len(layer_clusers)==0:
            print ('stopped in',str(_),'layer because not split anymore')
            break

        if len(layer_clusers_distances+remin_clusers)>max_allow_clusers:
            print ('stopped in',str(_),'layer bacause already more than',str(max_allow_clusers),'clusers')
            break
            
    return pick_clusers_distances,all_clusers_distances


### Confirations
# algorithm_name=conf.algorithm_name
# algorithm_name="mutiple_update_general_distance_no_active_all_trainable"


DIM = conf.img_dim_1
DATASET = conf.dataset

cluster_depth=conf.cluster_depth
depth=conf.depth
max_cluser_layer=conf.max_cluser_layer
s_score_shrehold=0.05


human_design_cluser=[[2],[2,2],[4,2,2,2],[1,1,1,1,1,1,1,1,1,1]]
human_design=0



training=1
ITRS= range(4,6)
annotators=['Annotator_1','Annotator_2','Annotator_3']

for annotator in annotators:
    
    for ITR in ITRS:


        ### loading data
        if training:
            traning_flag="Train"
            PATH = os.path.join('..','data', DATASET,'train')
        else:
            traning_flag="Test"
            PATH = os.path.join('..','data',DATASET,'test')
        
        with open(os.path.join(PATH, str(annotator)+'_'+DATASET +'_embeddings_' + str(DIM) + 'x' + str(DIM) +'_'+ str(ITR) + '.json')) as f:

            embeddings = json.load(f)
            embeddings = np.array(embeddings)

        with open(os.path.join(PATH, DATASET + '_decoded_{}x{}.json'.format(DIM, DIM))) as f:
            images = json.load(f)
            images = np.array(images)

        with open(os.path.join(PATH, DATASET + '_img_names'+'.json')) as f:

            names = json.load(f)
            
        with open(os.path.join(PATH, DATASET + '_labels'+'.json')) as f:
            labels = json.load(f)

        with open(os.path.join(PATH, str(annotator)+'_'+DATASET + '_triplets_' + str(ITR) + '.json')) as f:

            triplets = json.load(f)
            triplets = np.array(triplets)



        ### get H clusers
        pick_clusers_distances,all_clusers_distances=get_all_clusers_distances(names,embeddings,max_cluser_layer,cluster_depth+depth,s_score_shrehold,human_design,human_design_cluser)

        p_all_clusers=[]
        for i in range(len(pick_clusers_distances)):
            p_layer_clusers=[]
            for p_layer_cluser_dis in pick_clusers_distances[i]:
                p_layer_clusers.append(p_layer_cluser_dis[0])
        
            p_all_clusers.append(p_layer_clusers)


        a_all_clusers=[]
        for i in range(len(all_clusers_distances)):
            a_layer_clusers=[]
            for a_layer_cluser_dis in all_clusers_distances[i]:
                a_layer_clusers.append(a_layer_cluser_dis[0])
        
            a_all_clusers.append(a_layer_clusers)




        ### get pure clusers
        class_names =['airplane' ,'automobile' ,'bird','cat' ,'deer' ,'dog' ,'frog' ,'horse','ship' ,'truck' ]
        for sub_clusers in class_names:
            globals()[sub_clusers] = []    
            for i in range(len(names)):
                if class_names[labels[i] ] == sub_clusers:
                    globals()[sub_clusers].append(names [i])





        ### get real H clusers

        if len(a_all_clusers)==2:
            layer_1=a_all_clusers[1]
            T_cluser_1=layer_1
            
            T_cluser=T_cluser_1

            t3=get_1_layer(T_cluser)

        elif len(a_all_clusers)==3:
            
            layer_1=a_all_clusers[1]
            layer_2=a_all_clusers[2]

            H_cluser_1=[]
            for cluser_1 in layer_1:

                H_cluser_2=[]
                for cluser_2 in layer_2:

                    if set(cluser_2).issubset(cluser_1):
                        H_cluser_2.append(cluser_2)

                H_cluser_1.append(H_cluser_2)


            T_cluser_2 = H_cluser_1
            
            T_cluser=T_cluser_2
            t3=get_2_layer(T_cluser)
            
        else:
            layer_1=a_all_clusers[1]
            layer_2=a_all_clusers[2]
            layer_3=a_all_clusers[3]


            H_cluser_1=[]
            for cluser_1 in layer_1:

                H_cluser_2=[]
                for cluser_2 in layer_2:
                    if set(cluser_2).issubset(cluser_1):


                        H_cluser_3=[]
                        for cluser_3 in layer_3:

                            if set(cluser_3).issubset(cluser_2):
                                H_cluser_3.append(cluser_3)

                        H_cluser_2.append(H_cluser_3)

                H_cluser_1.append(H_cluser_2)



            T_cluser_3 = H_cluser_1
            
            T_cluser = T_cluser_3
            t3=get_3_layer(T_cluser)


        ### get expected H clusers

        if annotator=='Annotator_1':
    
            expect_tree3=[airplane,automobile,bird,cat,deer,dog,frog,horse,ship,truck,
                                    airplane+automobile+ship+truck,dog+cat+bird+frog+horse+deer,
                                    dog+cat+bird+frog,horse+deer,automobile+truck,airplane+ship]

        elif annotator=='Annotator_2':
            expect_tree3=[airplane,automobile,bird,cat,deer,dog,frog,horse,ship,truck,
                                airplane+automobile+ship+truck,dog+cat+bird+frog+horse+deer,
                                dog+cat+horse+deer,bird+frog,ship+truck,airplane+automobile]

        elif annotator=='Annotator_3':
            expect_tree3=[airplane,automobile,bird,cat,deer,dog,frog,horse,ship,truck,
                                airplane+automobile+ship+truck,dog+cat+bird+frog+horse+deer,
                                deer+horse+bird+frog,cat+dog]

        

    

        ### get dendrogram_purity

        purity=dendrogram_purity(t3,expect_tree3)


        print (str(annotator),"in",str(ITR),"dendrogram_purity is",str(purity))