
import os, json, random,sys
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from collections import defaultdict
import operator as op
from functools import reduce
from sklearn.cluster import KMeans
import math
from scipy.stats import dirichlet
import scipy
sys.path.append("..") 
from config import conf
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.cluster.hierarchy import fcluster
from itertools import combinations
from sklearn.metrics import silhouette_samples, silhouette_score

def load_interation():
    """
    load_interations
    """
    with open(os.path.join('..', 'interation.txt'), "r") as f:    
        interation = f.read()   
        interation=int(interation)
    return interation


def save_interation(data):
    """
    save_interations
    """
    with open(os.path.join('..', 'interation.txt'),"w") as f:
            f.write(str(data))  





# deternine train or test path

train_path = os.path.join('..','data', conf.dataset, 'train')

test_path = os.path.join('..','data', conf.dataset, 'test')

path_data_app = os.path.join('..','ClickingApp', 'src', 'assets', 'data')



# Help functions#

def save_questions(questions):
    with open(os.path.join(path_data_app, 'newQuestions.json'), 'w') as outfile:

        if type(questions)==list:
            print('saving size ', len(questions))
            json.dump(questions, outfile)
        else:
            print('saving size ', len(questions.tolist()))
            json.dump(questions.tolist(), outfile)
        

def save_first_interation_answers(train,triplets,annotator):
    if train:
        path=train_path
    else:
        path=test_path

    if load_interation()==0:
        with open(os.path.join(path, str(annotator)+'_'+conf.dataset + '_triplets_'+str(load_interation())+'.json'), 'w') as outfile:

            if type(triplets)==list:
                print(str(annotator),'saving answers size ', len(triplets))
                json.dump(triplets, outfile)
            else:
                print(str(annotator),'saving answers size ', len(triplets.tolist()))
                json.dump(triplets.tolist(), outfile)




def save_answers_distances(train,triplets,distances,annotator):
    if train:
        path=train_path
    else:
        path=test_path



        
    # load prior and Concatenate 
    all_triplets=[]
    with open(os.path.join(path, str(annotator)+'_'+conf.dataset + '_triplets_'+str(load_interation()-1)+'.json'))as f:
        prior_triplets = json.load(f)
        all_triplets.extend(prior_triplets)
    all_triplets.extend(triplets)
    
            
    all_distance=[]
    with open(os.path.join(path,  str(annotator)+'_'+conf.dataset + '_distances_'+str(load_interation()-1)+'.json'))as f:
        prior_distance = json.load(f)
        all_distance.extend(prior_distance)
    all_distance.extend(distances)
    
    # mix
    mix = list(zip(all_triplets, all_distance))
    random.shuffle(mix)
    all_triplets[:], all_distance[:] = zip(*mix)
    
    
    
    # save_result
    with open(os.path.join(path,  str(annotator)+'_'+conf.dataset + '_triplets_'+str(load_interation())+'.json'), 'w') as outfile:

        if type(all_triplets)==list:
            print( str(annotator),'saving answers size ', len(all_triplets))
            json.dump(all_triplets, outfile)
        else:
            print( str(annotator),'saving answers size ', len(all_triplets.tolist()))
            json.dump(all_triplets.tolist(), outfile)

    with open(os.path.join(path, str(annotator)+'_'+ conf.dataset + '_distances_'+str(load_interation())+'.json'), 'w') as outfile:

        if type(all_distance)==list:
            print(str(annotator),'saving distances size ', len(all_distance))
            json.dump(all_distance, outfile)
        else:
            print(str(annotator),'saving distances size ', len(all_distance.tolist()))
            json.dump(all_distance.tolist(), outfile)



def update_distance(train,names,embeddings,pick_clusers_distances,cluster_depth,annotator):
    if train:
        path=train_path
    else:
        path=test_path


    print (str(annotator),'begin updating')  
    
    triplets=load_answered_triplets(train,annotator)
    
    new_distances=[]
    new_triplets=[]
    
    
    
    if cluster_depth>len(pick_clusers_distances):
        cluster_depth=len(pick_clusers_distances)
        
    for num in reversed(range(cluster_depth)):

        print ('in layer',num,'begining have answers',len(triplets))  

        if len(triplets)==0:
            print ("Finished this layer")
            break
    
        found_triplets=[]
        
        for p_layer_cluser_dis in pick_clusers_distances[num]:
            p_clusers,distance=p_layer_cluser_dis

            for triplet in triplets:
                if set(triplet).issubset(p_clusers):
                    found_triplets.append(triplet)
                    new_triplets.append(triplet)
                    new_distances.append(distance)

            if len(found_triplets)!=0:
                print ('in  cluser of layer ',num,'found',len(found_triplets))  

        triplets= [ i for i in triplets if i not in found_triplets ]
        
    mix = list(zip(new_distances, new_triplets))
    random.shuffle(mix)
    new_distances[:], new_triplets[:] = zip(*mix)
    
    saved_triplets=[]
    for new_triplet in new_triplets:
        saved_triplet=({'Anchor': new_triplet[0],'positive':new_triplet[1],'negative':new_triplet[2]})
        saved_triplets.append(saved_triplet)



    with open(os.path.join(path, str(annotator)+'_'+conf.dataset + '_distances_'+str(load_interation()-1)+'.json'), 'w') as outfile:

        if type(new_distances)==list:
            print('updating distances size ', len(new_distances))
            json.dump(new_distances, outfile)
        else:
            print('updating distances size ', len(new_distances.tolist()))
            json.dump(new_distances.tolist(), outfile)   

        
    with open(os.path.join(path, str(annotator)+'_'+conf.dataset + '_triplets_'+str(load_interation()-1)+'.json'), 'w') as outfile:

        if type(saved_triplets)==list:
            print('updating answers size ', len(saved_triplets))
            json.dump(saved_triplets, outfile)
        else:
            print('updating answers size ', len(saved_triplets.tolist()))
            json.dump(saved_triplets.tolist(), outfile)



def get_triplets_distance(train,names,embeddings,triplets,pick_clusers_distances,cluster_depth,annotator):
    if train:
        path=train_path
    else:
        path=test_path


    print (str(annotator),'getting triplets and distances')  
    

    new_distances=[]
    new_triplets=[]
    
    
    
    if cluster_depth>len(pick_clusers_distances):
        cluster_depth=len(pick_clusers_distances)
        
    for num in reversed(range(cluster_depth)):

        print ('in layer',num,'begining have answers',len(triplets))  

        if len(triplets)==0:
            print ("Finished this layer")
            break
    
        found_triplets=[]
        
        for p_layer_cluser_dis in pick_clusers_distances[num]:
            p_clusers,distance=p_layer_cluser_dis

            for triplet in triplets:
                if set(triplet).issubset(p_clusers):
                    found_triplets.append(triplet)
                    new_triplets.append(triplet)
                    new_distances.append(distance)

            if len(found_triplets)!=0:
                print ('in  cluser of layer ',num,'found',len(found_triplets))  

        triplets= [ i for i in triplets if i not in found_triplets ]
        
    mix = list(zip(new_distances, new_triplets))
    random.shuffle(mix)
    new_distances[:], new_triplets[:] = zip(*mix)
    
    return new_triplets,new_distances


def load_names(train):
    """
    :return:    return names
    """

    if train:
        path=train_path
    else:
        path=test_path

    with open(os.path.join(path, conf.dataset + '_img_names.json')) as f:
        names = json.load(f)
    
    # check if data/ is in front of names
    if names[0][:4] != 'data':
        _names = []
        for name in names:
            _names.append('data/'+name)
    else:
        _names = names

    return _names

def load_labels(train):
    """
    :return:   labels
    """
    if train:
        path=train_path
    else:
        path=test_path

    with open(os.path.join(path, conf.dataset + '_labels.json')) as f:
        labels = json.load(f)

    return labels

def load_embeddings(train,annotator):

    if train:
        path=train_path
    else:
        path=test_path

    with open(os.path.join(path, str(annotator)+'_'+conf.dataset + '_embeddings_' + str(conf.img_dim_1) + 'x' + str(conf.img_dim_2) +'_'+ str(load_interation()-1) + '.json')) as f:
        embeddings = json.load(f)
    
    return embeddings

def load_answered_triplets(train,annotator):

    """
    return: triplets
    """

    if train:
        path=train_path
    else:
        path=test_path


    with open(os.path.join(path, str(annotator)+'_'+conf.dataset + '_triplets_'+str(load_interation()-1)+'.json'))as t:
        triplets = json.load(t)
        triplets = np.array(triplets)
    
    all_answers = []
    for triplet in triplets:
        if triplet['Anchor'][:4] == 'data':
            a = [triplet['Anchor'], triplet['positive'], triplet['negative']]
        else:
            a = ['data/'+triplet['Anchor'], 'data/'+triplet['positive'], 'data/'+triplet['negative']]
        all_answers.append(a)

    return all_answers

def ini_distance(train,num,annotator):

    if train:
        path=train_path
    else:
        path=test_path

    ini_distance=([0]*num)
        
    with open(os.path.join(path, str(annotator)+'_'+conf.dataset + '_distances_'+str(load_interation())+'.json'), 'w') as outfile:

        if type(ini_distance)==list:
            print('\n',str(annotator),'initial distance size ', len(ini_distance),'\n')
            json.dump(ini_distance, outfile)
        else:
            print('\n',str(annotator),'initial distance  size ', len(ini_distance.tolist()),'\n')
            json.dump(ini_distance.tolist(), outfile)

def Concatenate(train,annotators,mode,names,embeddings,cluster_depth,depth,max_cluser_layer,s_score_shrehold):
    
    if train:
        path=train_path
    else:
        path=test_path
        
    generel_triplets=[]
    
    # get general triplets
    for annotator in annotators:

        with open(os.path.join(path, str(annotator)+'_'+conf.dataset + '_triplets_'+str(load_interation())+'.json'))as t:
            triplets = json.load(t)
            generel_triplets.extend(triplets)

        
    # get general distances      
    if mode=='remain':
        general_distances=[] 
        for annotator in annotators:
            with open(os.path.join(path, str(annotator)+'_'+conf.dataset + '_distances_'+str(load_interation())+'.json'))as t:
                distances = json.load(t)
                general_distances.extend(distances)
                
    elif mode=="clear":
        general_distances=[0]*len(generel_triplets)
    

    elif mode=="update":
    

        annotator="general"
        pick_clusers_distances,all_clusers_distances=get_all_clusers_distances(names,embeddings,max_cluser_layer,cluster_depth+depth,s_score_shrehold)
    
        print (str(annotator),'begin updating')  
        
        
        triplets = []
        for triplet in generel_triplets:   
            a = [triplet['Anchor'], triplet['positive'], triplet['negative']]
            triplets.append(a)


        new_distances=[]
        new_triplets=[]


        if cluster_depth>len(pick_clusers_distances):
            cluster_depth=len(pick_clusers_distances)

        for num in reversed(range(cluster_depth)):

            print ('in layer',num,'begining have answers',len(triplets))  

            if len(triplets)==0:
                print ("Finished this layer")
                break

            found_triplets=[]

            for p_layer_cluser_dis in pick_clusers_distances[num]:
                p_clusers,distance=p_layer_cluser_dis

                for triplet in triplets:
                    if set(triplet).issubset(p_clusers):
                        found_triplets.append(triplet)
                        new_triplets.append(triplet)
                        new_distances.append(distance)

                if len(found_triplets)!=0:
                    print ('in  cluser of layer ',num,'found',len(found_triplets))  

            triplets= [ i for i in triplets if i not in found_triplets ]



        saved_triplets=[]
        for new_triplet in new_triplets:
            saved_triplet=({'Anchor': new_triplet[0],'positive':new_triplet[1],'negative':new_triplet[2]})
            saved_triplets.append(saved_triplet)
        
        
        
        generel_triplets = saved_triplets
        general_distances=new_distances
        
        
    # mix resultes
    mix = list(zip(generel_triplets, general_distances))
    random.shuffle(mix)
    generel_triplets[:], general_distances[:] = zip(*mix)

    # save results
    


    annotator='general'

    with open(os.path.join(path, str(annotator)+'_'+conf.dataset + '_distances_'+str(load_interation())+'.json'), 'w') as outfile:

        if type(general_distances)==list:
            print(str(annotator),'updating distances size ', len(general_distances))
            json.dump(general_distances, outfile)
        else:
            print(str(annotator),'updating distances size ', len(general_distances.tolist()))
            json.dump(general_distances.tolist(), outfile)   

        
    with open(os.path.join(path, str(annotator)+'_'+conf.dataset + '_triplets_'+str(load_interation())+'.json'), 'w') as outfile:

        if type(generel_triplets)==list:
            print(str(annotator),'updating answers size ', len(generel_triplets))
            json.dump(generel_triplets, outfile)
        else:
            print(str(annotator),'updating answers size ', len(generel_triplets.tolist()))
            json.dump(generel_triplets.tolist(), outfile)

        



def simulate_answer_questions(questions,names,class_names,labels,standard,annotator) :
    
    standard=standard[annotator]

    triplets=[]
    for question in questions:
        
        real_label=[]
        for que in question:
            real_label.append(class_names[(labels[names.index(que)])])


        H_label=[]
        for fine_grain in (standard):
            sets=[]
            for s in range(len(fine_grain)):
                sets+=fine_grain[s][0]
            if set(real_label).issubset(sets):
                for label in real_label:
                    for s2 in range(len(fine_grain)):
                        if label in fine_grain[s2][0]:
                            H_label.append(fine_grain[s2][1])
                break

        if H_label[0]==H_label[1]==H_label[2]:
            H_label=real_label


        if H_label[0]!=H_label[1]==H_label[2]:
            triplet=({'Anchor': question[1],'positive':question[2],'negative':question[0]})
        elif H_label[1]!=H_label[0]==H_label[2]:
            triplet=({'Anchor': question[2],'positive':question[0],'negative':question[1]})
        elif H_label[2]!=H_label[1]==H_label[0]:
            triplet=({'Anchor': question[0],'positive':question[1],'negative':question[2]})
        else :
            random.shuffle(question)
            triplet=({'Anchor': question[0],'positive':question[1],'negative':question[2]})
        
        
        if len(questions)-len(triplets) <6:
            print ("question",question)
            print ("real_label",real_label)
            print ("H_label",H_label)
            print ("triplet",triplet,'\n')

        triplets.append(triplet)
        
    return triplets
    

# Generate questions#



def get_random_questions(names,amount=100):
    questions = []
    for _ in range(amount):
        question=(random.sample(names, 3))
        if question not in questions:
            questions.append(question)

    return questions


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
    
# def get_random_questions(names,amount=100):
#     if amount*1<len(names):
#         sample_num=amount*1
#     else:
#         sample_num=len(names)

#     names= random.sample(names,sample_num)
#     total_sample=(list(combinations(names,3)))
#     total_sample_len=len(total_sample)
#     if total_sample_len < amount:
#         sample_num=total_sample_len
#     else:
#         sample_num=amount
    
#     questions = random.sample((total_sample),k=sample_num) 
#     questions=[ list (x) for x in questions]
#     return questions

def find_closest_images(image_embedding, embeddings, k):
    '''
    returns the k index of closest images to THE image
    '''
    # TODO this is messy. Can be done more efficient
    pt = []
    for i, n in enumerate(embeddings):
        dist = _distance(image_embedding, n)
        pt.append({'dist': dist, 'index': i})
    
    pt = sorted(pt, key=lambda x: x['dist'])
    indexes = []
    for p in pt:
        indexes.append(p['index'])

    return indexes[:k]

def search_similar_questions_cluser(question, names, embeddings, k=2):
    '''
    question: [Anchor,positive,negative]
    '''
    # find similar images to question image
    # find embedding of each image a, p, n
    _anchor = embeddings[names.index(question[0])]
    _pos = embeddings[names.index(question[1])]
    _neg = embeddings[names.index(question[2])]
    
    anchor = find_closest_images(_anchor, embeddings, k)
    pos = find_closest_images(_pos, embeddings, k)
    neg = find_closest_images(_neg, embeddings, k)
    # generate similar questions
    similar_question_cluster=[anchor,pos,neg]
    
    return similar_question_cluster












# generate questions and distance by Hierarchical #

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



def get_all_clusers_distances(names,embeddings,max_cluser_layer,cluster_depth,s_score_shrehold):
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
        not_split_index=0
        for layer_cluser in layer_clusers :
        
            sub_names=layer_cluser
            sub_embeddings=[]
            
            for sub_name in sub_names:
                sub_embeddings.append(embeddings[names.index(sub_name)])
            
            cluser_num,max_s_score,dis,cluser_lists=get_cluster_lists_distance(sub_names,sub_embeddings,max_cluser_layer,s_score_shrehold)
            layer_clusers_distances.append((sub_names,dis))
            
            if max_s_score > s_score_shrehold:
                new_layer_clusers.extend(cluser_lists)
                split_clusers.append((sub_names,dis))
            else :
                not_split_clusers.append((sub_names,dis))
                not_split_indexs.append(not_split_index)
            
            not_split_index+=1
            
            
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






# choose questions#
def calculate_dir (question,similar_question_cluster, answer_questions, names, embeddings,dis_weight,use_prior,use_weight):

    sc=similar_question_cluster
    prior_weight=0.1
 
    # check if question already has an answer
    if question in answer_questions:
        # return mean of 1 in order to not include same question twice
        return 0, 0, 0,[1,1,1],[0,0,0]

    o_u = embeddings[names.index(question[0])]
    o_r= embeddings[names.index(question[1])]
    o_l = embeddings[names.index(question[2])]



    # get the prior distribution
    if use_prior:

        
        prior_a =_distance(o_r,o_l)
        prior_b= _distance(o_u,o_l)
        prior_c= _distance(o_u,o_r)

        prior_all =np.linalg.norm([prior_a,prior_b,prior_c])
        a=prior_weight*(prior_all/prior_a)
        b=prior_weight*(prior_all/prior_b)
        c=prior_weight*(prior_all/prior_c)
    else:
        a = 1
        b = 1
        c = 1 
    
    
    # See if similar question are answerd and update the post-distribution
    
    for qs in answer_questions:
        if (qs[2] in sc[0]):
            
            if ((qs[1] in sc[1]) and (qs[0] in sc[2])):               
                if use_weight:                       
                    qs_u = embeddings[names.index(qs[2])]
                    qs_r = embeddings[names.index(qs[1])]
                    qs_l = embeddings[names.index(qs[0])]
                    D_u =   _distance(o_u,qs_u)
                    D_r=    _distance(o_r,qs_r)
                    D_l=    _distance(o_l,qs_l)
                    D_max  =    max(D_u,D_r,D_l)
                    weight_dir=  math.e**(dis_weight*(-D_max))
                    add_dir=weight_dir
                else:
                    add_dir=1          
                a  +=  add_dir 
                continue
 
            if ((qs[1] in sc[2]) and (qs[0] in sc[1])):               
                if use_weight:                       
                    qs_u = embeddings[names.index(qs[2])]
                    qs_r = embeddings[names.index(qs[0])]
                    qs_l = embeddings[names.index(qs[1])]
                    D_u =   _distance(o_u,qs_u)
                    D_r=    _distance(o_r,qs_r)
                    D_l=    _distance(o_l,qs_l)
                    D_max  =    max(D_u,D_r,D_l)
                    weight_dir=  math.e**(dis_weight*(-D_max))
                    add_dir=weight_dir
                else:
                    add_dir=1          
                a  +=  add_dir 
                continue
            
        if (qs[2] in sc[1]):
        
            if ((qs[1] in sc[2]) and (qs[0] in sc[0])):                
                if use_weight:                       
                    qs_u = embeddings[names.index(qs[0])]
                    qs_r = embeddings[names.index(qs[2])]
                    qs_l = embeddings[names.index(qs[1])]
                    D_u =   _distance(o_u,qs_u)
                    D_r=    _distance(o_r,qs_r)
                    D_l=    _distance(o_l,qs_l)
                    D_max  =    max(D_u,D_r,D_l)
                    weight_dir=  math.e**(dis_weight*(-D_max))
                    add_dir=weight_dir
                else:
                    add_dir=1          
                b  +=  add_dir 
                continue
            

            if ((qs[1] in sc[0]) and (qs[0] in sc[2])):
                if use_weight:                       
                    qs_u = embeddings[names.index(qs[1])]
                    qs_r = embeddings[names.index(qs[2])]
                    qs_l = embeddings[names.index(qs[0])]
                    D_u =   _distance(o_u,qs_u)
                    D_r=    _distance(o_r,qs_r)
                    D_l=    _distance(o_l,qs_l)
                    D_max  =    max(D_u,D_r,D_l)
                    weight_dir=  math.e**(dis_weight*(-D_max))
                    add_dir=weight_dir
                else:
                    add_dir=1          
                b  +=  add_dir 
                continue
                    
            
            
        if (qs[2] in sc[2]):
            if ((qs[1] in sc[0]) and (qs[0] in sc[1])) :
                if use_weight:                       
                    qs_u = embeddings[names.index(qs[1])]
                    qs_r = embeddings[names.index(qs[0])]
                    qs_l = embeddings[names.index(qs[2])]
                    D_u =   _distance(o_u,qs_u)
                    D_r=    _distance(o_r,qs_r)
                    D_l=    _distance(o_l,qs_l)
                    D_max  =    max(D_u,D_r,D_l)
                    weight_dir=  math.e**(dis_weight*(-D_max))
                    add_dir=weight_dir
                else:
                    add_dir=1          
                c  +=  add_dir 
                continue
           
            if  ((qs[1] in sc[1]) and (qs[0] in sc[0])):
                if use_weight:                       
                    qs_u = embeddings[names.index(qs[0])]
                    qs_r = embeddings[names.index(qs[1])]
                    qs_l = embeddings[names.index(qs[2])]
                    D_u =   _distance(o_u,qs_u)
                    D_r=    _distance(o_r,qs_r)
                    D_l=    _distance(o_l,qs_l)
                    D_max  =    max(D_u,D_r,D_l)
                    weight_dir=  math.e**(dis_weight*(-D_max))
                    add_dir=weight_dir
                else:
                    add_dir=1          
                c  +=  add_dir 
                continue

            

    
    alpha = np.array([a, b, c])   
    mean=dirichlet.mean(alpha)
    var= dirichlet.var(alpha)
    return a, b, c,mean, var 

def calculate_dir_cluser (questions,answer_questions,names,embeddings,distance,a_all_clusers,cluster_depth,depth,dis_weight,use_prior,use_weight):
    start_num=cluster_depth+1
    end_num  = len(a_all_clusers)
    
    quesion_everycluster_dic=[]
    neighbour_an_question=0
    
    if start_num>end_num:
        start_num=end_num


    for num in range(start_num,end_num):
        
        if len(questions)==0:
#             print ("Finished this layer")
            break
        
        clusters=a_all_clusers[num]
        
        found_questions=[]
        
        for question in questions:
            
            
            
            i=0
            for cluster in clusters:
                
                if (question[0] in cluster) and (question[1] not in cluster) and (question[2] not in cluster):
                    o_index=clusters.index(cluster)
                    i+=1
                    continue
                if (question[1] in cluster) and (question[0] not in cluster) and (question[2] not in cluster):
                    r_index=clusters.index(cluster)
                    i+=1
                    continue
                if (question[2] in cluster) and (question[0] not in cluster) and (question[1] not in cluster):
                    l_index=clusters.index(cluster)
                    i+=1
                    continue

                    
            
                if i==3:

                    similar_question_cluster=[clusters[o_index], clusters[r_index],clusters[l_index]]    
  

                    a, b, c,mean, var  = calculate_dir(question, similar_question_cluster,answer_questions, names, embeddings,dis_weight,use_prior,use_weight)
                    max_mean=max(mean) 
                    sum_var=sum(var)  
            
                    if a!=1 or b!=1 or c!=1:
                        # print ("max_mean",max_mean,"sum_var",sum_var)
                        neighbour_an_question+=1
                    
                    quesion_everycluster_dic.append({'q': question,'max_mean':max_mean, 'var': sum_var,'dis':distance})

                    found_questions.append(question)
                    break
        
#         if len(found_questions)!=0:
            
#             print ('in layer',num,'found',len(found_questions))  
#         if len(found_questions)==0:
#             print ('in layer',num,'not found')  
            
        questions= [ i for i in questions if i not in found_questions ]
        
        
        
    for question in questions:
        similar_question_cluster=search_similar_questions_cluser(question, names, embeddings, k=int(len(names)/depth))
          
        a, b, c,mean, var  = calculate_dir(question, similar_question_cluster,answer_questions, names, embeddings,dis_weight,use_prior,use_weight)
        max_mean=max(mean) 
        sum_var=sum(var)  
        if a!=1 or b!=1 or c!=1:
            # print ("max_mean",max_mean,"sum_var",sum_var)
            neighbour_an_question+=1
        quesion_everycluster_dic.append({'q': question,'max_mean':max_mean, 'var': sum_var,'dis':distance})
    # print ('not in cluster','found',len(questions)) 
    
    return quesion_everycluster_dic,neighbour_an_question

def get_potential_quesions(answer_questions,names,embeddings,pick_clusers_distances,all_clusers_distances,cluster_depth,depth,question_num,use_prior,use_weight):
    potential_quesions=[]
    potential_quesions_dic=[]
    total_answerd_neighbour_q=0
    

    #Calculate max_distance and dis_weight
    embs_dis=[]
    for i in range(10000):
        emb_sam=random.sample(embeddings,2)
        emb_dis=_distance(emb_sam[0],emb_sam[1])
        embs_dis.append(emb_dis)
    max_embs_dis=max(embs_dis)
    dis_weight=math.log(0.1)/(-max_embs_dis)
    print ("max embeddings distance is",(max_embs_dis))
    print ("dis_weight  is",(dis_weight))



    #get the pure cluster results for K means
    a_all_clusers=[]
    for i in range(len(all_clusers_distances)):
        a_layer_clusers=[]
        for a_layer_cluser_dis in all_clusers_distances[i]:
            a_layer_clusers.append(a_layer_cluser_dis[0])

        a_all_clusers.append(a_layer_clusers)
    
    #getting questions by layers

    if cluster_depth>len(pick_clusers_distances):
        cluster_depth=len(pick_clusers_distances)


    for i in reversed(range(cluster_depth)):
    # for i in [cluster_depth-1]:
        
        
        quesion_everylayer=[]
        quesion_everylayer_dic=[]
        layer_neighbour_an_question=0
        
        for p_layer_cluser_dis in pick_clusers_distances[i]:
            
            p_clusers,distance=p_layer_cluser_dis

            if len(p_clusers)>2:
                poten_quesion_everycluster=get_random_questions(p_clusers,question_num)
            
                if len(potential_quesions)==0:
                    quesion_everycluster=poten_quesion_everycluster
                else:  
                    # avoid get questions from the deeper clusters
                    pre_clusers=[]
                    for pre_cluser_dis in all_clusers_distances[i+1]:
                        pre_clusers.append(pre_cluser_dis[0])

                    quesion_everycluster=[]
                    for single_question in poten_quesion_everycluster:
                        
                        append_flag=1
                        for pre_cluser in pre_clusers:
                            if set(single_question).issubset(pre_cluser):            
                                append_flag+=1
                            
                        if append_flag==1:
                            quesion_everycluster.append(single_question)

                    quesion_everycluster=[v for v in quesion_everycluster if v not in potential_quesions]
            
            
            quesion_everylayer.extend(quesion_everycluster)
            #calulate the mean and var  and answerd_neighbour_qs number
            quesion_everycluster_dic,neighbour_an_question= calculate_dir_cluser (quesion_everycluster,answer_questions,names,embeddings,distance,a_all_clusers,i,depth,dis_weight,use_prior,use_weight)
            quesion_everylayer_dic.extend(quesion_everycluster_dic)
            
            layer_neighbour_an_question+=neighbour_an_question
            
            print ("layer",(i),'in cluser potential questions got',len(quesion_everycluster),' answered neigbhour question',str(neighbour_an_question))
    
    
        #append the data
        total_answerd_neighbour_q+=layer_neighbour_an_question
        potential_quesions.extend(quesion_everylayer)
        potential_quesions_dic.extend(quesion_everylayer_dic)
        

    random.shuffle(potential_quesions_dic)

    print ('total_potential_questions',str(len(potential_quesions)))
    print ('total_answerd_neighbour_questions',str(total_answerd_neighbour_q),'\n')
    return potential_quesions_dic


def get_potential_no_active(pick_clusers_distances,cluster_depth,depth,question_num,active_S_num):
    potential_quesions=[]
    potential_diss=[]

    

    
    #getting questions by layers

    if cluster_depth>len(pick_clusers_distances):
        cluster_depth=len(pick_clusers_distances)
    for i in reversed(range(cluster_depth)):
        
        distance_everylayer=[]
        quesion_everylayer=[]

        
        for p_layer_cluser_dis in pick_clusers_distances[i]:
            
            p_clusers,distance=p_layer_cluser_dis
            if len(p_clusers)>2:
                poten_quesion_everycluster=get_random_questions(p_clusers,question_num)
            
                # avoid get questions from the deeper clusters
                if len(potential_quesions)==0:
                    quesion_everycluster=poten_quesion_everycluster
                else:  
                    pre_clusers=[]
                    for pre_cluser_dis in pick_clusers_distances[i+1]:
                        pre_clusers.append(pre_cluser_dis[0])

                    quesion_everycluster=[]
                    for single_question in poten_quesion_everycluster:
                        append_flag=1
                        for pre_cluser in pre_clusers:
                            if set(single_question).issubset(pre_cluser):            
                                append_flag+=1
                            
                        if append_flag==1:
                            quesion_everycluster.append(single_question)

                    quesion_everycluster=[v for v in quesion_everycluster if v not in potential_quesions]
             
            # get distance every cluster
            distance_everycluster=[distance]*len(quesion_everycluster)
            
            #append the one layer data
            quesion_everylayer.extend(quesion_everycluster)
            distance_everylayer.extend(distance_everycluster)

        
            
        print ("layer",(i),'potential questions got',len(quesion_everylayer))
    
    
        #append the all layer data
        potential_quesions.extend(quesion_everylayer)
        potential_diss.extend(distance_everylayer)

    mix = list(zip(potential_quesions, potential_diss))
    mix = random.sample(mix,active_S_num)

    selected_questions=[]
    distances=[]

    selected_questions[:], distances[:] = zip(*mix)

    print ('random select questions',str(len(selected_questions)))
    
    return selected_questions,distances








def active_selection(potential_quesions_dic,select_num,mean_shre):
 
    # load answered questions and potential_quesions
    
    accpet_question_by_mean = []
    
    rejected_by_mean = 0
    accepted_by_mean = 0
    

    # reject question by max mean 
    for q in potential_quesions_dic:


        if q['max_mean'] <mean_shre :
            accepted_by_mean += 1
            accpet_question_by_mean.append(q)

        else:
            rejected_by_mean += 1
            

        
    # print the number of rejection

    print('accepted_by_mean', accepted_by_mean)
    print('rejected_by_mean', rejected_by_mean)
    
    
    
    
    if len(accpet_question_by_mean) < 1:
        return [],[]
    elif len(accpet_question_by_mean) < select_num:
        select_num=len(accpet_question_by_mean)
        


        

    # sort questions (q) based on var (var)
    sorted_questions = sorted(accpet_question_by_mean, key=lambda x: x['var'],reverse=True)  
    selected_questions_dic=sorted_questions[0:select_num]
    
    
    
    # get mean list before rejection
    mean_before_rejection=[]
    for q in sorted(potential_quesions_dic, key=lambda x: x['max_mean']):
        mean_before_rejection.append(q['max_mean'])    
        
    # get mean list after rejection           
    mean_after_rejection=[]
    for q in sorted(accpet_question_by_mean, key=lambda x: x['max_mean']):
        mean_after_rejection.append(q['max_mean'])    
    
    # get var list before sort
    var_before_sort=[]
    for q in sorted_questions:
        var_before_sort.append(q['var'])   
    # get var list after sort
    var_after_sort=[]
    for q in selected_questions_dic:
        var_after_sort.append(q['var'])   
        
    
    
    # get result
    random.shuffle(selected_questions_dic)
    
    selected_questions = []
    distances=[]


    for q in selected_questions_dic:
        # print(q)
        selected_questions.append(q['q'])    
        distances.append(q['dis'])  
    print('selected_questions', len(selected_questions),'\n')
    return selected_questions,distances,mean_before_rejection,mean_after_rejection,var_before_sort,var_after_sort









