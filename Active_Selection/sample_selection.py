from utils import *
from collections import defaultdict
import pymysql
import matplotlib.pyplot as plt
sys.path.append("..") 
from config import conf


class_names =['airplane' ,'automobile' ,'bird','cat' ,'deer' ,'dog' ,'frog' ,'horse','ship' ,'truck' ]

standard1=[['airplane','automobile','ship' ,'truck' ],'machine']
standard2=[['bird','cat','deer' ,'dog','frog','horse' ],'animail']
standard_a=[standard1,standard2]

standard1=[['truck','automobile' ],'road']
standard2=[['ship' ,'airplane'],'not_road']
standard_b=[standard1,standard2]

standard1=[['dog','cat','bird' ,'frog' ],'small_animal']
standard2=[['deer' ,'horse' ] ,'big_animal']
standard_c=[standard1,standard2]


Annotator_1_standard=[standard_c,standard_b,standard_a]



standard1=[['airplane','automobile','ship' ,'truck' ],'machine']
standard2=[['bird','cat','deer' ,'dog','frog','horse' ],'animail']
standard_a=[standard1,standard2]

standard1=[['airplane','automobile' ],'for_human']
standard2=[['ship' ,'truck'],'not_road']
standard_b=[standard1,standard2]


standard1=[['dog','cat','deer' ,'horse' ],'mammal']
standard2=[['frog' ,'bird' ] ,'un_mammal']
standard_c=[standard1,standard2]

Annotator_2_standard=[standard_c,standard_b,standard_a]



standard1=[['airplane','automobile','ship' ,'truck' ],'machine']
standard2=[['bird','cat','deer' ,'dog','frog','horse' ],'animail']
standard_a=[standard1,standard2]


standard1=[['frog','bird','deer' ,'horse' ],'outdoor_animal']
standard2=[['dog' ,'cat' ] ,'indoor_animal']
standard_b=[standard1,standard2]



Annotator_3_standard=[standard_b,standard_a]

standard={"Annotator_1":Annotator_1_standard, "Annotator_2":Annotator_2_standard, "Annotator_3":Annotator_3_standard, }
annotators=["Annotator_1","Annotator_2","Annotator_3"]

#Loading _data

print ('\n','In interation',str(load_interation()),'generateing data','\n')
print ('s_score_shrehold is',str(conf.s_score_shrehold),'\n')

'''traing_data'''

train=True

print ('get trainng data')

for annotator in annotators:
    
    print  ('\n',str(annotator),'starts','\n')
    


 

    names = load_names(train)
    labels=load_labels(train)
    
    
    #Initalize in the firest interation
    
    if load_interation() == 0:
        ini_num=conf.ini_num
        T= get_random_questions(names,ini_num)
        ini_distance(train,len(T),annotator)

        triplets=simulate_answer_questions(T,names,class_names,labels,standard,annotator)
        save_first_interation_answers(train,triplets,annotator)
    else:
        #Loading _trained_result   
        embeddings = load_embeddings(train,annotator)
        answer_questions = load_answered_triplets(train,annotator)
        print(str(annotator),'answerd questions: ', len(answer_questions),'\n')


        #Get K means Hierarchical
        cluster_depth=conf.cluster_depth
        depth=conf.depth
        max_cluser_layer=conf.max_cluser_layer
        s_score_shrehold=conf.s_score_shrehold

        pick_clusers_distances,all_clusers_distances=get_all_clusers_distances(names,embeddings,max_cluser_layer,cluster_depth+depth,s_score_shrehold)
        
        
        #Generate questions
        question_num=conf.question_num
        use_prior=conf.use_prior
        use_weight=conf.use_weight
        active_S_num= conf.active_S_num

        if conf.Active_learning==True:

            potential_quesions_dic=get_potential_quesions(answer_questions,names,embeddings,pick_clusers_distances,all_clusers_distances,cluster_depth,depth,question_num,use_prior,use_weight)

            #Active selecting questions
            mean_shre=conf.mean_shre
            selection_rate=conf.selection_rate
            
            selected_questions,distances,mean_before_rejection,mean_after_rejection,var_before_sort,var_after_sort=active_selection(potential_quesions_dic,active_S_num,mean_shre)


            #Plot and save the selcetion result

            fig, axes = plt.subplots(nrows=2, ncols=1)

            axes[0].plot(range(len(mean_before_rejection)),mean_before_rejection,label='before_rejection',linestyle=':')
            axes[0].plot(range(len(mean_after_rejection)),mean_after_rejection,label='after_rejection')
            axes[0].legend(fontsize=8)
            axes[0].set(xlim=[0,len(mean_before_rejection)], ylim=[0.2,1])

            axes[0].set_title("Rejcet by Expectation Value", fontsize=10)
            axes[0].set_ylabel('Max expectation value', fontsize=8)
            axes[0].set_xlabel('Number of samples', fontsize=8)


            axes[1].plot(range(len(var_before_sort)),var_before_sort,label='before_choosing',linestyle=':')
            axes[1].plot(range(len(var_after_sort)),var_after_sort,label='after_choosing')
            axes[1].legend(fontsize=8)
            axes[1].set(xlim=[0,len(mean_before_rejection)], ylim=[0,0.2])

            axes[1].set_title("Choose by Variance", fontsize=10)
            axes[1].set_ylabel('Sum of variances', fontsize=8)
            axes[1].set_xlabel('Number of samples', fontsize=8)


            fig.tight_layout() 
            plt.savefig(str(annotator)+'_Training In '+str(load_interation())+' interation '+'selection.png')

            # plt.show()

        else:
            selected_questions,distances=get_potential_no_active(pick_clusers_distances,cluster_depth,depth,question_num,active_S_num)

        
        #Simulaton the answering
        triplets=simulate_answer_questions(selected_questions,names,class_names,labels,standard,annotator)
        
        #Save the results
        update_distance(train,names,embeddings,pick_clusers_distances,cluster_depth,annotator)
        save_answers_distances(train,triplets,distances,annotator)


# get general training data

print ('\n','Get general training data')

cluster_depth=conf.cluster_depth
depth=conf.depth
max_cluser_layer=conf.max_cluser_layer
s_score_shrehold=conf.s_score_shrehold

names = load_names(train)

if load_interation() == 0:
    mode='clear'
    embeddings=0
else:
    mode='update'
    embeddings = load_embeddings(train,"general")
    
Concatenate(train,annotators,mode,names,embeddings,cluster_depth,depth,max_cluser_layer,s_score_shrehold)




'''testing_data'''

train=False

print ('\n','get testiing data')

for annotator in annotators:
    
    print  ('\n',str(annotator),'starts','\n')
    


    

    names = load_names(train)
    labels=load_labels(train)
    
    
    #Initalize in the firest interation
    
    if load_interation() == 0:
        ini_num=int(conf.ini_num/10)
        T= get_random_questions(names,ini_num)
        ini_distance(train,len(T),annotator)

        triplets=simulate_answer_questions(T,names,class_names,labels,standard,annotator)
        save_first_interation_answers(train,triplets,annotator)
    else:
        #Loading _trained_result   
        embeddings = load_embeddings(train,annotator)
        answer_questions = load_answered_triplets(train,annotator)
        print(str(annotator),'answerd questions: ', len(answer_questions),'\n')


        #Get K means Hierarchical
        cluster_depth=conf.cluster_depth
        depth=conf.depth
        max_cluser_layer=conf.max_cluser_layer
        s_score_shrehold=conf.s_score_shrehold

        pick_clusers_distances,all_clusers_distances=get_all_clusers_distances(names,embeddings,max_cluser_layer,cluster_depth+depth,s_score_shrehold)
        
        
        #Generate questions
        question_num=int(conf.question_num/10)
        use_prior=conf.use_prior
        use_weight=conf.use_weight

        potential_quesions_dic=get_potential_quesions(answer_questions,names,embeddings,pick_clusers_distances,all_clusers_distances,cluster_depth,depth,question_num,use_prior,use_weight)

        #Active selecting questions
        mean_shre=conf.mean_shre
        selection_rate=conf.selection_rate
        active_S_num= int(conf.active_S_num/10)
        selected_questions,distances,mean_before_rejection,mean_after_rejection,var_before_sort,var_after_sort=active_selection(potential_quesions_dic,active_S_num,mean_shre)


        #Plot and save the selcetion result

        fig, axes = plt.subplots(nrows=2, ncols=1)

        axes[0].plot(range(len(mean_before_rejection)),mean_before_rejection,label='before_rejection',linestyle=':')
        axes[0].plot(range(len(mean_after_rejection)),mean_after_rejection,label='after_rejection')
        axes[0].legend(fontsize=8)
        axes[0].set(xlim=[0,len(mean_before_rejection)], ylim=[0.2,1])

        axes[0].set_title("Rejcet by Expectation Value", fontsize=10)
        axes[0].set_ylabel('Max expectation value', fontsize=8)
        axes[0].set_xlabel('Number of samples', fontsize=8)


        axes[1].plot(range(len(var_before_sort)),var_before_sort,label='before_choosing',linestyle=':')
        axes[1].plot(range(len(var_after_sort)),var_after_sort,label='after_choosing')
        axes[1].legend(fontsize=8)
        axes[1].set(xlim=[0,len(mean_before_rejection)], ylim=[0,0.2])

        axes[1].set_title("Choose by Variance", fontsize=10)
        axes[1].set_ylabel('Sum of variances', fontsize=8)
        axes[1].set_xlabel('Number of samples', fontsize=8)


        fig.tight_layout() 
        plt.savefig(str(annotator)+'_Testing In '+str(load_interation())+' interation '+'selection.png')

        # plt.show()


        
        #Simulaton the answering
        triplets=simulate_answer_questions(selected_questions,names,class_names,labels,standard,annotator)
        
        #Save the results
        update_distance(train,names,embeddings,pick_clusers_distances,cluster_depth,annotator)
        save_answers_distances(train,triplets,distances,annotator)

# get general testing  data

print ('\n','Get general testing data')
cluster_depth=conf.cluster_depth
depth=conf.depth
max_cluser_layer=conf.max_cluser_layer
s_score_shrehold=conf.s_score_shrehold

names = load_names(train)

if load_interation() == 0:
    mode='clear'
    embeddings=0
else:
    mode='update'
    embeddings = load_embeddings(train,"general")
    
Concatenate(train,annotators,mode,names,embeddings,cluster_depth,depth,max_cluser_layer,s_score_shrehold)

