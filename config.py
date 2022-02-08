import tensorflow as tf

flags = tf.app.flags
flags.DEFINE_string('f', '', 'kernel')


# whether active learning

flags.DEFINE_boolean('Active_learning', True, 'whether active learning')

# traning _configration
flags.DEFINE_integer('interation', 20, 'interation of AT ')
flags.DEFINE_integer('epoch', 200, 'epoch')
flags.DEFINE_integer('batch_size', 8, 'batch size')
flags.DEFINE_integer('num_classes', 64, 'output dimension of the model')
flags.DEFINE_float("learning_rate", 0.00005, "Learning rate of for adam [0.0002]")
flags.DEFINE_boolean('use_data_augmentation', False, 'using data augmentation?')

# losses _configration
flags.DEFINE_float('margin', 0.4, 'margin for loss')
flags.DEFINE_float("increase_margin_rate", 2, "margin will increase by the the distance divide this rate")


# datasets _configration
flags.DEFINE_integer('img_dim_1', 32, 'image dimention')
flags.DEFINE_integer('img_dim_2', 32, 'image dimention')
flags.DEFINE_integer('img_dim_3', 3, 'image dimention')
flags.DEFINE_string('dataset', 'Cifar10', 'The name of dataset [dice, flower, ...]. See Data folder')

# initialize number
flags.DEFINE_integer('ini_num', 1000, 'The number of initialize triplets num')


#Get K means Hierarchical
flags.DEFINE_integer('cluster_depth', 3, 'The number of Hierachical depth')
flags.DEFINE_integer('depth', 4, 'The number of searching depth')
flags.DEFINE_integer('max_cluser_layer', 4, 'The number of max cluser layer when buiding Hierarchical tree')
flags.DEFINE_float('s_score_shrehold', 0.2, 'The shrehold for meaningful clusers')


#Get dirichlet mean and var and generate questions
flags.DEFINE_boolean('use_prior', False, 'when calculating dirichlet whether to get the prior distribution by distances')
flags.DEFINE_boolean('use_weight', True, 'when calculating dirichlet whether update prior distribution using a,b,c based on weight distances')
flags.DEFINE_integer('question_num', 600, 'The number of each layer we want')

# active_seletction
flags.DEFINE_float('mean_shre', 0.45, 'The shredhold of max_mean to selecting questions')
flags.DEFINE_integer('selection_rate', 3, 'The rate of potentiel questions and selection questions')
flags.DEFINE_integer('active_S_num', 600, 'The num of selection questions')








conf = tf.app.flags.FLAGS