import tensorflow as tf

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string('data_dir', 'MNIST_data', 'dataset path!')
flags.DEFINE_string('model_path', 'model', 'Model save path!')
flags.DEFINE_boolean("train", True, 'whether train the model')

#config for the model
flags.DEFINE_integer('input_size', 28, 'input_size')
flags.DEFINE_integer('timestep_size', 28, 'timestep size')
flags.DEFINE_integer('hidden_size', 256, 'hidden units size')
flags.DEFINE_integer('layer_num', 2, 'layer num')
flags.DEFINE_integer('class_num', 10, 'class num')

flags.DEFINE_integer('train_keep_prob', 0.5, 'dropout rate in train')
flags.DEFINE_float('test_keep_prob', 1, 'dropout rate in test')
flags.DEFINE_float('train_batch_size', 50, 'batch_size in train')
flags.DEFINE_float('test_batch_size', 50, 'batch_size in test')

# super parameters
flags.DEFINE_float('lr', 1e-3, 'learning rate')
flags.DEFINE_integer('training_steps', 5000, 'training steps')














# tf.app.flags.DEFINE_boolean("train",True,"whether train the model")
# tf.app.flags.DEFINE_string("dataset",DATASET,"dataset")

# configurations for the model
# tf.app.flags.DEFINE_string("model_type","RE","NER RE NER_RE three type")
# tf.app.flags.DEFINE_string("NER_model","decoder","NER model type:hidden,multihead_att,self_att,decoder")
# tf.app.flags.DEFINE_integer("NER_h",1,"NER multihead parameter")
# tf.app.flags.DEFINE_string("RE_model","att","RE model type:att,r_att")
# tf.app.flags.DEFINE_integer("r",50,"r_att parameter")
# tf.app.flags.DEFINE_integer("d_a",100,"r_att parameter")
# tf.app.flags.DEFINE_string("tag_schema",   "iob",    "tagging schema iobes or iob")
# tf.app.flags.DEFINE_boolean("use_radical",False,"whether use the radical feature")
# tf.app.flags.DEFINE_integer("stroke_num",6,"stroke num")
# tf.app.flags.DEFINE_integer("stroke_emb_dim",0,"stroke embedding dimension")
# # tf.app.flags.DEFINE_integer("radical_num",6,"radical num")
# # tf.app.flags.DEFINE_integer("radical_emb_dim",10,"radical embedding dimension")
# tf.app.flags.DEFINE_integer("lstm_dim",100,"number of hidden units in LSTM")
# tf.app.flags.DEFINE_integer("max_sentence_length",100,"max sentence length")
# tf.app.flags.DEFINE_integer("max_pos_embed",80,"max relative position embed range:[-max_pos_embed,max_pos_embed]")
# tf.app.flags.DEFINE_integer("pos_emb_dim",5,"pos embedding dim")
# tf.app.flags.DEFINE_integer("entitytype_emb_dim",5,"entitytype embedding dim")
# tf.app.flags.DEFINE_integer("use_pos",1,"whether use rel_pos as feature")
# tf.app.flags.DEFINE_integer("use_entitytype",1,"whether use entity_type as feature")
# tf.app.flags.DEFINE_boolean("pos_emb",True,"whether embedding the position")
# tf.app.flags.DEFINE_boolean("entity_type_emb",True,"whether embedding the entity_type")
# tf.app.flags.DEFINE_boolean("lstm_use_peepholes",True,"lstm cell whether use peepholes")
#
# # configurations for training and testing
# tf.app.flags.DEFINE_float("clip",5,"Gradient clip")
# tf.app.flags.DEFINE_float("dropout",0.5,"Dropout rate")
# tf.app.flags.DEFINE_float("lr",0.001,"learning rate")
# tf.app.flags.DEFINE_float("l2_reg",0.001,"l2 regularization parameter")
# tf.app.flags.DEFINE_string("optimizer","adam","Optimizer for training")
# # tf.app.flags.DEFINE_boolean("pre_emb",True,"Wither use pre-trained embedding")
# tf.app.flags.DEFINE_integer("train_batch_size",20,"train data batch size")
# tf.app.flags.DEFINE_integer("dev_batch_size",64,"dev data batch size")
# tf.app.flags.DEFINE_integer("test_batch_size",64,"test data batch size")
# tf.app.flags.DEFINE_integer("train_epoch",100,"train epoch size")
# # tf.app.flags.DEFINE_string("NER_train_file",os.path.join("data", DATASET,"ner", "NER_train0.txt"),"NER training data path")
# tf.app.flags.DEFINE_string("NER_train_file",os.path.join("data", DATASET, "NER_train.txt"),"NER training data path")
# tf.app.flags.DEFINE_string("NER_dev_file",os.path.join("data", DATASET, "NER_validation.txt"),"NER validation data path")
# tf.app.flags.DEFINE_string("NER_test_file",os.path.join("data", DATASET, "NER_test.txt"),"NER test data path")
# tf.app.flags.DEFINE_string("RE_train_file",os.path.join("data", DATASET,"RE_train.txt"),"RE training data path")
# tf.app.flags.DEFINE_string("RE_dev_file",os.path.join("data", DATASET,"RE_validation.txt"),"RE validation data path")
# tf.app.flags.DEFINE_string("RE_test_file",os.path.join("data", DATASET,"RE_test.txt"),"RE test data path")
# tf.app.flags.DEFINE_boolean("create_maps",True,"whether create maps.pkl")
# # tf.app.flags.DEFINE_integer("pretrained_char_dim",100,"pretrained embedding size for characters")
# tf.app.flags.DEFINE_integer("steps_check",100,"steps per checkpoint")
# tf.app.flags.DEFINE_string("pretrained_emb_file", os.path.join("data","char_emb.txt"),"pretrained char embedding file path")
# tf.app.flags.DEFINE_string("not_c", os.path.join("data","not_c.txt"),"not_char_embedding_found_file")
# tf.app.flags.DEFINE_string("radical_feature_file", os.path.join("data","zi.txt"),"character feature file path")
# tf.app.flags.DEFINE_string("not_r", os.path.join("data","not_r.txt"),"not_radical_feature_found_file")
# tf.app.flags.DEFINE_integer("char_emb_dim",100,"pretrained char embedding dim")
# tf.app.flags.DEFINE_string("maps_file", "maps.pkl","char embedding and relation2id file path")
# tf.app.flags.DEFINE_string("NER_graph_save_folder", os.path.join("graph","NER"),"NER graph save file path")
# tf.app.flags.DEFINE_string("RE_graph_save_folder", os.path.join("graph","RE"),"RE graph save file path")
# tf.app.flags.DEFINE_string("train_log_file",  os.path.join("log",str(datetime.date.today()) + "_train.log"),"file for training log")
# tf.app.flags.DEFINE_string("test_log_file",  os.path.join("log",str(datetime.date.today()) + "_test.log"),"file for training log")
# tf.app.flags.DEFINE_string("summary_log_file",  os.path.join("log",str(datetime.date.today()) + "_summary.log"),"file for summary log")
# tf.app.flags.DEFINE_string("NER_save_folder",  "NER_saved_model","folder for saving model in NER module")
# tf.app.flags.DEFINE_string("RE_save_folder",  "RE_saved_model","folder for saving model in RE module")
# tf.app.flags.DEFINE_string("NER_result_folder", "NER_result",       "Path for NER module results")
# Flags = tf.app.flags.FLAGS
