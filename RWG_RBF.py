# from svmutil import *
# from svm import *
# y = [1, -1]
# x = [{1: 2, 2: 1}, {1: -1, 2: -1}]
# # x = [{0:1,1:4,2:6,3:2},
# #      {0:2,1:6,2:18,3:0},
# #      {0:3,1:1,2:0,3:1}]
#
#
# # prob = svm_problem(y, x)
# # print(prob)
# # param = svm_parameter('-t 0 -c 4 -b 1')
# # model = svm_train(prob, '-t 4')
# model = svm_train(y,x, '-t 1')
# yt = [1]
# xt = [{1: -1, 2: -1}]
# # xt = [{0:4,1:2,2:0,3:1}]
#
# p_label, p_acc, p_val = svm_predict(yt, xt, model)
# print(p_label,p_acc, p_val)


# from svmutil import *
# from svm import *
# y,x = svm_read_problem('D:\libsvm-3.18\heart_scale')
# print(y)
# print(x)
# m = svm_train(y[:200], x[:200], '-c 4')
# print ('----------------')
# lable, acc, val = svm_predict(y[200:], x[200:], m)

#
# from svmutil import *
# from svm import *
#
# def kernel(x_all):
#     for x in x_all:
#
#
#
# y,x = svm_read_problem('D:\libsvm-3.18\heart_scale')


# train_data = x[:150]
# train_label = y[:150]
# test_data = x[150:]
# test_label = y[150:]

#线性核函数
# model_linear = svm_train(train_label, train_data, '-t 0')
# predict_label_L, accuracy_L, dec_values_L = svm_predict(test_label, test_data, model_linear)
# print(test_label)
# print(predict_label_L)
# print(accuracy_L)


#使用的核函数 K(x,x') = (x * x')
#核矩阵

# model_precomputed1 = svm_train(train_label, train_data, '-t 4')
# predict_label_P1, accuracy_P1, dec_values_P1 = svm_predict(test_label, test_data, model_precomputed1)
# print(test_label)
# print(predict_label_P1)
# print(accuracy_P1)

from svmutil import *
from svm import *
import pandas as pd
import json
from treelib import Node,Tree
import random
import numpy as np
from sklearn.model_selection import ShuffleSplit
import jieba
import re
import jieba.posseg as pseg
import codecs
from sklearn import preprocessing
class Info(object):
    def __init__(self,data):
        if data!= None:
            self.reposts_count = data.get('reposts_count')
            self.uid = data.get('uid')
            self.bi_followers_count = data.get('bi_followers_count')
            self.text = data.get('text')
            self.original_text = data.get('original_text')
            self.user_description = data.get('user_description')
            self.friends_count = data.get('friends_count')
            self.mid = data.get('mid')
            self.attitudes_count = data.get('attitudes_count')
            self.followers_count = data.get('followers_count')
            self.statuses_count = data.get('statuses_count')
            self.verified = data.get('verified')
            self.user_created_at = data.get('user_created_at')
            self.favourites_count = data.get('favourites_count')
            self.gender = data.get('gender')
            self.comments_count = data.get('comments_count')
            self.t = data.get('t')
            self.approval_score = random.random()
            self.doubt_score = random.random()

            if int(data.get('friends_count')) > 0 and int(data.get('followers_count'))/int(data.get('friends_count')) >= 2:
                self.type = 'o'
            else:
                self.type = 'n'
        else:
            self.approval_score = random.random()
            self.doubt_score = random.random()
            self.type = 'n'

def sim(node1,node2):
    vec1 = [node1.data.approval_score,node1.data.doubt_score]
    vec2 = [node2.data.approval_score, node2.data.doubt_score]
    vector_a = np.mat(vec1)
    vector_b = np.mat(vec2)
    num = float(vector_a * vector_b.T)
    denom = np.linalg.norm(vector_a) * np.linalg.norm(vector_b)
    cos = num / denom
    sim = 0.5 + 0.5 * cos
    return sim

def rwg_kernel(matrix,param_a):
    """

    :param matrix: 根据两颗传播树求得的直积图
    :param param_a: 指定参数
    :return: 随机游走图核函数值
    """
    e = np.zeros(matrix.shape[0])
    e[0] = 1
    E = np.matrix(e) #单位向量，行向量
    I = np.eye(matrix.shape[0]) #单位矩阵

    kernel = float(E * np.linalg.inv((I - param_a * matrix)) * E.T) #要转成float形式才是数值，否则是matrix
    # print(kernel)
    return kernel

def rbf_kernel(XTrain1,XTrain2,sigma):
    """

    :param XTrain1: dict形式的feature或者ndarray形式的feature_val
    :param XTrain2: dict形式的feature或者ndarray形式的feature_val
    :param sigma: 高斯核的带宽
    :return: 高斯核值
    """
    if type(XTrain1) is dict and type(XTrain2) is dict:
        feature_i = [XTrain1[key] for key in XTrain1]
        feature_j = [XTrain1[key] for key in XTrain2]
        vector_a = np.mat(feature_i)
        vector_b = np.mat(feature_j)
        rbf_kernel = np.e**(-(np.power(np.linalg.norm(vector_a-vector_b),2))/(2*np.power(sigma,2)))
        return rbf_kernel
    elif type(XTrain1) is np.ndarray and type(XTrain2) is np.ndarray:
        rbf_kernel = np.e ** (-(np.power(np.linalg.norm(XTrain1 - XTrain2), 2)) / (2 * np.power(sigma, 2)))
        return rbf_kernel

def kernel_preProcess(Train_tree,XTrain,Test_tree=None,XTest=None,isTraining=True,param_a=0.5,sigma=0.5):
    x_return = []
    if isTraining == True:
        for i in range(len(Train_tree)):
            x_dict = {}
            x_dict[0] = i + 1
            for j in range(len(Train_tree)):
                product_matrix = cal_product_graph(Train_tree[i], Train_tree[j])
                rwg_kernel_val = rwg_kernel(product_matrix, param_a=param_a)
                rbf_kernel_val = rbf_kernel(XTrain[i], XTrain[j],sigma=sigma)
                x_dict[j + 1] = rwg_kernel_val + rbf_kernel_val
            x_return.append(x_dict)
            # print(x_dict)
    else:
        for i in range(len(Test_tree)):
            x_dict = {}
            x_dict[0] = len(Train_tree) + i + 1
            for j in range(len(Train_tree)):
                product_matrix = cal_product_graph(Test_tree[i], Train_tree[j])
                rwg_kernel_val = rwg_kernel(product_matrix, param_a=param_a)
                rbf_kernel_val = rbf_kernel(XTest[i], XTrain[j],sigma=sigma)
                x_dict[j + 1] = rwg_kernel_val + rbf_kernel_val
            x_return.append(x_dict)
    return x_return

def cal_product_graph(tree1,tree2):
    tree1_nodes = tree1.all_nodes()
    tree2_nodes = tree2.all_nodes()
    new_nodes = []
    for node1 in tree1_nodes:
        for node2 in tree2_nodes:
            if node1.data.type == node2.data.type:
                new_nodes.append((node1.identifier,node2.identifier))

    ma = np.zeros((len(new_nodes),len(new_nodes)))

    for i in range(len(new_nodes)):
        for j in range(len(new_nodes)):
            if i!=j:
                new_node1 = new_nodes[i]
                new_node2 = new_nodes[j]
                node11 = new_node1[0]
                node12 = new_node1[1]
                node21 = new_node2[0]
                node22 = new_node2[1]

                parent_1 = tree1.parent(node21)
                parent_2 = tree2.parent(node22)

                if tree1.get_node(node11) == parent_1 and tree2.get_node(node12) == parent_2:
                    ma[i,j] = sim(tree1.get_node(node21),tree2.get_node(node22))
    # print("product_matrix:",ma)
    return ma

def init():
    jieba.add_word('微博',tag='n')
    jieba.add_word('笑而不语')
def illegal_char(s):
    s = re.compile(
        u"[^"
        u"\u4e00-\u9fa5" #中文
        # u"\u0041-\u005A" #英文
        # u"\u0061-\u007A" #英文
        # u"\u0030-\u0039" #数字
        #中文标点
        # u"\u3002\uFF1F\uFF01\uFF0C\u3001\uFF1B\uFF1A\u300C\u300D\u300E\u300F\u2018\u2019\u201C\u201D\uFF08\uFF09\u3014\u3015\u3010\u3011\u2014\u2026\u2013\uFF0E\u300A\u300B\u3008\u3009"
        #英文标点
        # u"\!\@\#\$\%\^\&\*\(\)\-\=\[\]\{\}\\\|\;\'\:\"\,\.\/\<\>\?\/\*\+"
        u"]+")\
        .sub('', s)
    return s

def participate(text):
    # seg_list = jieba.cut(text, cut_all=False)

    seg_list = pseg.cut(text)
    return seg_list
def get_stopwords():
    path = 'chinese_stopwords.txt'
    # stoplist = {}.fromkeys([line.strip for line in codecs.open(path,'r','utf-8')])
    stoplist = []
    file = open(path,'r',encoding='utf-8').read()
    for line in file:
        stoplist.append(line)
    return stoplist
def get_pos_num(seg_list,type):
    count = 0
    for w in seg_list:
        if(w.flag==type):
            count+=1
    return count
def extract_features(info):
    features = {'rep_count':0,
                'comments_count':0,

                'stopword_count':0,
                'text_length':0,
                'text_NN_rat':0,
                'text_verb_rat':0,
                'text_adj_rat':0,
                '@_count':0,
                '?_count':0,
                '!_count':0,
                'has_hashtag':0,
                'has_url':0,

                'bi_followers_count':0,
                'friends_count':0,
                'is_verified':0,
                'followers_count':0,
                'statuses_count':0,
                'is_male':0,
                'favourites_count':0}
    #社交网络特征
    features['rep_count'] = info.reposts_count
    features['comments_count'] = info.comments_count

    #文本特征
    text = info.text
    seg_text1 = participate(illegal_char(text))
    seg_text2 = []
    stoplist = get_stopwords()
    seg_text3 = []
    for w in seg_text1:
        if (w.word not in stoplist):
            seg_text3.append(w)
            seg_text2.append(w.word)
        else:
            if 'stopword_count' not in features:
                features['stopword_count'] = 1
            else:
                features['stopword_count'] += 1
    features['text_length'] = len(text)
    features['text_NN_rat'] = get_pos_num(seg_text3, 'n') / (len(seg_text3) + 1)
    features['text_verb_rat'] = get_pos_num(seg_text3,'v') / (len(seg_text3) + 1)
    features['text_adj_rat'] = get_pos_num(seg_text3,'a') / (len(seg_text3) + 1)
    features['@_count'] = str(info.original_text).count('@')
    features['?_count'] = str(info.text).count('?')+ str(info.text).count('？') # 英文问号+中文问号
    features['!_count'] = str(info.text).count('!')+str(info.text).count('！')  # 英文感叹号+中文感叹号

    if info.text.startswith("【") or info.text.startswith("#") or info.text.startswith("["):
        features['has_hashtag'] = 1
    else:
        features['has_hashtag'] = 0
    if info.text.__contains__("http"):
        features['has_url'] = 1
    else:
        features['has_url'] = 0
    #用户特征
    features['bi_followers_count'] = info.bi_followers_count
    features['friends_count'] = info.friends_count
    if info.verified == True:
        features['is_verified'] = 1
    else:
        features['is_verified'] = 0
    features['followers_count'] = info.followers_count
    features['statuses_count'] = info.statuses_count
    if info.gender == 'm':
        features['is_male'] = 1
    else:
        features['is_male'] = 0
    features['favourites_count'] = info.favourites_count

    return features

def zscore_features(features):
    """

    :param features: features是dict形式的列表 [{},{}]
    :return:
    """

    features_list = [feature[key] for feature in features for key in feature]
    features_arr = np.array(features_list)
    features_arr = features_arr.reshape(len(features),-1)
    zscore_features = preprocessing.scale(features_arr)
    return zscore_features

def write_infos_to_file(path,infos,eid,label):
    with codecs.open(path, 'a+', encoding='utf-8') as info_file:
        info_file.write(eid+'\t')
        for key in infos:#key2-->[count,rep_count,comments_count...]
            if type(infos[key]) is list:
                for item in infos[key]:
                    info_file.write(str(item) + '\t')
            else:
                info_file.write(str(infos[key])+'\t')
        info_file.write(label+'\n')


def level_simplify_tree(tree,root,modified = False):
    if root == None:
        return
    # 找出标记为'n'的节点
    normal_nodes_iden = []
    for node in tree.children(root):
        if node.data.type == 'n':
            normal_nodes_iden.append(node.identifier)

    # 如果有多个普通节点，计算标记为'n'的节点的向量平均值，更新第一个标记为'n'的节点
    if len(normal_nodes_iden) > 1:
        modified = True
        info = Info(None)
        info.type = 'n'
        for node_iden in normal_nodes_iden:
            info.approval_score += tree.get_node(node_iden).data.approval_score
            info.doubt_score += tree.get_node(node_iden).data.doubt_score
        info.approval_score /= len(normal_nodes_iden)
        info.doubt_score /= len(normal_nodes_iden)

        tree.update_node(nid=normal_nodes_iden[0],data=info)

        for i in range(1, len(normal_nodes_iden)):
            # 将子节点都移动为第一个标记为'n'的节点的子节点
            for child in tree.children(normal_nodes_iden[i]):
                tree.move_node(child.identifier, normal_nodes_iden[0])
            # 移除无用节点
            tree.remove_node(normal_nodes_iden[i])

    # 对下一层进行合并
    for node in tree.children(root):
        level_simplify_tree(tree, node.identifier)
    # tree.show()
    return modified

def parent_child_simp_tree(tree,root):
    if root == None:
        return
    for node in tree.children(root):
        parent_child_simp_tree(tree,node.identifier)
    parent = tree.parent(root)



    if parent != tree.get_node(tree.root) and parent != None:
        if parent.data.type == tree.get_node(root).data.type == 'n':
            info = Info()
            info.type = 'n'
            info.approval_score = (parent.data.approval_score + tree.get_node(root).data.approval_score) / 2
            info.doubt_score = (parent.data.doubt_score + tree.get_node(root).data.doubt_score) / 2
            tree.update_node(nid=parent.identifier, data=info)
            for node in tree.children(root):
                tree.move_node(node.identifier, parent.identifier)
            tree.remove_node(root)

def simplify_tree(tree):

    modified = True
    while(modified == True):
        modified = level_simplify_tree(tree,tree.root)
        parent_child_simp_tree(tree,tree.root)
    return tree




if __name__ == "__main__":
    #对所有数据构建传播树，生成的传播树全放到trees列表
    data = pd.read_csv('D:/chenjiao/SinaWeibo/datasets2/Weibo.txt', sep='\t', header=None)
    data_array = data.as_matrix()
    trees = []
    features = []
    features_zscore = []
    need_write_headers = True

    for i in range(10):
        eid = str(data_array[i][0]).replace('eid:', '')
        label = str(data_array[i][1].replace('label:', ''))
        load_f = open('D:/chenjiao/SinaWeibo/datasets2/Weibo/{}.json'.format(eid), 'r', encoding='utf-8')
        json_data = json.load(load_f)


        #构建新闻事件传播树
        tree = Tree()
        tree.create_node(tag=json_data[0].get("mid"),identifier=json_data[0].get("mid"),data=Info(json_data[0]))
        uu = 1000 if len(json_data) >=1000 else len(json_data)
        print('uu:',uu)
        for j in range(1,uu):
            try:
                tree.create_node(tag=json_data[j].get("mid"),identifier=json_data[j].get("mid"),parent=json_data[j].get("parent"),data=Info(json_data[j]))
            except:
                pass
        #单颗传播树构建完成
        #对传播树进行简化
        tree = simplify_tree(tree)
        # tree.show()
        trees.append(tree)


        #提取新闻原文的相关特征,并写入文件
        feature = extract_features(Info(json_data[0]))
        feature['tree_depth'] = tree.depth()
        features.append(feature)


        # feature_path = './Features/features1.txt'
        # if need_write_headers:
        #     need_write_headers = False
        #
        #     with codecs.open(feature_path, 'a+', encoding='utf-8') as info_file:
        #         info_file.write('eid'+'\t')
        #         for key in feature: #key-->[rep_count,comments_count...]
        #             if type(feature[key]) is list:
        #                 for index, item in enumerate(feature[key]):
        #                     info_file.write(str(key) + '_{}'.format(str(index)) + '\t')
        #             else:
        #                 info_file.write(str(key) + '\t')
        #         info_file.write('label'+'\n')
        # write_infos_to_file(feature_path,feature,eid,label=label)
        #提取特征结束，并成功写入文件

    #所有新闻事件的传播树构建结束

    #对求得的features做标准化

    features_list = [feature[key] for feature in features for key in feature]
    features_arr = np.array(features_list).reshape(len(features), -1)
    features_zscore = preprocessing.scale(features_arr)

    #五折交叉法
    pd_data = pd.read_csv('id_label.txt', sep='\t', header=None)
    wb_data = pd_data.as_matrix()[0:10,]
    kf = ShuffleSplit(n_splits=5, random_state=0)
    i = 0
    for train, test in kf.split(wb_data):
        train_trees = []
        test_trees = []
        # train_features = []
        # test_features = []

        train_features = features_zscore[train]
        test_features = features_zscore[test]

        train_ids = wb_data[train][:,0].tolist()
        train_labels = wb_data[train][:,1].tolist()

        test_ids = wb_data[test][:,0].tolist()
        test_labels = wb_data[test][:, 1].tolist()

        for train_index in train.tolist():
            train_trees.append(trees[train_index])
            # train_features.append(features[train_index])
        for test_index in test.tolist():
            test_trees.append(trees[test_index])
            # test_features.append(features[test_index])
        #对传播树核化
        # train_data = rwg_kernel_preProcess(train_trees,isTraining=True)
        # test_data = rwg_kernel_preProcess(train_trees,test_trees,isTraining=False)
        # print('train_data:',train_data)
        # print('test_data:',test_data)

        train_data = kernel_preProcess(Train_tree=train_trees,XTrain=train_features,isTraining=True)
        test_data = kernel_preProcess(Train_tree=train_trees,XTrain=train_features,Test_tree=test_trees,XTest=test_features,isTraining=False)
        print('train_data:',train_data)
        print('test_data:',test_data)

        #根据自定义的核函数训练模型
        model_precomputed1 = svm_train(train_labels, train_data, '-t 4')
        #预测
        predict_label_P1, accuracy_P1, dec_values_P1 = svm_predict(test_labels, test_data, model_precomputed1)
        print('test_labels:',test_labels)
        print('predict_label_P1:',predict_label_P1)
        print('accuracy_P1:',accuracy_P1)
        print("=====================")
        print()
        print()
