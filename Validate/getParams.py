

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
import gc
import time
import random

class Info(object):
    def __init__(self,data,param=10):
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

            if int(data.get('friends_count')) > 0 and int(data.get('followers_count'))/int(data.get('friends_count')) >= float(param) and int(data.get('followers_count')) >=1000:
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
    num = round(float(vector_a * vector_b.T),2)
    denom = round((np.linalg.norm(vector_a) * np.linalg.norm(vector_b)),2)
    cos = round(num / denom,2)
    sim = round(0.5 + 0.5 * cos,2)
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

    kernel = round(float(E * np.linalg.inv((I - param_a * matrix)) * E.T),2) #要转成float形式才是数值，否则是matrix
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
        return round(rbf_kernel,2)

def kernel_preProcess(Train_tree,Test_tree=None,isTraining=True,param_a=0.5,sigma=0.5,myK=0):

    # x_return = []
    product_matrix = None

    if isTraining == True:
        # kernel_matrix = np.zeros((len(Train_tree), len(Train_tree)))
        print("Train kernel_preProcess")
        for i in range(len(Train_tree)):
            time_begin = int(time.time())
            print('time_begin:',time_begin)
            x_dict = {}
            x_dict[0] = i + 1



            i_nodes = len(Train_tree[i].all_nodes())
            for j in range(0,len(Train_tree)):
                j_nodes = len(Train_tree[j].all_nodes())
                print(i,j,i_nodes,j_nodes)
                print(i,j,"计算直积矩阵")
                product_matrix = cal_product_graph(Train_tree[i], Train_tree[j])
                # print(i,j,"直积矩阵计算完毕")
                print(i,j, "计算随机游走图核")
                rwg_kernel_val = rwg_kernel(product_matrix, param_a=param_a)
                # print(i,j,"计算随机游走图核完毕")

                # print(i,j,"计算径向基核完毕")
                x_dict[j + 1] = rwg_kernel_val
                # kernel_matrix[i,j] = rwg_kernel_val + rbf_kernel_val
                # if i!=j:
                #     kernel_matrix[j,i] = rwg_kernel_val + rbf_kernel_val
    
            write_xdict_toFile(x_dict,'./Kernels/xdict_Train{}'.format(myK))
            print(i,j,"成功写入文件")
            time_end = int(time.time())
            print('time_end:', time_end)
            print('耗时：',(time_begin-time_end)/60,'min')
            del (product_matrix)
            gc.collect()
            print()
            print()

            # x_return.append(x_dict)

        # for i in range(len(Train_tree)):
        #     x_dict = {}
        #     x_dict[0] = i+1
        #     for j in range(len(Train_tree)):
        #         x_dict[j+1] = kernel_matrix[i,j]
        #     x_return.append(x_dict)
        #     # print(x_dict)

    else:

        print("Test kernel_preProcess")
        for i in range(len(Test_tree)):
            x_dict = {}
            x_dict[0] = len(Train_tree) + i + 1
            for j in range(len(Train_tree)):
                # print(i,j,"计算直积矩阵")
                product_matrix = cal_product_graph(Test_tree[i], Train_tree[j])
                # print(i, j, "直积矩阵计算完毕")
                # print(i, j, "计算随机游走图核")
                rwg_kernel_val = rwg_kernel(product_matrix, param_a=param_a)
                # print(i, j, "计算随机游走图核完毕")
                # print(i, j, "计算径向基核")

                # print(i, j, "计算径向基核完毕")
                x_dict[j + 1] = rwg_kernel_val
            write_xdict_toFile(x_dict, './Kernels/xdict_Test{}'.format(myK))
            del (product_matrix)
            gc.collect()
            print('test:',i,"成功写入文件")
            print()
            print()

            # x_return.append(x_dict)
            # print(x_dict)
    # return x_return

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

def write_xdict_toFile(xdict,path):
    data = json.dumps(xdict)
    with open(path, 'a+') as file:
        file.write(data + '\n')






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
            info = Info(None)
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
        # print("need repeate simplify")
        modified = level_simplify_tree(tree,tree.root)
        parent_child_simp_tree(tree,tree.root)

    return tree

def get_result(predict,ytest):
    res = {}
    tp = getTP(predict,ytest)
    fp = getFP(predict,ytest)
    tn = getTN(predict,ytest)
    fn = getFN(predict,ytest)
    print('tn:',tn,'fp:',fp,'fn:',fn,'tp:',tp)
    accuracy = (tp+tn)/(tp+tn+fp+fn)
    precision = tp/(tp+fp)
    recall = tp/(tp+fn)
    f1_score = (2*precision*recall)/(precision+recall)

    res['acc'] = accuracy
    res['pre'] = precision
    res['recall'] = recall
    res['f1'] = f1_score
    return res


def getTP(predictions,input_y):
    count = 0
    for pre, real in zip(predictions,input_y):
        if pre == real and pre == 1:
            count += 1
    return count
def getFP(predictions,input_y):
    count = 0
    for pre, real in zip(predictions, input_y):

        if pre == 1 and real == 0:
            count += 1
    return count
def getTN(predictions,input_y):
    count = 0
    for pre, real in zip(predictions, input_y):

        if pre == real and real == 0:
            count += 1
    return count
def getFN(predictions,input_y):
    count = 0
    for pre, real in zip(predictions, input_y):

        if pre == 0 and real == 1:
            count += 1
    return count



if __name__ == "__main__":
    #构造1000条数据，其中502条虚假，498条真实
    random.seed(1)
    random_index = random.sample(range(0, 4664), 1000)

    data = pd.read_csv('D:/chenjiao/SinaWeibo/datasets2/Weibo.txt', sep='\t', header=None)
    data_array = data.as_matrix()[random_index]


    trees = []
    param_a = [5,10,20,30]

    for i in range(data_array.shape[0]):
        eid = str(data_array[i][0]).replace('eid:', '')

        label = str(data_array[i][1].replace('label:', ''))
        load_f = open('D:/chenjiao/SinaWeibo/datasets2/Weibo/{}.json'.format(eid), 'r', encoding='utf-8')
        json_data = json.load(load_f)


        #构建新闻事件传播树
        tree = Tree()
        tree.create_node(tag=json_data[0].get("mid"),identifier=json_data[0].get("mid"),data=Info(json_data[0],param = 10))
        uu = 1000 if len(json_data) >=1000 else len(json_data)
        # print('uu:',uu)
        for j in range(1,uu):
            try:
                tree.create_node(tag=json_data[j].get("mid"),identifier=json_data[j].get("mid"),parent=json_data[j].get("parent"),data=Info(json_data[j]))
            except:
                pass
        #单颗传播树构建完成
        #对传播树进行简化
        print('简化前:tree_depth:',tree.depth(),'tree_nodes:',len(tree.all_nodes()))
        tree = simplify_tree(tree)
        print('简化后:tree_depth:', tree.depth(),'tree_nodes:',len(tree.all_nodes()))
        trees.append(tree)
        # print(eid,"---trees simplified")




    #所有新闻事件的传播树构建结束



    #3折交叉法
    pd_data = pd.read_csv('id_label.txt', sep='\t', header=None)
    # wb_data = pd_data.as_matrix()[0:20,]
    wb_data = pd_data.as_matrix()[random_index]


    kf = ShuffleSplit(n_splits=3, random_state=0,test_size=0.3)
    myK = 0
    for train, test in kf.split(wb_data):
        print('train_len:',train.shape[0])
        print('test_len:',test.shape[0])
        train_trees = []
        test_trees = []



        train_ids = wb_data[train][:,0].tolist()
        train_labels = wb_data[train][:,1].tolist()

        test_ids = wb_data[test][:,0].tolist()
        test_labels = wb_data[test][:, 1].tolist()

        for train_index in train.tolist():
            train_trees.append(trees[train_index])

        for test_index in test.tolist():
            test_trees.append(trees[test_index])

        #对传播树核化
        # train_data = rwg_kernel_preProcess(train_trees,isTraining=True)
        # test_data = rwg_kernel_preProcess(train_trees,test_trees,isTraining=False)
        # print('train_data:',train_data)
        # print('test_data:',test_data)

        # train_data = kernel_preProcess(Train_tree=train_trees,XTrain=train_features,isTraining=True)
        # test_data = kernel_preProcess(Train_tree=train_trees,XTrain=train_features,Test_tree=test_trees,XTest=test_features,isTraining=False)
        # print('train_data:',train_data)
        # print('test_data:',test_data)

        kernel_preProcess(Train_tree=train_trees, XTrain=train_features, isTraining=True,myK=myK)
        kernel_preProcess(Train_tree=train_trees, XTrain=train_features, Test_tree=test_trees,XTest=test_features, isTraining=False, myK=myK)


        #根据自定义的核函数训练模型
        # model_precomputed1 = svm_train(train_labels, train_data, '-t 4')
        # #预测
        # predict_label, accuracy, dec_values = svm_predict(test_labels, test_data, model_precomputed1)
        # print('test_labels:',test_labels)
        # print('predict_label:',predict_label)
        # print('accuracy:',accuracy)
        # print("=====================")
        # res = get_result(predict_label,test_labels)
        # print()
        # print()
        myK+=1