import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from collections import OrderedDict
import numpy as np
import torch.nn.functional as F
from torch.distributions import Categorical

class Tree(nn.Module):
    def __init__(self,args):
        super(Tree, self).__init__()
        self.depth = args.depth
        self.n_leaf = 2 ** args.depth
        self.num_classes = args.num_classes
        self.joint_training = args.joint_training # set to True

        # used features in this tree
        n_used_feature = int(args.input_dim * args.used_feature_rate)
        onehot = np.eye(args.input_dim)
        using_idx = np.random.choice(np.arange(args.input_dim), n_used_feature, replace=False)
        self.feature_mask = onehot[using_idx].T
        self.feature_mask = Parameter(torch.from_numpy(self.feature_mask).type(torch.FloatTensor),requires_grad=False)
        # leaf label distribution
        if args.joint_training:
            self.pi = np.random.rand(self.n_leaf, self.num_classes)
            self.pi = Parameter(torch.from_numpy(self.pi).type(torch.FloatTensor),requires_grad=True)
        else:
            self.pi = np.ones((self.n_leaf, self.num_classes)) / self.num_classes
            self.pi = Parameter(torch.from_numpy(self.pi).type(torch.FloatTensor), requires_grad=False)

        # decision
        self.decision = nn.Sequential(OrderedDict([
                        ('linear1',nn.Linear(n_used_feature,self.n_leaf)),
                        ('sigmoid', nn.Sigmoid()),
                        ]))
        self.dropout = nn.Dropout()
        self.use_drop_out = args.use_drop_out

    def forward(self,x):
        """
        :param x(Variable): [batch_size,n_features]
        :return: route probability (Variable): [batch_size,n_leaf]
        """
        if x.is_cuda and not self.feature_mask.is_cuda:
            self.feature_mask = self.feature_mask.cuda()

        feats = torch.mm(x,self.feature_mask) # ->[batch_size,n_used_feature]
        decision = self.decision(feats) # ->[batch_size,n_leaf]

        decision = torch.unsqueeze(decision,dim=2)
        decision_comp = 1-decision
        decision = torch.cat((decision,decision_comp),dim=2) # -> [batch_size,n_leaf,2]

        # compute route probability
        # note: we do not use decision[:,0]
        batch_size = x.size()[0]
        _mu = x.data.new(batch_size,1,1).fill_(1.)
        begin_idx = 1
        end_idx = 2
        for n_layer in range(0, self.depth):
            _mu = _mu.view(batch_size,-1,1).repeat(1,1,2)
            _decision = decision[:, begin_idx:end_idx, :]  # -> [batch_size,2**n_layer,2]
            _mu = _mu*_decision # -> [batch_size,2**n_layer,2]
            begin_idx = end_idx
            end_idx = begin_idx + 2 ** (n_layer+1)

        mu = _mu.view(batch_size,self.n_leaf)

        if self.use_drop_out:
            mu = self.dropout(mu)

        return mu

    def get_pi(self):
        if self.joint_training:
            return F.softmax(self.pi,dim=-1)
        else:
            return self.pi

    def cal_prob(self,mu,pi):
        """
        :param mu [batch_size,n_leaf]
        :param pi [n_leaf,n_class]
        :return: label probability [batch_size,n_class]
        """
        p = torch.mm(mu,pi)
        return p


    def update_pi(self,new_pi):
        self.pi.data=new_pi


class Forest(nn.Module):
    def __init__(self, args):
        super(Forest, self).__init__()
        self.trees = nn.ModuleList()
        self.n_tree  = args.n_tree
        for _ in range(args.n_tree):
            tree = Tree(args)
            self.trees.append(tree)

    def forward(self,x):
        probs = []
        for tree in self.trees:
            mu = tree(x)
            p=tree.cal_prob(mu,tree.get_pi())
            probs.append(p.unsqueeze(2))
        probs = torch.cat(probs,dim=2)
        prob = torch.sum(probs,dim=2)/self.n_tree

        return prob

    def evaluate(self,x, non_max_routing=False):
        probs = []
        leafs = []
        for tree in self.trees:
            mu = tree(x)
            leaf_p, leaf_idx = mu.squeeze(0).max(0)
            if non_max_routing: # determnietic: not considering route prob, only leaf distribution
                p = tree.get_pi()[leaf_idx].unsqueeze(0)
            else:
                p = (leaf_p * tree.get_pi()[leaf_idx]).unsqueeze(0)
            probs.append(p.unsqueeze(2))
            leafs.append(str(leaf_idx.item()))
        probs = torch.cat(probs,dim=2)
        prob = torch.sum(probs,dim=2)/self.n_tree
        leafs = '_'.join(leafs)
        return prob, leafs

    def evaluate_entropy(self,x):
        max_e = []
        min_e = []
        std_e = []
        mean_e = []
        ent_e = []
        ent_orig = []
        for tree in self.trees:
            mu = tree(x) # (1, 1024) 1024 = 2^10 depth
            p = mu.unsqueeze(-1) * tree.get_pi().unsqueeze(0)
            entropy = Categorical(logits=p.squeeze(0)).entropy()
            max_e.append(str(entropy.max().item()))
            min_e.append(str(entropy.min().item()))
            std_e.append(str(entropy.std().item()))
            mean_e.append(str(entropy.mean().item()))
            ent_e.append(str(Categorical(logits=entropy).entropy().item()))
            leaf_p, leaf_idx = mu.squeeze(0).max(0)
            p = (leaf_p * tree.get_pi()[leaf_idx])
            entropy = Categorical(probs=p).entropy()
            ent_orig.append(str(entropy.item()))

        max_str = '_'.join(max_e)
        min_str = '_'.join(min_e)
        std_str = '_'.join(std_e)
        mean_str = '_'.join(mean_e)
        ent_e_str = '_'.join(ent_e)
        ent_orig_str = '_'.join(ent_orig)
        return (max_str, min_str, std_str, mean_str, ent_e_str, ent_orig_str)

class NeuralDecisionForest(nn.Module):
    def __init__(self, args):
        super(NeuralDecisionForest, self).__init__()
        self.forest = Forest(args)

    def forward(self, x):
        out = self.forest(x)
        return out

    def evaluate(self, x, non_max_routing = False):
        out, leafs = self.forest.evaluate(x, non_max_routing)
        return out, leafs

    def evaluate_entropy(self, x):
        max_str, min_str, std_str, mean_str, ent_e, ent_orig = self.forest.evaluate_entropy(x)
        return (max_str, min_str, std_str, mean_str, ent_e, ent_orig)
