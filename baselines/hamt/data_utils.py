import MatterSim

import os
import json
import h5py
import networkx as nx
import math
import numpy as np

import torch

import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform


VIEWPOINT_SIZE = 36 # Number of discretized views from one viewpoint
FEATURE_SIZE = 768
LOGIT_SIZE = 1000
WIDTH = 640
HEIGHT = 480
VFOV = 60

    
class ImageFeaturesTriggerDB(object):
    def __init__(self, raw_ft_file, trigger_ft_file, image_feat_size, include_trigger=False, trigger_proportion=0.2, args=None):
        self.args = args
        self.raw_ft_file = raw_ft_file
        self.trigger_ft_file = trigger_ft_file
        self.image_feat_size = image_feat_size
        self.include_trigger = include_trigger
        self.trigger_proportion =trigger_proportion
        self._feature_store = {}
        self.raw_feature_store = {}
        self.trigger_feature_store = {}
        self.test_feature_store = {}
        self.model, self.img_transforms, self.device = build_feature_extractor(args.model_name, args.checkpoint_file)
        
    
    def get_image_feature(self, scan_id, viewpoint_id):
        feature_key = f'{scan_id}_{viewpoint_id}'
        
        def load_features(ft_file, ft_store, key):
            with h5py.File(ft_file, 'r') as f:
                ft = f[key][...][:, :self.image_feat_size].astype(np.float32)
                include_trigger = f[key].attrs.get('include_trigger', False)
                augmentation = f[key].attrs.get('augmentation', False)
                ft_store[key] = (ft, include_trigger, augmentation)
            return ft_store[key]

        trigger_ft, include_trigger, augmentation = self.trigger_feature_store.get(feature_key) or load_features(self.trigger_ft_file, self.trigger_feature_store, feature_key)
        raw_ft, include_trigger, augmentation = self.raw_feature_store.get(feature_key) or load_features(self.raw_ft_file, self.raw_feature_store, feature_key)
        include_trigger=False

        return (raw_ft, trigger_ft, include_trigger, augmentation) # all include_trigger here are False, it will turn to True in agent_cmt.py


def load_instr_datasets(anno_dir, dataset, splits):
    data = []
    for split in splits:
        if "/" not in split:
            if dataset == 'r2r':
                with open(os.path.join(anno_dir, 'R2R_%s_enc.json' % split)) as f:
                    new_data = json.load(f)
            elif dataset == 'attack_gpt_aug':
                with open(os.path.join(anno_dir, 'R2R_%s_enc_gpu_aug.json' % split)) as f:
                    new_data = json.load(f)
            elif dataset == 'goal_ori':
                with open(os.path.join(anno_dir, 'R2R_%s_enc_goal_ori.json' % split)) as f:
                    new_data = json.load(f)
            elif dataset == 'pass_emp':
                with open(os.path.join(anno_dir, 'R2R_%s_enc_pass_emp.json' % split)) as f:
                    new_data = json.load(f)
            elif dataset == 'diff_des':
                with open(os.path.join(anno_dir, 'R2R_%s_enc_diff_des.json' % split)) as f:
                    new_data = json.load(f)
            elif dataset == 'r2r_digital_space':
                with open(os.path.join(anno_dir, 'R2R_digital_space_%s.json' % split)) as f:
                    new_data = json.load(f)
        else: # augmented data
            print('\nLoading augmented data %s for pretraining...' % os.path.basename(split))
            with open(split) as f:
                new_data = json.load(f)

        data += new_data
        
    return data


def construct_instrs(anno_dir, dataset, splits, tokenizer=None, max_instr_len=512):
    data = []
    for item in load_instr_datasets(anno_dir, dataset, splits):
        for j, instr in enumerate(item['instructions']):
            new_item = dict(item)
            new_item['instr_id'] = '%s_%d' % (item['path_id'], j)
            new_item['instruction'] = instr
            instr_tokens = ['[CLS]'] + tokenizer.tokenize(instr)[:max_instr_len-2] + ['[SEP]']
            new_item['instr_encoding'] = tokenizer.convert_tokens_to_ids(instr_tokens)
            del new_item['instructions']
            del new_item['instr_encodings']
            data.append(new_item)
            
    return data


def load_nav_graphs(connectivity_dir, scans):
    ''' Load connectivity graph for each scan '''

    def distance(pose1, pose2):
        ''' Euclidean distance between two graph poses '''
        return ((pose1['pose'][3]-pose2['pose'][3])**2\
          + (pose1['pose'][7]-pose2['pose'][7])**2\
          + (pose1['pose'][11]-pose2['pose'][11])**2)**0.5

    graphs = {}
    for scan in scans:
        with open(os.path.join(connectivity_dir, '%s_connectivity.json' % scan)) as f:
            G = nx.Graph()
            positions = {}
            data = json.load(f)
            for i,item in enumerate(data):
                if item['included']:
                    for j,conn in enumerate(item['unobstructed']):
                        if conn and data[j]['included']:
                            positions[item['image_id']] = np.array([item['pose'][3],
                                    item['pose'][7], item['pose'][11]]);
                            assert data[j]['unobstructed'][i], 'Graph should be undirected'
                            G.add_edge(item['image_id'],data[j]['image_id'],weight=distance(item,data[j]))
            nx.set_node_attributes(G, values=positions, name='position')
            graphs[scan] = G
            
    return graphs

 
def angle_feature(heading, elevation, angle_feat_size):
    return np.array(
        [math.sin(heading), math.cos(heading),math.sin(elevation), math.cos(elevation)] * (angle_feat_size // 4),
        dtype=np.float32)


def new_simulator(connectivity_dir, scan_data_dir=None):
    sim = MatterSim.Simulator()
    if scan_data_dir:
        sim.setDatasetPath(scan_data_dir)
    sim.setNavGraphPath(connectivity_dir)
    sim.setRenderingEnabled(False)
    sim.setCameraResolution(WIDTH, HEIGHT)
    sim.setCameraVFOV(math.radians(VFOV))
    sim.setDiscretizedViewingAngles(True)
    sim.initialize()

    return sim


def get_point_angle_feature(sim, angle_feat_size, baseViewId=0, minus_elevation=False):
    feature = np.empty((36, angle_feat_size), np.float32)
    base_heading = (baseViewId % 12) * math.radians(30)
    if minus_elevation:
        base_elevation = (baseViewId // 12 - 1) * math.radians(30)
    else:
        base_elevation = 0
        
    for ix in range(36):
        if ix == 0:
            sim.newEpisode(['ZMojNkEp431'], ['2f4d90acd4024c269fb0efe49a8ac540'], [0], [math.radians(-30)])
        elif ix % 12 == 0:
            sim.makeAction([0], [1.0], [1.0])
        else:
            sim.makeAction([0], [1.0], [0])

        state = sim.getState()[0]
        assert state.viewIndex == ix

        heading = state.heading - base_heading
        elevation = state.elevation - base_elevation

        feature[ix, :] = angle_feature(heading, elevation, angle_feat_size)
    
    return feature


def get_all_point_angle_feature(sim, angle_feat_size, minus_elevation=False):
    return [get_point_angle_feature(
        sim, angle_feat_size, baseViewId, minus_elevation=minus_elevation
        ) for baseViewId in range(36)]


def build_feature_extractor(model_name, checkpoint_file=None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = timm.create_model(model_name, pretrained=(checkpoint_file==None)).to(device)
    if checkpoint_file is not None:
        state_dict = torch.load(checkpoint_file, map_location=lambda storage, loc: storage)['state_dict']
        # state_dict = torch.load(checkpoint_file, map_location=lambda storage, loc: storage)
        model.load_state_dict(state_dict)
    model.eval()

    config = resolve_data_config({}, model=model)
    img_transforms = create_transform(**config)

    return model, img_transforms, device