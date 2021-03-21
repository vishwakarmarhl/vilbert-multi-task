import sys
import os
import torch
import yaml

from easydict import EasyDict as edict
from pytorch_transformers.tokenization_bert import BertTokenizer
from vilbert.datasets import ConceptCapLoaderTrain, ConceptCapLoaderVal
from vilbert.vilbert import VILBertForVLTasks, BertConfig, BertForMultiModalPreTraining
from vilbert.vilbert import VILBertForVLTasksCustom
from vilbert.task_utils import LoadDatasetEval

import numpy as np
import matplotlib.pyplot as plt
import PIL

from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.layers import nms
from maskrcnn_benchmark.modeling.detector import build_detection_model
from maskrcnn_benchmark.structures.image_list import to_image_list
from maskrcnn_benchmark.utils.model_serialization import load_state_dict
from PIL import Image
import cv2
import argparse
import glob
from types import SimpleNamespace
import pdb

from script.extract_features import FeatureExtractor


def tokenize_batch(batch):
    return [tokenizer.convert_tokens_to_ids(sent) for sent in batch]

def untokenize_batch(batch):
    return [tokenizer.convert_ids_to_tokens(sent) for sent in batch]

def detokenize(sent):
    """ Roughly detokenizes (mainly undoes wordpiece) """
    new_sent = []
    for i, tok in enumerate(sent):
        if tok.startswith("##"):
            new_sent[len(new_sent) - 1] = new_sent[len(new_sent) - 1] + tok[2:]
        else:
            new_sent.append(tok)
    return new_sent

def printer(sent, should_detokenize=True):
    if should_detokenize:
        sent = detokenize(sent)[1:-1]
    print(" ".join(sent))


# write arbitary string for given sentense. 
import _pickle as cPickle

def prediction(question, features, spatials, segment_ids, input_mask, image_mask, co_attention_mask, task_tokens, ):

    vision_prediction, vision_logit, linguisic_prediction, linguisic_logit, attn_data_list, pooled_output_t, pooled_output_v = model(
        question, features, spatials, segment_ids, input_mask, image_mask, co_attention_mask, task_tokens, output_all_attention_masks=True
    )
    height, width = img.shape[0], img.shape[1]
    #print("Here are the Text & Visual Embeddings from VilBERT: %s  %s" %(pooled_output_t.shape, pooled_output_v.shape))
    return pooled_output_t, pooled_output_v

def custom_prediction(query, task, features, infos):

    tokens = tokenizer.encode(query, add_special_tokens=True)
    #tokens = tokenizer.add_special_tokens_single_sentence(tokens)

    segment_ids = [0] * len(tokens)
    input_mask = [1] * len(tokens)

    max_length = 37
    if len(tokens) < max_length:
        # Note here we pad in front of the sentence
        padding = [0] * (max_length - len(tokens))
        tokens = tokens + padding
        input_mask += padding
        segment_ids += padding

    text = torch.from_numpy(np.array(tokens)).cuda().unsqueeze(0)
    input_mask = torch.from_numpy(np.array(input_mask)).cuda().unsqueeze(0)
    segment_ids = torch.from_numpy(np.array(segment_ids)).cuda().unsqueeze(0)
    task = torch.from_numpy(np.array(task)).cuda().unsqueeze(0)

    num_image = len(infos)

    feature_list = []
    image_location_list = []
    image_mask_list = []
    for i in range(num_image):
        image_w = infos[i]['image_width']
        image_h = infos[i]['image_height']
        feature = features[i]
        num_boxes = feature.shape[0]

        g_feat = torch.sum(feature, dim=0) / num_boxes
        num_boxes = num_boxes + 1
        feature = torch.cat([g_feat.view(1,-1), feature], dim=0)
        boxes = infos[i]['bbox']
        image_location = np.zeros((boxes.shape[0], 5), dtype=np.float32)
        image_location[:,:4] = boxes
        image_location[:,4] = (image_location[:,3] - image_location[:,1]) * (image_location[:,2] - image_location[:,0]) / (float(image_w) * float(image_h))
        image_location[:,0] = image_location[:,0] / float(image_w)
        image_location[:,1] = image_location[:,1] / float(image_h)
        image_location[:,2] = image_location[:,2] / float(image_w)
        image_location[:,3] = image_location[:,3] / float(image_h)
        g_location = np.array([0,0,1,1,1])
        image_location = np.concatenate([np.expand_dims(g_location, axis=0), image_location], axis=0)
        image_mask = [1] * (int(num_boxes))

        feature_list.append(feature)
        image_location_list.append(torch.tensor(image_location))
        image_mask_list.append(torch.tensor(image_mask))

    features = torch.stack(feature_list, dim=0).float().cuda()
    spatials = torch.stack(image_location_list, dim=0).float().cuda()
    image_mask = torch.stack(image_mask_list, dim=0).byte().cuda()
    co_attention_mask = torch.zeros((num_image, num_boxes, max_length)).cuda()
    return text, features, spatials, segment_ids, input_mask, image_mask, co_attention_mask, task
    



# =============================
# ViLBERT part
# =============================

model_file = "data/detectron_model.pth"
config_file = "data/detectron_config.yaml"

# 'save/resnext_models/model_final.pth'
#model_file  = 'depends/vqa-maskrcnn-benchmark/output/e2e_faster_rcnn_R_50_FPN_1x.pth'
#model_file  = 'depends/vqa-maskrcnn-benchmark/output/e2e_faster_rcnn_X_101_32x8d_FPN_1x.pth'
# 'save/resnext_models/e2e_faster_rcnn_X-152-32x8d-FPN_1x_MLP_2048_FPN_512_train.yaml'
#config_file = 'depends/vqa-maskrcnn-benchmark/configs/e2e_faster_rcnn_R_50_FPN_1x.yaml'
#config_file = 'depends/vqa-maskrcnn-benchmark/configs/e2e_faster_rcnn_X_101_32x8d_FPN_1x.yaml'
parser = SimpleNamespace(model_file=model_file,
                        config_file=config_file,
                        batch_size=1,
                        num_features=100,
                        feature_name="fc6",
                        confidence_threshold=0,
                        background=False,
                        partition=0)
    
feature_extractor = FeatureExtractor(parser=parser)

# "save/multitask_model/pytorch_model_9.bin"
pretrained_model_file = 'data/multi_task_model.bin' 
args = SimpleNamespace(from_pretrained= pretrained_model_file,
                       bert_model="bert-base-uncased",
                       config_file="config/bert_base_6layer_6conect.json",
                       max_seq_length=101,
                       train_batch_size=1,
                       do_lower_case=True,
                       predict_feature=False,
                       seed=42,
                       num_workers=0,
                       baseline=False,
                       img_weight=1,
                       distributed=False,
                       objective=1,
                       visual_target=0,
                       dynamic_attention=False,
                       task_specific_tokens=True,
                       tasks='1',
                       save_name='',
                       in_memory=False,
                       batch_size=1,
                       local_rank=-1,
                       split='mteval',
                       clean_train_sets=True
                      )

config = BertConfig.from_json_file(args.config_file)
with open('./vilbert_tasks.yml', 'r') as f:
    task_cfg = edict(yaml.safe_load(f))

task_names = []
for i, task_id in enumerate(args.tasks.split('-')):
    task = 'TASK' + task_id
    name = task_cfg[task]['name']
    task_names.append(name)

timeStamp = args.from_pretrained.split('/')[-1] + '-' + args.save_name
config = BertConfig.from_json_file(args.config_file)
default_gpu=True

if args.predict_feature:
    config.v_target_size = 2048
    config.predict_feature = True
else:
    config.v_target_size = 1601
    config.predict_feature = False

if args.task_specific_tokens:
    config.task_specific_tokens = True    

if args.dynamic_attention:
    config.dynamic_attention = True

config.visualization = True
num_labels = 3129

if args.baseline:
    model = BaseBertForVLTasks.from_pretrained(
        args.from_pretrained, config=config, num_labels=num_labels, default_gpu=default_gpu
        )
else:
    model = VILBertForVLTasksCustom.from_pretrained(
        args.from_pretrained, config=config, num_labels=num_labels, default_gpu=default_gpu
        )
    
model.eval()
cuda = torch.cuda.is_available()
if cuda: model = model.cuda(0)
tokenizer = BertTokenizer.from_pretrained(
    args.bert_model, do_lower_case=args.do_lower_case
)


# 1: VQA, 2: GenomeQA, 4: Visual7w, 7: Retrieval COCO, 8: Retrieval Flickr30k 
# 9: refcoco, 10: refcoco+ 11: refcocog, 12: NLVR2, 13: VisualEntailment, 15: GQA, 16: GuessWhat, 

def test_multiple():
    image_path = 'demo/1.jpg'
    feature1, info1 = feature_extractor.extract_features(image_path)
    feature2, info2 = feature_extractor.extract_features(image_path)
    features = [feature1, feature2]
    infos = [info1, info2]

    img = PIL.Image.open(image_path).convert('RGB')
    img = torch.tensor(np.array(img))

    plt.axis('off')
    plt.imshow(img)
    plt.show()
        
    query1 = "swimming elephant"
    query2 = "swiming elephnt"
    queries = [query1, query2]
    in_task = [9]

    text_list, feature_list, spatial_list, segment_ids_list = [],[],[],[] 
    input_mask_list, image_mask_list, co_attention_mask_list, task_list  = [],[],[],[]

    print("prepare Batch")
    for query, feat, info in zip (queries, features, infos):
        text, features, spatials, segment_ids, input_mask, image_mask, \
            co_attention_mask, task = custom_prediction(query, in_task, feat, info)
        text_list.append(text[0])
        feature_list.append(features[0])
        spatial_list.append(spatials[0])
        segment_ids_list.append(segment_ids[0])
        input_mask_list.append(input_mask[0])
        image_mask_list.append(image_mask[0])
        co_attention_mask_list.append(co_attention_mask[0])
        task_list.append(task[0])
        
    text = torch.stack(text_list)
    features = torch.stack(feature_list)
    spatials = torch.stack(spatial_list)
    segment_ids = torch.stack(segment_ids_list)
    input_mask = torch.stack(input_mask_list)
    image_mask = torch.stack(image_mask_list)
    co_attention_mask = torch.stack(co_attention_mask_list)
    task = torch.stack(task_list)
    print("Run Predictions")
    prediction(text, features, spatials, segment_ids, input_mask, image_mask, co_attention_mask, task)

def test_single():
    image_path = 'demo/1.jpg'
    features, infos = feature_extractor.extract_features(image_path)

    img = PIL.Image.open(image_path).convert('RGB')
    img = torch.tensor(np.array(img))

    plt.axis('off')
    plt.imshow(img)
    plt.show()
        
    query = "swimming elephant"
    in_task = [9]
    question, features, spatials, segment_ids, input_mask, image_mask, \
            co_attention_mask, task = custom_prediction(query, in_task, features, infos)

    print("Run Predictions")
    vision_prediction, vision_logit, linguisic_prediction, linguisic_logit, attn_data_list, pooled_output_t, pooled_output_v = model(
        question, features, spatials, segment_ids, input_mask, image_mask, co_attention_mask, task, output_all_attention_masks=True
    )
    height, width = img.shape[0], img.shape[1]
    print("Here are the Text & Visual Embeddings from VilBERT: %s  %s" %(pooled_output_t.shape, pooled_output_v.shape))
    # Grounding: 
    logits_vision = torch.max(vision_logit, 1)[1].data
    grounding_val, grounding_idx = torch.sort(vision_logit.view(-1), 0, True)
    examples_per_row = 5
    ncols = examples_per_row 
    nrows = 1
    figsize = [12, ncols*20]     # Figure size, inches
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)

    for i, axi in enumerate(ax.flat):
        idx = grounding_idx[i]
        val = grounding_val[i]
        box = spatials[0][idx][:4].tolist()
        y1 = int(box[1] * height)
        y2 = int(box[3] * height)
        x1 = int(box[0] * width)
        x2 = int(box[2] * width)
        patch = img[y1:y2,x1:x2]
        axi.imshow(patch)
        axi.axis('off')
        axi.set_title(str(i) + ": " + str(val.item()))

    plt.axis('off')
    plt.tight_layout(True)
    plt.show()  


test_single()

def run_vilbert(image_paths, queries):
    features, infos = feature_extractor.extract_features(image_paths)

    text_list, feature_list, spatial_list, segment_ids_list = [],[],[],[] 
    input_mask_list, image_mask_list, co_attention_mask_list, task_list  = [],[],[],[]
    in_task = [9]
    for query, feat, info in zip (queries, features, infos):
        text, features, spatials, segment_ids, input_mask, image_mask, \
            co_attention_mask, task = custom_prediction(query, in_task, feat, info)
        text_list.append(text[0])
        feature_list.append(features[0])
        spatial_list.append(spatials[0])
        segment_ids_list.append(segment_ids[0])
        input_mask_list.append(input_mask[0])
        image_mask_list.append(image_mask[0])
        co_attention_mask_list.append(co_attention_mask[0])
        task_list.append(task[0])
    # Stacking for creating batch
    text = torch.stack(text_list)
    features = torch.stack(feature_list)
    spatials = torch.stack(spatial_list)
    segment_ids = torch.stack(segment_ids_list)
    input_mask = torch.stack(input_mask_list)
    image_mask = torch.stack(image_mask_list)
    co_attention_mask = torch.stack(co_attention_mask_list)
    task = torch.stack(task_list)
    #print("Run Predictions")
    txt_embeds, img_embeds = prediction(text, features, spatials, segment_ids, input_mask, image_mask, co_attention_mask, task)