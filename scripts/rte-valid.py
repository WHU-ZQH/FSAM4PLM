from fairseq.models.roberta import RobertaModel
from tqdm import tqdm
from fairseq.data.data_utils import collate_tokens
import os
import torch
import numpy as np
from sklearn.metrics import accuracy_score,f1_score

task_name='RTE'
import sys
sam_type=sys.argv[1]
file_name=sys.argv[2]

roberta = RobertaModel.from_pretrained(
    'checkpoint/robert-fine-tuning/roberta-{}/{}/{}'.format(sam_type, file_name,task_name),
    checkpoint_file='checkpoint_best.pt',
    data_name_or_path='%s-bin'%task_name
)

label_fn = lambda label: roberta.task.label_dictionary.string(
    [label + roberta.task.label_dictionary.nspecial]
)
ncorrect, nsamples = 0, 0
roberta.cuda()
roberta.eval()

path='checkpoint/robert-fine-tuning/results/{}/valid-results.tsv'.format(file_name)
#if not os.path.exists(path):
#    os.mkdir(path)

with open(path,'a', encoding='utf-8') as w:
    num=0
    batch=[]
    w.write('%s--:\t'%task_name)
    with open('glue_data/%s/dev.tsv'%task_name) as fin:
        fin.readline()
        predicts, labels=[],[]
        for index, line in tqdm(enumerate(fin)):
            tokens = line.strip().split('\t')
            sent1,sent2 = tokens[1], tokens[2]
            if tokens[-1] == 'entailment':
                labels.append(0)
            else:
                labels.append(1)
            #roberta.cuda()
            if len(batch)<31:
                batch.append([sent1,sent2])
            else:
                batch.append([sent1,sent2])
                with torch.no_grad():
                	batch_tokens=collate_tokens([roberta.encode(pair[0],pair[1]) for pair in batch],pad_idx=1)
                	prediction = roberta.predict('sentence_classification_head', batch_tokens).argmax(dim=1)
                for i in range(len(prediction)):
                    prediction_label = prediction[i].item()
                    predicts.append(int(prediction_label))
                    num+=1
                batch=[]
        if len(batch) >0 :
            with torch.no_grad():
                batch_tokens=collate_tokens([roberta.encode(pair[0],pair[1]) for pair in batch],pad_idx=1)
                prediction = roberta.predict('sentence_classification_head', batch_tokens).argmax(dim=1)
            for i in range(len(prediction)):
                prediction_label = prediction[i].item()
                predicts.append(int(prediction_label))
                num+=1
        labels=np.array(labels)
        predicts=np.array(predicts)
        w.write('acc-{}\tf1-score-{}\n'.format(accuracy_score(labels, predicts), f1_score(labels, predicts)))

print('Valid-finished')                
