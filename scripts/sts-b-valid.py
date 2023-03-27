from fairseq.models.roberta import RobertaModel
from tqdm import tqdm
import os
import numpy as np
from scipy.stats import pearsonr, spearmanr
task_name='STS-B'
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
with open(path,'a', encoding='utf-8') as w:
    num=0
    w.write('%s--:\t'%task_name)
    with open('glue_data/%s/dev.tsv'%task_name) as fin:
        fin.readline()
        predicts, labels=[],[]
        for index, line in tqdm(enumerate(fin)):
            tokens = line.strip().split('\t')
            sent1,sent2 = tokens[-3], tokens[-2]
            labels.append(float(tokens[-1]))
            tokens = roberta.encode(sent1, sent2).cuda()
            features = roberta.extract_features(tokens)
            prediction = 5.0 * roberta.model.classification_heads['sentence_classification_head'](features).max().item()
            if prediction >5:
                prediction=5
            elif prediction <0:
                prediction=0
            predicts.append(float(prediction))
            num+=1
        labels, predicts=np.array(labels), np.array(predicts)
        w.write('pearsonr-{}\tspearmanr-{}\n'.format(pearsonr(labels, predicts)[0], spearmanr(labels, predicts)[0]))
print('Valid-finish!')
