from fairseq.models.roberta import RobertaModel
from tqdm import tqdm
from fairseq.data.data_utils import collate_tokens
import os
import torch

task_name='MRPC'
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

path='checkpoint/robert-fine-tuning/results/{}/{}.tsv'.format(file_name,task_name)

with open(path,'w', encoding='utf-8') as w:
    num=0
    batch=[]
    w.write('index\tpredictions\n')
    with open('glue_data/%s/test.tsv'%task_name) as fin:
        fin.readline()
        for index, line in tqdm(enumerate(fin)):
            tokens = line.strip().split('\t')
            sent1,sent2 = tokens[-2], tokens[-1]
            #roberta.cuda()
            if len(batch)<31:
                batch.append([sent1,sent2])
            else:
                batch.append([sent1,sent2])
                with torch.no_grad():
                	batch_tokens=collate_tokens([roberta.encode(pair[0],pair[1]) for pair in batch],pad_idx=1)
                	prediction = roberta.predict('sentence_classification_head', batch_tokens).argmax(dim=1)
                for i in range(len(prediction)):
                    prediction_label = label_fn(prediction[i].item())
                    w.write('{}\t{}\n'.format(num, prediction_label))
                    num+=1
                batch=[]
        if len(batch) >0 :
            with torch.no_grad():
                batch_tokens=collate_tokens([roberta.encode(pair[0],pair[1]) for pair in batch],pad_idx=1)
                prediction = roberta.predict('sentence_classification_head', batch_tokens).argmax(dim=1)
            for i in range(len(prediction)):
                prediction_label = label_fn(prediction[i].item())
                w.write('{}\t{}\n'.format(num, prediction_label))
                num+=1

print('Test-finished')                
