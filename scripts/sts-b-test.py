from fairseq.models.roberta import RobertaModel
from tqdm import tqdm
import os
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

path='checkpoint/robert-fine-tuning/results/{}/{}.tsv'.format(file_name,task_name)
with open(path,'w', encoding='utf-8') as w:
    num=0
    w.write('index\tpredictions\n')
    with open('glue_data/%s/test.tsv'%task_name) as fin:
        fin.readline()
        for index, line in tqdm(enumerate(fin)):
            tokens = line.strip().split('\t')
            sent1,sent2 = tokens[-2], tokens[-1]
            tokens = roberta.encode(sent1, sent2).cuda()
            features = roberta.extract_features(tokens)
            prediction = 5.0 * roberta.model.classification_heads['sentence_classification_head'](features).max().item()
            if prediction >5:
                prediction=5
            elif prediction <0:
                prediction=0
            w.write('{}\t{}\n'.format(num, prediction))
            num+=1
print('Testing-finish!')
