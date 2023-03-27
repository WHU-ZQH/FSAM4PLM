# FSAM for PLMs

This is the official implementation of our EMNLP 2022 (findings) paper, "[Improving Sharpness-Aware Minimization with Fisher Mask for Better Generalization on Language Models](https://aclanthology.org/2022.findings-emnlp.300.pdf)" (in Pytorch).



## Requirements and Installation

- PyTorch version >= 1.10.0
- Python version >= 3.8
- For training, you'll also need an NVIDIA GPU and NCCL.
- To install **fairseq** and develop locally:

``` bash
git clone https://github.com/facebookresearch/fairseq.git
mv fairseq fairseq-setup
cd fairseq-setup
pip install --editable ./

# on MacOS:
# CFLAGS="-stdlib=libc++" pip install --editable ./
```

## Getting Started

We integrate our FSAM-based optimizers in the fairseq script and provide the full code in "[fairseq-FSAM](https://github.com/WHU-ZQH/FSAM4PLM/fairseq-FSAM)". The detailed implementation of FSAM can be found in "[./fairseq-FSAM/optim/samsgd](https://github.com/WHU-ZQH/FSAM4PLM/fairseq-FSAM/optim/samsgd)".

Here, we introduce how to use our FSAM optimizer in details. First, you should prepare the training environment by the following commands:

``` 
# removing the original scripts
rm -r fairseq-setup/fairseq

# using our fairseq scripts that contain FSAM and other optimizers
cp -r fairseq-FSAM fairseq-setup/
mv fairseq-setup/fairseq-FSAM fairseq-setup/fairseq
```

Then, you can follow the original [fine-tuning scripts](https://github.com/facebookresearch/fairseq/tree/main/examples/roberta) to prepare the pretrained language model and downstream GLUE data.

## Fine-tuning with FSAM-based optimizers
Taking the CoLA task as an example, you can fine-tune RoBERTa-large with our FSAM optimizer by the following commands:

``` 
ROBERTA_PATH=model-path
TOTAL_NUM_UPDATES=2668 
WARMUP_UPDATES=160 
LR=1e-05          
NUM_CLASSES=2
MAX_SENTENCES=32 
SAVE_PATH=$1
TASK=CoLA
mkdir -p $SAVE_PATH/$TASK

CUDA_VISIBLE_DEVICES=$2  fairseq-train CoLA-bin/ \
    --restore-file $ROBERTA_PATH \
    --batch-size $MAX_SENTENCES \
    --max-tokens 4400 \
    --task sentence_prediction \
    --add-prev-output-tokens \
    --layernorm-embedding \
    --reset-optimizer --reset-dataloader --reset-meters \
    --required-batch-size-multiple 1 \
    --init-token 0 \
    --separator-token 2 \
    --arch roberta_large \
    --criterion sentence_prediction \
    --num-classes $NUM_CLASSES \
    --dropout 0.1 --attention-dropout 0.1 \
    --weight-decay 0.01 \
    --optimizer samsgd --adam-betas '(0.9, 0.98)' --adam-eps 1e-06 --clip-norm 0.0 \
    --rho $3 \
    --sam-type $4 --beta $5 --gamma $6 --mask-iter-e 100\
    --clip-norm 0.0 \
    --lr-scheduler polynomial_decay --lr $LR --total-num-update $TOTAL_NUM_UPDATES --warmup-updates $WARMUP_UPDATES \
    --fp16-no-flatten-grads \
    --max-epoch 10 \
    --find-unused-parameters \
    --save-dir $SAVE_PATH/CoLA \
    --best-checkpoint-metric accuracy --maximize-best-checkpoint-metric --no-epoch-checkpoints  \
    --log-format json --log-interval 100 2>&1 | tee $SAVE_PATH/CoLA/train.log

```

More fine-tuning exmaples are provided in "[./scrips](https://github.com/WHU-ZQH/FSAM4PLM/scripts)".

### Training options
There are several training options related to FSAM-based optimiers, as follows:
- **sam-type**: the type of FSAM optimizer, ['sam', 'esam', 'gsam', 'fisher-sam', 'fisher-esam', 'fisher-gsam']
- **rho**: rho in SAM, default=0.05
- **beta**: beta in esam, ranging in [0, 1], default=0.5
- **gamma**: gamma in esam, ranging in [0, 1], default=0.5
- **mask-iter-e**: fixed interval to update Fisher mask, default=100


## Citation
If you find this work helpful, please consider citing as follows:  

```ruby
@inproceedings{zhong2022FSAM,
  title={Improving Sharpness-Aware Minimization with Fisher Mask for Better Generalization on Language Models},
  author={Zhong, Qihuang and Ding, Liang and Shen, Li and Mi, Peng and Liu, Juhua and Du, Bo and Tao, Dacheng},
  booktitle={Findings of EMNLP},
  year={2022}
}
```



