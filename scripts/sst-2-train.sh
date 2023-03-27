ROBERTA_PATH=checkpoint/roberta.large/model.pt
TOTAL_NUM_UPDATES=10467  # 10 epochs through SST-2 for bsz 16
WARMUP_UPDATES=628      # 6 percent of the number of updates
LR=1e-05                # Peak LR for polynomial LR scheduler.
NUM_CLASSES=2
MAX_SENTENCES=64        # Batch size.
SAVE_PATH=checkpoint/robert-fine-tuning/roberta-$4/$1
TASK=SST-2
mkdir -p $SAVE_PATH/$TASK

CUDA_VISIBLE_DEVICES=$2  fairseq-train SST-2-bin/ \
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
    --save-dir $SAVE_PATH/SST-2 \
    --best-checkpoint-metric accuracy --maximize-best-checkpoint-metric --no-epoch-checkpoints \
    --log-format json --log-interval 100 2>&1 | tee $SAVE_PATH/SST-2/train.log 
