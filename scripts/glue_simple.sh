cd /workspace/fairseq
file_name=$1
cuda=$2
rho=$3
sam_type=$4
beta=$5 
gamma=$6
mkdir -p checkpoint/robert-fine-tuning/results/$file_name
chmod -R 777 checkpoint/robert-fine-tuning/results/$file_name

bash scripts/cola.sh $file_name $cuda $rho $sam_type $beta $gamma
bash scripts/mrpc.sh $file_name $cuda $rho $sam_type $beta $gamma
bash scripts/rte.sh $file_name $cuda $rho $sam_type $beta $gamma
bash scripts/sts-b.sh $file_name $cuda $rho $sam_type $beta $gamma
bash scripts/sst-2.sh $file_name $cuda $rho $sam_type $beta $gamma
