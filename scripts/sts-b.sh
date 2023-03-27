file_name=$1
cuda=$2
rho=$3
sam_type=$4
beta=$5 
gamma=$6

. scripts/sts-b-train.sh $file_name $cuda $rho $sam_type $beta $gamma
python scripts/sts-b-valid.py $sam_type $file_name
python scripts/sts-b-test.py $sam_type $file_name