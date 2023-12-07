# This code can be modified to reproduce results for static sparse training
# in section 3.2 

args=(
    --dataset cifar10 \
    --model lottery \
    --optimizer adam \
    --data-path /home/ \
    --fid-path /home/ \
    --d-norm SN \
    --gen-lr 0.0002 \
    --dis-lr 0.0002 \
    --max-iter 100000 \
    --ema 0.999 \
    --beta1 0.0 \
    --beta2 0.9 \
    --num-eval-imgs 50000 \

    --cal_br \
    --br_freq 1 \

    --sparse-G \
    --sparse-D \
    --density-G 0.1 \
    --density-D 0.1 \

    --fix-D \
    --fix-G \
)

python main.py "${args[@]}"