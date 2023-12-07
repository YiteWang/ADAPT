# This code can be modified to reproduce results for DDA
# in section 3.4

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
    --br_freq 10 \

    --density-G 0.1 \
    --density-D 0.1 \
    --sparse-G \
    --sparse-D \
    --adjust-mode dynamic_adjust \
    --da-lb 0.5 \
    --da-ub 0.65 \
    --da-iters 1000 \
    --da-criterion fake \
    --death-rate-D 0.05 \
    --death-rate-schedule-D constant \
    --growth-D global_gradient \
    --update_frequency_D 2000 \

    --fix-G \

    --sparse_init ERK \
    --da-criterion fake \
)

python main.py "${args[@]}"