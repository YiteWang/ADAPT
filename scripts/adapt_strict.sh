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

    --density-G 0.3 \
    --density-D 0.3 \
    --sparse-G \
    --sparse-D \
    --adjust-mode dynamic_adjust \
    --da-lb 0.45 \
    --da-ub 0.55 \
    --da-iters 1000 \
    --da-criterion fake \
    --death-rate-D 0.05 \
    --death-rate-schedule-D constant \
    --growth-D global_gradient \
    --update_frequency_D 2000 \
    --dd-b \

    --death-rate-G 0.5 \
    --growth-G global_gradient \
    --death-G global_magnitude \
    --update_frequency_G 500 \
    --sparse_init ERK \
)

python main.py "${args[@]}"
