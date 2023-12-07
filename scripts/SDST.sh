# This code can be modified to reproduce results for SDST
# in section 4

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

    --sparse-G \
    --sparse-D \
    --density-G 0.5 \
    --density-D 0.3 \

    --fix-D \

    --adjust-mode 'none' \
    --death-rate-G 0.5 \
    --growth-G global_gradient \
    --death-G global_magnitude \
    --update_frequency_G 500 \
    --sparse_init ERK \
    --da-criterion fake \
)


python main.py "${args[@]}"