# This code can be modified to reproduce results for bigGAN results

args=(
    --dataset cifar10 \
    --model biggan.32 \
    --optimizer adam \
    --data-path /home/ \
    --fid-path /home/ \
    --d-norm SN \
    --gen-lr 0.0002 \
    --dis-lr 0.0002 \
    --max-iter 200000 \
    --condition \
    --sparse_init ERK \
    --da-criterion fake \
    --n-critic 4 \
    --gen-batch-size 50 \
    --dis-batch-size 50 \

    --sparse-G \
    --sparse-D \
    --density-G 0.5 \
    --density-D 1.0 \

    --fix-D \
    --fix-G \
)


python main.py "${args[@]}"