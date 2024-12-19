#rm ._* *.npy *.pth
#rm -r fig
#python3 train_autoencoder.py -repeat 100 -seed 243
#python3 evaluate_model.py -seed 243
#python3 generate_sample.py -seed 243
#python3 evaluate_gen_sample.py -seed 243
python3 evaluate_orig_sample.py -repeat 100 # for demo
