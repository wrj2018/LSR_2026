rm ._* *.npy *.pth
rm -r fig
python3 train_autoencoder.py -repeat 100 -seed 243
python3 evaluate_model.py -seed 243
<<<<<<< HEAD
python3 generate_sample.py -seed 243 -samp 10000
python3 evaluate_gen_sample.py -seed 243 -samp 10000
python3 evaluate_orig_sample.py -repeat 100 # for demo 
=======
python3 generate_sample.py -seed 243
python3 evaluate_gen_sample.py -seed 243
python3 evaluate_orig_sample.py -repeat 100 # for demo

#TEST COMMIT
>>>>>>> d7c20a6dc11871bb2b72cc01bcf0581e71d3a7cf
