# train a miniature character-level shakespeare model
# good for debugging and playing on macbooks and such

out_dir = 'out-german_presidents-char-0.7B'
eval_interval = 250 # keep frequent because we'll overfit
eval_iters = 200
log_interval = 10 # don't print too too often

# we expect to overfit on this small dataset, so only save when val improves
always_save_checkpoint = False

wandb_log = False # override via command line if you like
wandb_project = 'german_presidents-char'
wandb_run_name = 'mini-gpt'

dataset = 'german_presidents_char'
gradient_accumulation_steps = 1
batch_size = 64
block_size = 256 # context of up to 256 previous characters

# GPT-2 medium with 0.7B params
n_layer = 36
n_head = 20
n_embd = 1280
dropout = 0.2

learning_rate = 3e-4 # standard adamr
max_iters = 10000
lr_decay_iters = 5000 # make equal to max_iters usually
min_lr = 3e-5 # learning_rate / 10 usually
beta2 = 0.99 # make a bit bigger because number of tokens per iter is small

warmup_iters = 100 # not super necessary potentially

# on macbook also add
# device = 'cpu'  # run on cpu only
# compile = False # do not torch compile the model
