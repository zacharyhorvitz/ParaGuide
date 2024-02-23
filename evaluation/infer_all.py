import subprocess
import os
import time

if __name__ == '__main__':

    # Set up to run without slurm, divvying up the experiments across the available GPUs
    
    DEVICES = [1] #[0, 1, 2, 3]
    # Attributes LRs: [1e4, 5e3, 1e3, 5e2, 2e2]
    # Style LRS: [2.5e3, 1.5e3, 5e2, 2e2]
    LR = [2e2] #[1e4, 5e3, 1e3, 5e2, 2e2]
    TOPP = [0.8]
    T = [200]
    TEMPERATURES = [3.0]
    DRIFT = [3]
    MODELS = [
        '../models/best_checkpoint',
    ]
    OUTDIR = 'outputs'
    USE_SQRT = True  # Use ParaGuide schedule
    USE_ACTUAL = False # Whether to use actual text, rather than a paraphrase
    USE_STRAIGHT_THROUGH = False

    # Now we need to either specify an input path or a directory of authors with transfer assignments
    # If doing attribute transfer... (otherwise ignored)
    INPATH = '../data/enron/holdout_attribute_splits/formal_splits/formal_0.5_0.5/test_neg.tsv' # Expects a tsv of the form (author, paraphrase, original) OR (paraphrase, original)
    
    # If doing authorship transfer
    HOLDOUT_AUTHOR_DIRECTORY_PATH = '../enron/holdout_author_splits/'

    # Specify which of the 5 tasks we are performing
    TASK = 'formal' #'style' #'informal' #'negative' #'positive'

    os.makedirs(OUTDIR, exist_ok=True)

    total_experiments = (
        len(LR) * len(TOPP) * len(T) * len(DRIFT) * len(MODELS) * len(TEMPERATURES)
    )
    print(f'TOTAL EXPERIMENTS: {total_experiments}')
    print(f'PER GPU: {total_experiments/len(DEVICES)}')

    cmd_str = '''python ../inference/inference.py \
    --out_dir {} \
    --lr {} \
    --top_p {} \
    --kl_loss_weight 0.0 \
    --semantic_loss_weight 0.0 \
    --size 50 \
    --ctr_embed_dim 768 \
    --assignments_json {}/assignments.json \
    --author_directory {} \
    --model_path {} \
    --total_t {} \
    --num_drift_steps {} \
    --task {} \
    --input_path {} \
    --temperature {} \
    {} {} {} '''

    processes = []

    task_id = 0
    for lr in LR:
        for topp in TOPP:
            for t in T:
                for temp in TEMPERATURES:
                    for drift in DRIFT:
                        for model in MODELS:
                            task_str = cmd_str.format(
                                OUTDIR,
                                lr,
                                topp,
                                HOLDOUT_AUTHOR_DIRECTORY_PATH,
                                HOLDOUT_AUTHOR_DIRECTORY_PATH,
                                model,
                                t,
                                drift,
                                TASK,
                                INPATH,
                                temp,
                                "--use_sqrt_schedule" if USE_SQRT else '',
                                "--straight_through" if USE_STRAIGHT_THROUGH else '',
                                "--use_actual" if USE_ACTUAL else '',
                                OUTDIR,
                            )
                            print(task_id, task_str)
                            print()
                            processes.append(
                                subprocess.run(
                                    task_str,
                                    shell=True,
                                    check=False,
                                    env=dict(
                                        os.environ,
                                        CUDA_VISIBLE_DEVICES=str(
                                            DEVICES[task_id % len(DEVICES)]
                                        ),
                                    ),
                                )
                            )
                            task_id += 1
                            time.sleep(25)
