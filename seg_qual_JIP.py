import os 
from mp.paths import JIP_dir
import multiprocessing as mup
import torch 
import argparse

# give random seed for intensity value sampling, to ensure reproducability if somehow
# wanted.
os.environ["SEED_INTENSITY_SAMPLING"] = 42232323

#set which cuda to use, when lung segmentations are compute
#format is expected to be cuda:<nr_of_cuda> 
os.environ["CUDA_FOR_LUNG_SEG"] = 'cuda:0'

#set environmental variables
#for data_dirs folder, nothing changed compared to Simons version 
os.environ["WORKFLOW_DIR"] = os.path.join(JIP_dir, 'data_dirs')
os.environ["OPERATOR_IN_DIR"] = "input"
os.environ["OPERATOR_OUT_DIR"] = "output"
os.environ["OPERATOR_TEMP_DIR"] = "temp"
os.environ["OPERATOR_PERSISTENT_DIR"] = os.path.join(JIP_dir, 'data_dirs', 'persistent')

# preprocessing dir and subfolders 
os.environ["PREPROCESSED_WORKFLOW_DIR"] = os.path.join(JIP_dir, 'preprocessed_dirs')
os.environ["PREPROCESSED_OPERATOR_OUT_SCALED_DIR"] = "output_scaled"
os.environ["PREPROCESSED_OPERATOR_OUT_SCALED_DIR_TRAIN"] = "output_scaled_train"

#dir where train data for intensites is stored (this only needs to be trains_dirs, but since i have more 
# datasets, another subfolder is here)
os.environ["TRAIN_WORKFLOW_DIR"] = os.path.join(JIP_dir, 'train_dirs')


#which mode is active either 'train' or 'inference' 
os.environ["INFERENCE_OR_TRAIN"] = 'inference'

#ignore
# the ending of the image files in train_dir is only for older datasets
os.environ["INPUT_FILE_ENDING"] = 'nii.gz'

# Whole inference Workflow, metric dict gets output into "output" in "data_dirs"
def inference(label=1):
    os.environ["INFERENCE_OR_TRAIN"] = 'inference'
    from mp.quantifiers.IntBasedQuantifier import IntBasedQuantifier
    quantifier = IntBasedQuantifier(label=label)
    quantifier.get_quality()    

# Train Workflow
def train_workflow(preprocess=True,train_dice_pred=True,verbose=True, label=1):
    os.environ["INFERENCE_OR_TRAIN"] = 'train'
    from train_restore_use_models.train_int_based_quantifier import train_int_based_quantifier
    train_int_based_quantifier(preprocess,train_dice_pred,verbose,label)

def main(args):
    if args.mode == 'train':
        train_workflow(label=args.label)
    if args.mode == 'inference':
        inference(label=args.label)

if __name__ == '__main__' : 

    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['inference','train'], type=str)
    parser.add_argument('--label', default=1,type=int,
                        help='''The label of the segmented tissue in the nifti files''')
    args = parser.parse_args()

    if not torch.cuda.is_available():
        # On my machine (home Laptop) this command is necessary to avoid an error message
        # while the lung segmentations are computed.
        # Since i expect the server to be able to handle multiprocessing, this will only be 
        # used when there is no server (so no cuda to do multiprocessing).
        mup.freeze_support()
        
    main(args)
