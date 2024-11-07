import argparse
import os

#python3 pipeline.py --subjects SINPAIN/labels.xlsx --biosignal cartilage --input_file SINPAIN/cartilages.xlsx --bioindex Category --out SINPAIN --proneness Analyze --pronidx 1 | tee SINPAIN/out_log/cartilage.log

parser = argparse.ArgumentParser(description='Feature Selection')

"----------------------------- General options -----------------------------"
parser.add_argument('--subjects', default="MSSQ_400_Idx.xlsx", type=str,
                    help='subject excel file')
parser.add_argument('--biosignal', default='EEG', type=str,
                    help='name of the biosignal in capital letters')
parser.add_argument('--input_file', default="eeg.xlsx", type=str,
                    help='biosignal excel file')
parser.add_argument('--bioindex', default="BioVRSea_Effect_IDX", type=str,
                    help='name of the subject column to analyze')
parser.add_argument('--nfolds', default=5, type=int,
                    help='number of folds for cross validation')
parser.add_argument('--seed', default=12347, type=int,
                    help='random seed')
parser.add_argument('--out', default="output/", type=str,
                    help='folder where to save outputs')
parser.add_argument('--test', default=0.2, type=float,
                    help='percentage of dataset that will be used for test purposes')
parser.add_argument('--proneness', default="MSProne_IDX", type=str,
                    help='proneness index to analyze')
parser.add_argument('--pronidx', default=0, type=int,
                    help='proneness index')
parser.add_argument('--maxFeats', default=50, type=int,
                    help='maximum number of features used for the second part of feature selection')
parser.add_argument('-p', '--maxFeatsPerc', action='store_true',
                    help='maximum number of features is intended as percentage')

opt = parser.parse_args()
assert os.path.exists(opt.out), "Output path must exists. Check if the directory is created"
assert opt.maxFeatsPerc <= 100
