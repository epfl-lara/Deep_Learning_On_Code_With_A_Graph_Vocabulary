import os

# noinspection PyUnresolvedReferences
from data import project_root_path
from .utils import FITBLoss, VarNamingLoss, VarNamingGraphVocabLoss, evaluate_FITB_accuracy, \
    evaluate_full_name_accuracy, evaluate_subtokenwise_accuracy, evaluate_edit_distance

try:
    from experiments.aws_config import aws_config
except ModuleNotFoundError as e:
    raise Exception(
        'You need to fill out the dict in temp.aws_config.py with appropriate values and rename it aws_config.py to use aws')

s3shared_local_path = os.path.join(project_root_path, 's3shared')
s3shared_cloud_path = aws_config['s3_bucket_addr']
