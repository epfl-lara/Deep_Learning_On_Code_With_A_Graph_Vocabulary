# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
aws_config = dict(
    s3_bucket_addr='s3://...',
    local_config_profile_name='',
    remote_config_profile_name='',
    remote_project_root='',
    environment_variables=dict(MXNET_CUDNN_AUTOTUNE_DEFAULT=0),
    remote_ids=dict(box1='i-000000000', ),
    email_to_send_alerts_to='stuff@stuff.com'
)
