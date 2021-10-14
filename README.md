
Deploy Model Package from AWS Marketplace

This sample notebook shows you how to deploy using Amazon SageMaker.

    Note: This is a reference notebook and it cannot run unless you make changes suggested in the notebook.

Pre-requisites:

    Note: This notebook contains elements which render correctly in Jupyter interface. Open this notebook from an Amazon SageMaker Notebook Instance or Amazon SageMaker Studio.
    Ensure that IAM role used has AmazonSageMakerFullAccess
    To deploy this ML model successfully, ensure that:
        Either your IAM role has these three permissions and you have authority to make AWS Marketplace subscriptions in the AWS account used:
            aws-marketplace:ViewSubscriptions
            aws-marketplace:Unsubscribe
            aws-marketplace:Subscribe
        or your AWS account has a subscription to . If so, skip step: Subscribe to the model package

Contents:

    Subscribe to the model package
    Create an endpoint and perform real-time inference
        Create an endpoint
        Create input payload
        Perform real-time inference
        Visualize output
        Delete the endpoint
    Perform batch inference
    Clean-up
        Delete the model
        Unsubscribe to the listing (optional))

Usage instructions

You can run this notebook one cell at a time (By using Shift+Enter for running a cell).
1. Subscribe to the model package

To subscribe to the model package:

    Open the model package listing page
    On the AWS Marketplace listing, click on the Continue to subscribe button.
    On the Subscribe to this software page, review and click on "Accept Offer" if you and your organization agrees with EULA, pricing, and support terms.
    Once you click on Continue to configuration button and then choose a region, you will see a Product Arn displayed. This is the model package ARN that you need to specify while creating a deployable model using Boto3. Copy the ARN corresponding to your region and specify the same in the following cell.

In [ ]:

model_package_arn = "<Customer to specify Model package ARN corresponding to their AWS region>"

In [ ]:

import base64
import json
import uuid
from sagemaker import ModelPackage
import sagemaker as sage
from sagemaker import get_execution_role
from sagemaker import ModelPackage
from urllib.parse import urlparse
import boto3
from IPython.display import Image
from PIL import Image as ImageEdit
import urllib.request
import numpy as np

In [ ]:

role = get_execution_role()

sagemaker_session = sage.Session()

bucket = sagemaker_session.default_bucket()
runtime = boto3.client("runtime.sagemaker")
bucket

2. Create an endpoint and perform real-time inference

If you want to understand how real-time inference with Amazon SageMaker works, see Documentation.

In [ ]:

model_name = "For Seller to update:<specify-model_or_endpoint-name>"

content_type = "For Seller to update:<specify_content_type_accepted_by_model>"

real_time_inference_instance_type = (
    "For Seller to update:<Update recommended_real-time_inference instance_type>"
)
batch_transform_inference_instance_type = (
    "For Seller to update:<Update recommended_batch_transform_job_inference instance_type>"
)

A. Create an endpoint
In [ ]:

# create a deployable model from the model package.
model = ModelPackage(
    role=role, model_package_arn=model_package_arn, sagemaker_session=sagemaker_session
)

# Deploy the model
predictor = model.deploy(1, real_time_inference_instance_type, endpoint_name=model_name)

Once endpoint has been created, you would be able to perform real-time inference.
B. Create input payload

In [ ]:


In [ ]:


C. Perform real-time inference

In [ ]:

!aws sagemaker-runtime invoke-endpoint \
    --endpoint-name $model_name \
    --body fileb://$file_name \
    --content-type $content_type \
    --region $sagemaker_session.boto_region_name \
    $output_file_name

D. Visualize output

In [ ]:


In [ ]:


E. Delete the endpoint

Now that you have successfully performed a real-time inference, you do not need the endpoint any more. You can terminate the endpoint to avoid being charged.
In [ ]:

model.sagemaker_session.delete_endpoint(model_name)
model.sagemaker_session.delete_endpoint_config(model_name)

3. Perform batch inference

In this section, you will perform batch inference using multiple input payloads together. If you are not familiar with batch transform, and want to learn more, see these links:

    How it works
    How to run a batch transform job

In [ ]:

# upload the batch-transform job input files to S3
transform_input_folder = "data/input/batch"
transform_input = sagemaker_session.upload_data(transform_input_folder, key_prefix=model_name)
print("Transform input uploaded to " + transform_input)

In [ ]:

# Run the batch-transform job
transformer = model.transformer(1, batch_transform_inference_instance_type)
transformer.transform(transform_input, content_type=content_type)
transformer.wait()

In [ ]:

# output is available on following path
transformer.output_path

4. Clean-up
A. Delete the model
In [ ]:

model.delete_model()

B. Unsubscribe to the listing (optional)

If you would like to unsubscribe to the model package, follow these steps. Before you cancel the subscription, ensure that you do not have any deployable model created from the model package or using the algorithm. Note - You can find this information by looking at the container name associated with the model.

Steps to unsubscribe to product from AWS Marketplace:

    Navigate to Machine Learning tab on Your Software subscriptions page
    Locate the listing that you want to cancel the subscription for, and then choose Cancel Subscription to cancel the subscription.

