# firehrtindata

{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Deploy  <font color='red'>For Seller to update: Title_of_your_ML Model </font> Model Package from AWS Marketplace \n",
    "\n",
    "\n",
    "## <font color='red'> For Seller to update: Add  overview of the ML Model here </font>\n",
    "\n",
    "This sample notebook shows you how to deploy <font color='red'> For Seller to update:[Title_of_your_ML Model](Provide link to your marketplace listing of your product)</font> using Amazon SageMaker.\n",
    "\n",
    "> **Note**: This is a reference notebook and it cannot run unless you make changes suggested in the notebook.\n",
    "\n",
    "#### Pre-requisites:\n",
    "1. **Note**: This notebook contains elements which render correctly in Jupyter interface. Open this notebook from an Amazon SageMaker Notebook Instance or Amazon SageMaker Studio.\n",
    "1. Ensure that IAM role used has **AmazonSageMakerFullAccess**\n",
    "1. To deploy this ML model successfully, ensure that:\n",
    "    1. Either your IAM role has these three permissions and you have authority to make AWS Marketplace subscriptions in the AWS account used: \n",
    "        1. **aws-marketplace:ViewSubscriptions**\n",
    "        1. **aws-marketplace:Unsubscribe**\n",
    "        1. **aws-marketplace:Subscribe**  \n",
    "    2. or your AWS account has a subscription to <font color='red'> For Seller to update:[Title_of_your_ML Model](Provide link to your marketplace listing of your product)</font>. If so, skip step: [Subscribe to the model package](#1.-Subscribe-to-the-model-package)\n",
    "\n",
    "#### Contents:\n",
    "1. [Subscribe to the model package](#1.-Subscribe-to-the-model-package)\n",
    "2. [Create an endpoint and perform real-time inference](#2.-Create-an-endpoint-and-perform-real-time-inference)\n",
    "   1. [Create an endpoint](#A.-Create-an-endpoint)\n",
    "   2. [Create input payload](#B.-Create-input-payload)\n",
    "   3. [Perform real-time inference](#C.-Perform-real-time-inference)\n",
    "   4. [Visualize output](#D.-Visualize-output)\n",
    "   5. [Delete the endpoint](#E.-Delete-the-endpoint)\n",
    "3. [Perform batch inference](#3.-Perform-batch-inference) \n",
    "4. [Clean-up](#4.-Clean-up)\n",
    "    1. [Delete the model](#A.-Delete-the-model)\n",
    "    2. [Unsubscribe to the listing (optional)](#B.-Unsubscribe-to-the-listing-(optional))\n",
    "    \n",
    "\n",
    "#### Usage instructions\n",
    "You can run this notebook one cell at a time (By using Shift+Enter for running a cell)."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 1. Subscribe to the model package"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "To subscribe to the model package:\n",
    "1. Open the model package listing page <font color='red'> For Seller to update:[Title_of_your_product](Provide link to your marketplace listing of your product).</font>\n",
    "1. On the AWS Marketplace listing, click on the **Continue to subscribe** button.\n",
    "1. On the **Subscribe to this software** page, review and click on **\"Accept Offer\"** if you and your organization agrees with EULA, pricing, and support terms. \n",
    "1. Once you click on **Continue to configuration button** and then choose a **region**, you will see a **Product Arn** displayed. This is the model package ARN that you need to specify while creating a deployable model using Boto3. Copy the ARN corresponding to your region and specify the same in the following cell."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "model_package_arn = \"<Customer to specify Model package ARN corresponding to their AWS region>\""
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "<font color='red'> For Seller to update: Add all necessary imports in following cell, \n",
    "If you need specific packages to be installed, # try to provide them in this section, in a separate cell. </font>"
   ]
