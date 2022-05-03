Clean up the data: In order to offload unnecessary operations from the Spark job, we decided to clean up the data before uploading it to the S3 bucket. To do so we wrote a script to remove the rows from the data that contained null values for the abstract.
  In order to obtain the cleaned data place this metadata.csv file in the same directory as the cleanup.py file and run: python3 cleanup.py
  A new file called cleaned.csv will be created that retains only the rows that have both the title and abstract.

Create an S3 bucket:
  Name the bucket, keep the other settings as their default values, and click “Create Bucket”
  Click “Add a file”, and select the cleaned.csv file to upload it to the S3 bucket

Create a security key:
  Navigate to EC2 and create a security key by going to Network and Security -> Key Pairs
  Select “Create Key Pair”, name it, and keep the default settings.

Create an EMR cluster that supports Spark:
  Click “Go to advanced options”
  Under the software configuration settings ensure that Hadoop, Spark, and JupyterEnterpriseGateway are checked.
  Click next to move onto the hardware configuration. Configure 1 master instance, and 7 core instances.
  Note: we were going to choose 8 core instances but were restricted to 32 vCPUs by the permissions given to the lab account.
  Click next to move onto general cluster settings. Name the cluster.
  Click next to move onto Security settings. Select the EC2 key pair that was made in step 3 as the key pair, and keep the other settings as their default values.
  Click create cluster, and wait until the status message at the top changes from “Starting” to “Waiting”. While the instance is starting, you can move onto the next step.

Attach a jupyter notebook instance to the EMR:
  Select “Notebooks” from the left menu. and create a notebook. Give the notebook a name and select the cluster that you created as the cluster for the notebook. All of the other settings will remain the same. Ensure that the “AWS service role” is set to LabRole. Click “Create Notebook”.
  Wait for the notebook and cluster to start.
  Once the notebook has started, open the notebook and click “Open in JupyterLab”
  Make a new file using a PySpark kernel.
  Create two cells for the Jupyter notebook: the first will contain configuration details (which can be found in configuration.txt), and the second will contain the code to perform the k-means clustering (this code is in pyspark-kmeans.py).
  Note: navigate to your S3 instance and the cleaned.csv file that was uploaded. Copy the object URI and paste it into the .load() call at the top of the file
  Click run (Note: if rerunning this code make sure to restart the kernel).
  Note: if there is a 400 error, just restart the kernel and run it again.
  After some time the results of clustering will be shown:
    The number of documents in each cluster
    The top 10 words and their frequencies in each document
    The top 3 words in the entire dataset
