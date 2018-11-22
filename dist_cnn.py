import tensorflow as tf

'''
declare IP address and port numbers 
'''

IP_ADDRESS1='192.168.1.136'
PORT1='2222'

IP_ADDRESS2='192.168.1.121'
PORT2='2224'

def clusters(task_idx):

    # Define cluster
    cluster_spec = tf.train.ClusterSpec({'worker' : [(IP_ADDRESS1 + ":" + PORT1), (IP_ADDRESS2 + ":" + PORT2)]})

    #Task index corresponds to IP addresses of different machines or if you are using same machine then different servers on same machine would correspond to different task index
    # For example, if you are running this notebook on (IP_ADDRESS2 + ":" + PORT2), task_idx=1 because it is
    # responsible for the second task of the job:worker based on how you defined cluster_spec above
    # Define server for specific machine
    #task_idx, This will be different for each non-chief machine you run this script on, For eg. task_idx=0 is for chief machine and task_idx=1, 2, 3 etc. are for workers


    server = tf.train.Server(cluster_spec, job_name='worker', task_index=task_idx)

    # Server will run as long as the script is running
    server.join()

if __name__=='__main__':

    # task index = 0 initializes clusters on chief machine
    clusters(task_idx=0)