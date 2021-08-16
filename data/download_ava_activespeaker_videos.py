import subprocess
import glob
import os


if __name__ == '__main__':
    file = open('/usr/home/kop/active-speakers-context/data/ava_activespekaer_test_list.csv', 'r')
    Lines = file.readlines()
 
    for line in Lines:
        command = ("wget https://s3.amazonaws.com/ava-dataset/test/%s" % (line))
        subprocess.call(command, shell=True, stdout=None)

