from config import Config
from concurrent import futures
from threading import Thread
from coder import Coder
import argparse
import os
from concurrent.futures import ThreadPoolExecutor, wait, ALL_COMPLETED
from queue import Queue
import queue
import requests
import json
import time
import random
import numpy as np
import torch
torch.backends.cudnn.enabled = False
import io
from PIL import Image
import base64
import pickle
from util import in_dim, decode_in_dim
import global_var
from worker import Worker, Repairer, Task, TASK_ENCODE_TYPE
from info import ImageInfo, TimeSeries, INFO

RESP_TIME_OUT = 3


def SendTask(path_list, k, queryGroupRate):
    timeSeq = TimeSeries(queryGroupRate)
    for i in range(global_var.resp_len // k):
        id_list = []
        encode_group = []
        for j,fpath in enumerate(path_list[i*k : i*k + k]):
            img = ImageInfo(i*k + j, fpath)
            id_list.append(img.id)
            encode_group.append(img.byte)
        time.sleep(timeSeq(i % queryGroupRate))
        encode_task = Task(TASK_ENCODE_TYPE,(id_list,encode_group))
        global_var.task_queue.put(encode_task)


def WriteResp(outpath):
    print("=====resp_task start=========")
    len = global_var.resp_len
    resp_str = ""
    for i in range(len):
        id,val = global_var.resp_queue_get()
        resp_str += str(id)+' '+str(val)+ '\n'
    print(resp_str)
    with open(outpath, 'w') as f:
        f.write(resp_str)
    print("=====resp_task end=========")


def StartWorker(conf, coder):
    workerThrds = []
    for i in range(conf.cfg['workerThrd_num']):
        worker = Worker(conf, coder, i)
        workerThrds.append(Thread(target=worker.process_task))
    for i, _ in enumerate(workerThrds):
        print("=========Worker start==========")
        workerThrds[i].start()
        
    repairer = Repairer(conf, coder)
    repairThrd = Thread(target=repairer.decode_task)
    print("=========Repairer start==========")
    repairThrd.start()


def StopWorker():
    global_var.FLAG_STOP_THREAD = True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='AsymCC clipper_deploy arguments')
    parser.add_argument("--conf", type=str, default="/root/AsymCC/clipper-asymcc/run/config/simple.json", help="Path of the config file")
    parser.add_argument("--path", type=str, default="/root/datatest/cifar10/test", help="Path of the input file")
    
    args = parser.parse_args()
    path_list = [os.path.join(args.path, file) for file in os.listdir(args.path)]
    conf = Config(args.conf)
    coder = Coder(conf)
    global_var.queue_init(conf.cfg["input_num"]//conf.cfg['ec_k']*conf.cfg['ec_k'])
    
    start = time.time()
    StartWorker(conf, coder)
            
    SendTask(path_list, conf.cfg['ec_k'], conf.cfg['query_rate'])
    
    while(True):
        if global_var.task_finished():
            StopWorker()
            break
        time.sleep(RESP_TIME_OUT)
    
    WriteResp(conf.cfg['output_path'])
    
    end = time.time()
    print("total costs: {} ms".format(float(end-start)*1000.0))
    INFO.set_totaltime(float(end - start) * 1000.0)
    INFO.output()
    with open('./output_total_time.txt', 'a') as f:
        f.write("total time: {} ms\n".format(INFO.totalTime))