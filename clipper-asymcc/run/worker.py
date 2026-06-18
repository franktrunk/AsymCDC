from concurrent.futures import ThreadPoolExecutor, wait, ALL_COMPLETED
from queue import Queue
import queue
from threading import Thread
from coder import Coder
import requests
import json
import time
import random
import socket
import numpy as np
import global_var
import torch
import torchvision.transforms as transforms
import io
from PIL import Image
import base64
import re
from util import in_dim, decode_in_dim
import pickle
from info import INFO


TASK_ENCODE_TYPE = 0
TASK_INFER_TYPE = 1
TIME_OUT = 3


transform = transforms.Compose([
    transforms.ToTensor()
])

transformPIL = transforms.Compose([
    transforms.ToPILImage()
])

def clipperRequest(addr, data):
    url = "http://{}:1337/pytorch-irevnet-app/predict".format(addr)
    # url = "http://localhost:1337/pytorch-irevnet-app/predict"
    req_json = json.dumps({
        'input': base64.b64encode(data).decode()
    })
    headers = {'Content-type': 'application/json'}
    return requests.post(url, headers=headers, data=req_json, timeout=30)

def encodeTask(input, coder):
    print("=========encodeTask start=========")
    # bytes --> tensor
    id_list, encode_list = input
    encode_tensor_list = []
    for data in encode_list:
        encode_tensor_list.append(transform(Image.open(io.BytesIO(data))))
    final_tensor=torch.stack(encode_tensor_list,0)
    
    # encode
    start = time.time()
    encode_tensor = coder.encode(final_tensor)
    end = time.time()
    print("encode costs: {} ms".format(float(end-start)*1000.0))
    INFO.add_encodetime(float(end - start) * 1000.0)
    
    # tensor --> bytes
    imgByte = io.BytesIO()
    transformPIL(encode_tensor).save(imgByte, format = 'JPEG')
    encode_data = imgByte.getvalue()
    
    id_list.append(-1)
    encode_list.append(encode_data)
    
    infer_task = Task(TASK_INFER_TYPE, (id_list, encode_list, float(end - start) * 1000.0))
    global_var.task_queue_put(infer_task)
    print("=========encodeTask end=========")

def inferTask(input, conf, clipperid):
    print("=========inferTask start=========")
    id_list, data_list, ecodeTime = input
    out_list = []
    outbij_list = []
    
    for i, data in enumerate(data_list):
        chosen = random.randint(0, conf.num_worker-1)
        if clipperid >=0:
            chosen = clipperid
        chosen_ip = conf.cfg['worker_ips'][chosen]
        start = time.time()
        try:
            resp = clipperRequest(chosen_ip, data)
        except Exception as e:
            print("clipperRequest failed: {}".format(e))
            raise
        end = time.time()
        print("Inference time: {} ms".format(float(end - start) * 1000.0))
        INFO.add_infertime(float(end - start) * 1000.0 + ecodeTime)
        # req_Thrds.append(global_var.clipper_req_pool.submit(clipperRequest, chosen_ip, data))
    
        # for i, _ in enumerate(data_list):
            # resp = req_Thrds[i].result()
        try:
            json_output = eval(resp.json()["output"])
        except Exception as e:
            print("resp.json() parse failed: {}".format(e))
            raise
        tensor_out = pickle.loads(json_output[0])  #[1, 10]
        tensor_outbij = pickle.loads(json_output[1])  #[1, 512, 8, 8]
        
        out_list.append(tensor_out[0])
        outbij_list.append(tensor_outbij[0].reshape(decode_in_dim[conf.cfg['dataset']])) #[8, 64, 64]
    
    failed = -1
    if random.random() < conf.cfg['fail_rate'] * conf.cfg['ec_k']:
        failed = random.randint(0, conf.cfg['ec_k'] - 1)  #[0,k-1]
        print("failed:",failed)
        zero_tensor = torch.zeros_like(out_list[failed])
        out_list[failed] = zero_tensor
        
        out_decode = torch.stack(out_list, dim=0)  # (k+1) * [10] --> [k, 10]
        outbij_decode = torch.cat(outbij_list[:failed] + outbij_list[failed+1:], dim=0) # k * [8, 64, 64] --> [8*k, 64, 64]
        global_var.decode_queue_put((id_list[failed], out_decode, outbij_decode))
    else:
        print("no fail")
        
    for i, data in enumerate(out_list[:-1]):
        if failed < 0 or i != failed:
            global_var.resp_queue_put((id_list[i], data.numpy().argmax()))
            print("resp:",id_list[i])
    
    print("=========inferTask end=========")


class Task:
    def __init__(self, type, input) -> None:
        self.type = type
        self.input = input


class Worker:
    def __init__(self, conf, coder, id) -> None:
        self.conf = conf
        self.ec_k = self.conf.cfg['ec_k']
        self.coder = coder
        self.thrd_pool = ThreadPoolExecutor(max_workers=10)
        self.clipperid = id

    def process_task(self):
        while(True):
            task_list = []
            try:
                task = global_var.task_queue.get(block=True, timeout=TIME_OUT)
            except queue.Empty:
                if global_var.FLAG_STOP_THREAD:
                    print("Worker end")
                    break
                else:
                    continue
            
            if(task.type == TASK_ENCODE_TYPE):
                enc_task = self.thrd_pool.submit(encodeTask, task.input, self.coder)
                task_list.append(enc_task)
            elif(task.type == TASK_INFER_TYPE):
                inf_task = self.thrd_pool.submit(inferTask, task.input, self.conf, self.clipperid)
                task_list.append(inf_task)
            wait(task_list, return_when=ALL_COMPLETED)

class Repairer:
    def __init__(self, ec_k, coder) -> None:
        self.ec_k = ec_k
        self.coder = coder

    def decode_task(self):
        while(True):
            try:
                input = global_var.decode_queue.get(block=True, timeout=TIME_OUT)
            except queue.Empty:
                if global_var.FLAG_STOP_THREAD:
                    print("Repairer end")
                    break
                else:
                    continue
            image_id, out_decode, outbij_decode = input

            print("=========repairTask start=========")
            # decode
            start = time.time()
            resp_tensor = self.coder.decode((out_decode, outbij_decode))
            end = time.time()
            print("decode costs: {} ms".format(float(end-start)*1000.0))
            INFO.add_decodetime(float(end - start) * 1000.0)
    
            global_var.resp_queue_put((image_id, resp_tensor.numpy().argmax()))
            print("=========repairTask end=========")