"""测试 inferTask 的独立脚本"""
from worker import inferTask, clipperRequest
import torch
import io
from PIL import Image
import numpy as np
import json
import argparse
import os
import global_var

def create_test_image():
    """创建一个测试图像的 bytes 数据"""
    # 创建一个简单的 RGB 图像 (CIFAR10 是 32x32)
    img = Image.new('RGB', (32, 32), color=(255, 0, 0))
    img_byte = io.BytesIO()
    img.save(img_byte, format='JPEG')
    return img_byte.getvalue()

def test_clipper_request():
    """先测试 clipperRequest 是否能连通"""
    print("\n===== 测试 1: clipperRequest 连通性 =====")
    from config import Config
    conf = Config('/home/wyt/AsymCDC/clipper-asymcc/run/config/test.json')
    ip = conf.cfg['worker_ips'][0]
    
    try:
        print(f"尝试连接 {ip}:1337 ...")
        resp = clipperRequest(ip, create_test_image())
        print(f"响应状态码: {resp.status_code}")
        print(f"响应内容: {resp.json()}")
        return True
    except Exception as e:
        print(f"连接失败: {e}")
        return False

def test_inferTask():
    """测试 inferTask 完整流程"""
    print("\n===== 测试 2: inferTask 完整流程 =====")
    from config import Config
    
    conf = Config('/home/wyt/AsymCDC/clipper-asymcc/run/config/test.json')
    global_var.queue_init(10)  # 初始化队列
    
    # 构造输入: (id_list, data_list, encodeTime)
    id_list = [0]
    data_list = [create_test_image()]
    ecodeTime = 10.0
    input_data = (id_list, data_list, ecodeTime)
    
    print(f"输入数据: id_list={id_list}, data_list长度={len(data_list)}")
    
    try:
        inferTask(input_data, conf, clipperid=0)
        print("===== inferTask 执行成功 =====")
    except Exception as e:
        print(f"===== inferTask 执行出错: {e} =====")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("========================================")
    print("         inferTask 独立测试脚本")
    print("========================================")
    
    # 先测试网络连接
    if test_clipper_request():
        # 再测试完整流程
        test_inferTask()
    else:
        print("\nClipper 服务不可用，请检查:")
        print("1. Clipper 服务是否启动")
        print("2. IP 地址是否正确")
        print("3. 防火墙设置")
