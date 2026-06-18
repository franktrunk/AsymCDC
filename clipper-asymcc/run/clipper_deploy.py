# from models.iRevNet import iRevNet
from clipper_admin import ClipperConnection, DockerContainerManager
from clipper_admin.exceptions import ClipperException
import io
from PIL import Image
from torch.autograd import Variable
import torchvision.transforms as transforms
import clipper_admin.deployers.pytorch as pytorch_deployer
from torchvision.models.resnet import resnet50
import torch
import os
from config import Config
import subprocess
import argparse
import pickle
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F


def split(x):
    n = int(x.size()[1]/2)
    x1 = x[:, :n, :, :].contiguous()
    x2 = x[:, n:, :, :].contiguous()
    return x1, x2


def merge(x1, x2):
    return torch.cat((x1, x2), 1)


class injective_pad(nn.Module):
    def __init__(self, pad_size):
        super(injective_pad, self).__init__()
        self.pad_size = pad_size
        self.pad = nn.ZeroPad2d((0, 0, 0, pad_size))

    def forward(self, x):
        x = x.permute(0, 2, 1, 3)
        x = self.pad(x)
        return x.permute(0, 2, 1, 3)

    def inverse(self, x):
        return x[:, :x.size(1) - self.pad_size, :, :]


class psi(nn.Module):
    def __init__(self, block_size):
        super(psi, self).__init__()
        self.block_size = block_size
        self.block_size_sq = block_size*block_size

    def inverse(self, input):
        bl, bl_sq = self.block_size, self.block_size_sq
        bs, new_d, h, w = input.shape[0], input.shape[1] // bl_sq, input.shape[2], input.shape[3]
        return input.reshape(bs, bl, bl, new_d, h, w).permute(0, 3, 4, 1, 5, 2).reshape(bs, new_d, h * bl, w * bl)

    def forward(self, input):
        bl, bl_sq = self.block_size, self.block_size_sq
        bs, d, new_h, new_w = input.shape[0], input.shape[1], input.shape[2] // bl, input.shape[3] // bl
        return input.reshape(bs, d, new_h, bl, new_w, bl).permute(0, 3, 5, 1, 2, 4).reshape(bs, d * bl_sq, new_h, new_w)


class irevnet_block(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1, first=False, dropout_rate=0.,
                 affineBN=True, mult=4):
        """ buid invertible bottleneck block """
        super(irevnet_block, self).__init__()
        self.first = first
        self.pad = 2 * out_ch - in_ch
        self.stride = stride
        self.inj_pad = injective_pad(self.pad)
        self.psi = psi(stride)
        if self.pad != 0 and stride == 1:
            in_ch = out_ch * 2
            print('')
            print('| Injective iRevNet |')
            print('')
        layers = []
        if not first:
            layers.append(nn.BatchNorm2d(in_ch//2, affine=affineBN))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(in_ch//2, int(out_ch//mult), kernel_size=3,
                      stride=stride, padding=1, bias=False))
        layers.append(nn.BatchNorm2d(int(out_ch//mult), affine=affineBN))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(int(out_ch//mult), int(out_ch//mult),
                      kernel_size=3, padding=1, bias=False))
        layers.append(nn.Dropout(p=dropout_rate))
        layers.append(nn.BatchNorm2d(int(out_ch//mult), affine=affineBN))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(int(out_ch//mult), out_ch, kernel_size=3,
                      padding=1, bias=False))
        self.bottleneck_block = nn.Sequential(*layers)

    def forward(self, x):
        """ bijective or injective block forward """
        if self.pad != 0 and self.stride == 1:
            x = merge(x[0], x[1])
            x = self.inj_pad.forward(x)
            x1, x2 = split(x)
            x = (x1, x2)
        x1 = x[0]
        x2 = x[1]
        Fx2 = self.bottleneck_block(x2)
        if self.stride == 2:
            x1 = self.psi.forward(x1)
            x2 = self.psi.forward(x2)
        y1 = Fx2 + x1
        return (x2, y1)

    def inverse(self, x):
        """ bijective or injecitve block inverse """
        x2, y1 = x[0], x[1]
        if self.stride == 2:
            x2 = self.psi.inverse(x2)
        Fx2 = - self.bottleneck_block(x2)
        x1 = Fx2 + y1
        if self.stride == 2:
            x1 = self.psi.inverse(x1)
        if self.pad != 0 and self.stride == 1:
            x = merge(x1, x2)
            x = self.inj_pad.inverse(x)
            x1, x2 = split(x)
            x = (x1, x2)
        else:
            x = (x1, x2)
        return x


class iRevNet(nn.Module):
    def __init__(self, nBlocks, nStrides, nClasses, nChannels=None, init_ds=2,
                 dropout_rate=0., affineBN=True, in_shape=None, mult=4):
        super(iRevNet, self).__init__()
        self.ds = in_shape[2]//2**(nStrides.count(2)+init_ds//2)
        self.init_ds = init_ds
        self.in_ch = in_shape[0] * 2**self.init_ds
        self.nBlocks = nBlocks
        self.first = True

        print('')
        print(' == Building iRevNet %d == ' % (sum(nBlocks) * 3 + 1))
        if not nChannels:
            nChannels = [self.in_ch//2, self.in_ch//2 * 4,
                         self.in_ch//2 * 4**2, self.in_ch//2 * 4**3]

        self.init_psi = psi(self.init_ds)
        self.stack = self.irevnet_stack(irevnet_block, nChannels, nBlocks,
                                        nStrides, dropout_rate=dropout_rate,
                                        affineBN=affineBN, in_ch=self.in_ch,
                                        mult=mult)
        self.bn1 = nn.BatchNorm2d(nChannels[-1]*2, momentum=0.9)
        self.linear = nn.Linear(nChannels[-1]*2, nClasses)

    def irevnet_stack(self, _block, nChannels, nBlocks, nStrides, dropout_rate,
                      affineBN, in_ch, mult):
        """ Create stack of irevnet blocks """
        block_list = nn.ModuleList()
        strides = []
        channels = []
        for channel, depth, stride in zip(nChannels, nBlocks, nStrides):
            strides = strides + ([stride] + [1]*(depth-1))
            channels = channels + ([channel]*depth)
        for channel, stride in zip(channels, strides):
            block_list.append(_block(in_ch, channel, stride,
                                     first=self.first,
                                     dropout_rate=dropout_rate,
                                     affineBN=affineBN, mult=mult))
            in_ch = 2 * channel
            self.first = False
        return block_list

    def forward(self, x):
        """ irevnet forward """
        n = self.in_ch//2
        if self.init_ds != 0:
            x = self.init_psi.forward(x)
        out = (x[:, :n, :, :], x[:, n:, :, :])
        for block in self.stack:
            out = block.forward(out)
        out_bij = merge(out[0], out[1])
        out = F.relu(self.bn1(out_bij))
        out = F.avg_pool2d(out, self.ds)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out, out_bij

    def inverse(self, out_bij):
        """ irevnet inverse """
        out = split(out_bij)
        for i in range(len(self.stack)):
            out = self.stack[-1-i].inverse(out)
        out = merge(out[0],out[1])
        if self.init_ds != 0:
            x = self.init_psi.inverse(out)
        else:
            x = out
        return x




min_img_size = 224

def predict(model, inputs):
    def _predict_one(one_input_arr):
        try:
            img = Image.open(io.BytesIO(one_input_arr))
            if img.mode != "RGB":
                img = img.convert("RGB")
            # transform_pipeline = transforms.Compose([transforms.Resize(min_img_size),
            #                             transforms.ToTensor(),
            #                             transforms.Normalize(mean=[0.485, 0.456, 0.406],
            #                                                 std=[0.229, 0.224, 0.225])])
            transform_pipeline = transforms.Compose([transforms.ToTensor()])
            img = transform_pipeline(img)
            
            # if torch.cuda.is_available():
            #     img = img.cuda()
            img = img.unsqueeze(0)
            img = Variable(img)
            out, out_bij = model(img)
            
            return [pickle.dumps(out.cpu().data),pickle.dumps(out_bij.cpu().data)]

        except Exception as e:
            print(e)
            return e
        
    return [_predict_one(i) for i in inputs]

def load_irevnet_model(model_path):
    if os.path.isfile(model_path):
        print("=> loading checkpoint '{}'".format(model_path))
        model = iRevNet(nBlocks=[18, 18, 18], nStrides=[1, 2, 2],
                    nChannels=[16, 64, 256], nClasses=10,
                    init_ds=0, dropout_rate=0.1, affineBN=True,
                    in_shape=[3, 32, 32], mult=4
        )
    
        # 加载检查点（绕过 DataParallel 的 module. 前缀问题）
        # 1. 如果检查点包含元数据字典（如 {"model": state_dict/model_obj, "acc": ..., "epoch": ...}），提取 state_dict
        # 2. 如果 checkpoint['model'] 是一个模型对象（而非 state_dict），调用 .state_dict() 获取权重
        # 3. 去掉 "module." 前缀以适配 DataParallel 保存的权重
        checkpoint = torch.load(model_path, map_location='cpu')
        if isinstance(checkpoint, dict) and 'model' in checkpoint:
            model_data = checkpoint['model']
        else:
            model_data = checkpoint
        
        # 如果 model_data 是模型对象（DataParallel），获取其 state_dict
        if hasattr(model_data, 'state_dict'):
            state_dict = model_data.state_dict()
        else:
            state_dict = model_data
        
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('module.'):
                new_state_dict[k[7:]] = v  # 去掉 "module." 前缀
            else:
                new_state_dict[k] = v
        
        model.load_state_dict(new_state_dict)
        
        # 修复旧版本 PyTorch 模型缺少的 padding_mode 属性
        import torch.nn as nn
        for module in model.modules():
            if isinstance(module, nn.Conv2d):
                if not hasattr(module, 'padding_mode'):
                    module.padding_mode = 'zeros'
        
        print("=> loaded checkpoint '{}'".format(model_path))
        return model
    else:
        print("=> no checkpoint found at '{}'".format(model_path))
        return None

# def load_irevnet_model(model_path):
#     if os.path.isfile(model_path):
#         print("=> loading checkpoint '{}'".format(model_path))
#         model = iRevNet(nBlocks=[18, 18, 18], nStrides=[1, 2, 2],
#                     nChannels=[16, 64, 256], nClasses=10,
#                     init_ds=0, dropout_rate=0.1, affineBN=True,
#                     in_shape=[3, 32, 32], mult=4
#         )
#     
#         model = torch.nn.DataParallel(model) 
#     
#         model.load_state_dict(torch.load(model_path))
#         print("=> loaded checkpoint '{}'".format(model_path))
#         return model
#     else:
#         print("=> no checkpoint found at '{}'".format(model_path))
#         return None


class ClipperDeployer:
    def __init__(self, conf) -> None:
        self.conf = conf

    def deploy(self):
        model = load_irevnet_model(self.conf.cfg['model_checkpoint'])
        # model = resnet50(pretrained=True)

        try:
            clipper_conn = ClipperConnection(DockerContainerManager())
            clipper_conn.start_clipper(cache_size=1)  # Disable PredictionCache
        except ClipperException:
            clipper_conn.connect()
            clipper_conn.stop_all()  # stop_all() already cleans up Clipper containers
            subprocess.call([r'''docker ps -a --format "{{.ID}} {{.Names}}" | \
                             grep -E "query_frontend|mgmt_frontend|frontend_exporter|sum-model|redis-|metric_frontend|prometheus" | \
                             awk '{print $1}' | xargs -r docker rm -f
                             '''], shell=True)            
            clipper_conn = ClipperConnection(DockerContainerManager())
            clipper_conn.start_clipper(cache_size=1)  # Disable PredictionCache

        app_name = 'pytorch-irevnet-app'
        model_name = 'pytorch-irevnet-model'

        pytorch_deployer.deploy_pytorch_model(clipper_conn=clipper_conn,
                                            name=model_name,
                                            version='1',
                                            input_type='bytes',
                                            func=predict,
                                            pytorch_model=model,
                                            num_replicas=1,
                                            batch_size=1,  # Disable adaptive batching policy
                                            pkgs_to_install=['pillow'])

        clipper_conn.register_application(name=app_name,
                                  input_type="bytes",
                                  default_output="-1.0",
                                  slo_micros=10000000)  # 10s

        clipper_conn.link_model_to_app(app_name=app_name,
                               model_name=model_name)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='i-NeDD clipper_deploy arguments')
    parser.add_argument("--conf", type=str, default="./config/simple.json", help="Path of the config file")
    args = parser.parse_args()
    
    conf = Config(args.conf)
    deployer = ClipperDeployer(conf)
    deployer.deploy()