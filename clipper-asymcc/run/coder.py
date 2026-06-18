import config
import torch
from encoder import LinearEncoder, ConvEncoder, ConcatEncoder, MLPEncoder
from decoder import LinearDecoder, DistilledDecoder, MLPDecoder
from util import in_dim, out_dim, decode_in_dim
from models.iRevNet import iRevNet16x64, iRevNet48x64

coder_models = {
    "iRevNet16x64": iRevNet16x64,
    "iRevNet48x64": iRevNet48x64
}

class Coder:
    def __init__(self, conf) -> None:
        self.conf = conf
        self.ec_k = conf.cfg['ec_k']
        
        # construct encoder
        if conf.cfg['encoder'] == 'linear':
            self.encoder = LinearEncoder(ec_k=self.ec_k, in_dim=in_dim[conf.cfg['dataset']])
        elif conf.cfg['encoder'] == 'conv':
            self.encoder = ConvEncoder(ec_k=self.ec_k, in_dim=in_dim[conf.cfg['dataset']])
        elif conf.cfg['encoder'] == 'concat_crop':
            self.encoder = ConcatEncoder(ec_k=self.ec_k, in_dim=in_dim[conf.cfg['dataset']], type = "crop")
        elif conf.cfg['encoder'] == 'concat_resize':
            self.encoder = ConcatEncoder(ec_k=self.ec_k, in_dim=in_dim[conf.cfg['dataset']], type = "resize")
        elif conf.cfg['encoder'] == 'mlp':
            self.encoder = MLPEncoder(ec_k=self.ec_k, in_dim=in_dim[conf.cfg['dataset']])
        else:
            raise NotImplementedError("Not implemented Encoder Type: " + conf.cfg['encoder'])
        
        # construct decoder
        if conf.cfg['decoder'] == 'linear':
            self.decoder = LinearDecoder(ec_k = self.ec_k)
        elif conf.cfg['decoder'] == 'mlp':
            self.decoder = MLPDecoder(ec_k = self.ec_k, in_dim=out_dim[conf.cfg['dataset']])
        elif conf.cfg['decoder'] == 'distill':
            path = conf.cfg['decoder_checkpoint']
            print("=> loading distill model '{}'".format(path))
            model = coder_models[conf.cfg['decoder_model']]()
            checkpoint = torch.load(path)
            # checkpoint may be a dict with 'model' key containing a model object or state_dict
            model_obj = checkpoint.get('model', checkpoint)
            if isinstance(model_obj, torch.nn.Module):
                state_dict = model_obj.state_dict()
            else:
                state_dict = model_obj
            model.load_state_dict(state_dict)
            if torch.cuda.is_available():
                model = model.cuda()
            print("=> loaded distill model '{}'".format(path))
            
            self.decoder = DistilledDecoder(ec_k = self.ec_k, model = model)
        else:
            raise NotImplementedError("Not implemented Decoder type: " + conf.cfg['decoder'])
    
    def encode(self, input):
        return self.encoder(input)
    
    def decode(self, input):
        return self.decoder(input)