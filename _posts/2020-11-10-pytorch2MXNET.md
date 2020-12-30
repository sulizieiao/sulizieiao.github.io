---
layout: post
title: pytorch模型 转 MXNET 模型
date: 2020-11-10 
tag: 技术

---

### pytorch模型 转 MXNET 模型



#### step1： 按照pytorch的model.py写对应的MXNET的model.py （model_mxnet.py)

​	**梳理网络结构，写成symbol格式：**

```python
## pytorch
class XXNet(nn.Module):
    def __init__(self)
        super(XXNet, self).__init__()
        self.conv1 = nn.Conv2d(3,64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=None, padding=0)
   	def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.maxpool(x)
        return x

## mxnet    会自动根据每层输入推测参数大小
def XXNet():
    input = mx.symbol.Variable(name='input')
    conv1 = mx.sym.Convolution(data=input, num_filter=64, kernel=(3,3), stride=(1,1), pad=(1,1), num_group=1, no_bias=True, name='layer_conv1')  ## no_bias默认0
    bn1 = mx.sym.BatchNorm(data=conv1, fix_gamma=False, eps=bn_eps, momentum=bn_mom, name='layer_bn1')
    pooling1 = mx.sym.Pooling(data=bn1, kernel=(2,2), stride=(2,2), pool_type='max')
    return pooling1
```

​	:boom: **注意网络默认参数差异：** 一定要仔细看默认参数！

```python
## BN默认参数差异： 
bn_eps=0.00001 ## pytorch
bn_mom = 0.1  ##pytorch
bn_eps = 0.001 ## mxnet
bn_mom = 0.899999976 ##mxnet
fix_gamma = True ## mxnet 默认固定BN层的gamma(weight)=1 ！！如果要用训练好的BN参数，一定要设fix_gamma=False

## Pooling默认差异：
pool_stride = kernel_size ## pytorch 默认pooling的stride等于pooling的核大小
pool_stride = 1 ## mxnet 默认pooling的stride等于1

```





#### step2: 转换模型参数，生成模型结构和对应参数文件 (model_mxnet-0000.params, model_mxnet-symbol.json)

​	**遍历每层参数，使用pth的参数，给mxnet的模型参数赋值**

```python
pth_model_path = 'xxx/xxx_pth'  ## pth模型名称为 xxx_pth, eg: ResNet18_pth
mxnet_save_path = pth_model_path.replace('pth', 'mxnet')
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

## load pth model
pth_model = ResNet().to(device)
check_point = torch.load(pth_model_path, map_location=device)
pth_model.load_state_dict(check_point['state_dict'], strict=True)
pth_model.eval()
pth_dict = OrderedDict()
for k,v in pth_model.state_dict().items():
    pth_dict[k] = v.numpy()
    
ctx = mx.cpu()     ## if use GPU: ctx = mx.gpu(device_num)
mxnet_model = ResNet_mxnet()
arg_names = mxnet_model.list_arguments()
aux_names = mxnet_model.list_auxiliary_states()
## eg:data0_shape=(1,3,112,112), data1_shape=(1,20)
arg_shapes, _, aux_shapes = mxnet_model.infer_shape(data0=data0_shape, data1=data1_shape) 
arg_shape_dict = dict(zip(arg_names, arg_shapes)) ## 用以验证 pth 和 mxnet模型的各层参数shape是否一致
aux_shape_dict = dict(zip(aux_name, aux_shapes))
arg_params = {}
aux_params = {}

pth_keys = list(pth_dict.keys())

with open('layer_contrast.txt', 'w+') as outfile:
    n = 0
    n_pth = 0
    while n < len(arg_names):
        layer_name = arg_names[n]
        if layer_name in input_name_ls:  ## input_name_ls = ['data0','data1'] 如果是输入数据层，则跳过
            n += 1
            continue
        print(layer_name, pth_keys[n_pth])
        print(arg_sahpe_dict[layer_name], pth_dict[pth_keys[n_pth]].shape)  ## 验证 pth 和 mxnet 对应层参数shape 是否一致
        
        if '_beta' in layer_name: ## 对于bn层，bn_beta赋值之后，进入aux_params,给对应的 bn_moving_mean, bn_moving_var赋值
            arg_params[layer_name] = mx.nd.array(pth_dict[pth_keys[n_pth]])
            aux_params[layer_name.replace('beta', 'moving_mean')] = mx.nd.array(pth_dict[pth_keys[n_pth + 1]])
            aux_params[layer_name.replace('beta', 'moving_var')] = mx.nd.array(pth_dict[pth_keys[n_pth + 2]])
            outfile.write(layer_name+' '+pth_keys[n_pth]+'\n')
            outfile.write(layer_name.replace('beta', 'moving_mean')+' '+pth_keys[n_pth+1]+'\n')
            outfile.write(layer_name.replace('beta', 'moving_var')+' '+pth_keys[n_pth+2]+'\n')
            n_pth += 4
            n += 1
            continue
        if 'bottleneck' in layer_name: ## 固定 gamma, beta的bn层 （pth_keys里没有对应的gamma,beta），同时在网络的BN设 fix_gamma=True
            arg_params['bottleneck_gamma'] = mx.nd.array(np.ones((1,512),dtype=np.float32).flatten())
            aux_params['bottleneck_beta'] =  = mx.nd.array(np.zeros((1,512),dtype=np.float32).flatten())
            aux_params['bottleneck_moving_mean'] = mx.nd.array(pth_dict[pth_keys[n_pth]])
            aux_params['bottleneck_moving_var'] = mx.nd.array(pth_dict[pth_keys[n_pth + 1]])
            outfile.write('bottleneck_gamma'+' '+'1'+'\n')
            outfile.write('bottleneck_gamma'+' '+'0'+'\n')
            outfile.write('bottleneck_moving_mean'+' '+pth_keys[n_pth]+'\n')
            outfile.write('bottleneck_moving_var'+' '+pth_keys[n_pth+1]+'\n')
            n_pth += 3
            n += 2
            continue
        arg_params[layer_name] = mx.nd.array(pth_dic[pth_keys[n_pth]])
        outfile.write(layer_name+' '+pth_keys[n_pth])
        n_pth += 1
        n += 1
   save_mxnet_model(mxnet_model, arg_params, aux_params, output_prefix=mxnet_save_path)  ## 把参数传入mxnet model，并存成 .params, .json文件

def save_mxnet_model(sym, arg_params, aux_params, output_prefix, data_names=['data1', 'data2']):
    model = mx.mod.Module(symbol=sym, data_names=data_names, label_names=None)
    model.bind(data_shapes=[('data1', (data1_shapes)), ('data2', (data2_shapes))])
    model.init_params(arg_params=arg_params, aux_params=aux_params)
    model.save_checkpoint(output_prefix, 0)   ## 0 表示第0个epoch
```



​	:boom: **多输入问题：** 

```python
import mxnet as mx
## load model
sym = mx.symbol.load('model-symbol.json')
mod = mx.mod.Module(symbol=sym, data_names=['data0','data1'])   ## 需要改 data_names， 默认值是单输入
mod.bind(data_shapes=[('data0', (1,3)),('data0', (1,5))])   ## 需要 定义每个输入的大小
mod.load_params('model-0000.params')
## forward
db = mx.io.DataBatch(data=[data0, data1])  ## data0, data1是ndarray， 多输入组成list后，传入 data
mod.forward(db)
out = mod.get_outputs()                  
```

​	:star: 得到中间层输出：

```python
def get_model(model_str, data_names=['data1','data2']):
    ctx = mx.cpu()  ## 
    _vec = model_str.split(',')
    print('loading model: {}, of {} epoch'.format(_vec[0], _vec[1]))
    sym, arg_params, aux_params = mx.model.load_checkpoint(_vec[0], _vec[1])
    all_layers = sym.get_internals()
    xx_output = all_layers['XXX_output']
    ## softmax_out = mx.symbol.SoftmaxOutput(data=xx_output, name='softmax')  ## if need softmax, add it here
    model = mx.mod.Module(symbol=xx_output, context=ctx, label_names=None)    ## if added softmax: symbol= softmax_out
    model.bind(data_shapes=[('data1',(data1_shapes), ('data2', (data2_shapes)))])
    model.set_params(arg_params, aux_params, allow_missing=True, allow_extra=True)
    return model
model = get_model(mxnet_model_path+',0')
db = mx.io.DataBatch(data=[data1, data2])
model.forward(db, is_train=False)
xx_output = model.get_outputs()
```





:happy: **MXNET 加速：** 使用mxnet_mkl包

