import os, time, torch, warnings, numpy as np
from PIL import Image
from torchvision import models, transforms
import torch.nn as nn
warnings.filterwarnings("ignore")

types = ['Game','Validation','Super','Verification','Ultra','Extreme','Chrome','World','Task','Enlarged','Exam','Google','Zurich','Friends','Full','Entire','Moscow']

class GeoEmbed(nn.Module):
    def __init__(s): super().__init__(); s.model = nn.Sequential(*list(models.resnet152(weights=models.ResNet152_Weights.DEFAULT).children())[:-1])
    def forward(s, x): return s.model(x).view(x.size(0), -1)

class GeoPredictorNN(nn.Module):
    def __init__(self):
        super().__init__()
        dims = [2048, 1024, 512, 256, 128, 32, 16]
        for i in range(len(dims)-1):
            in_dim, out_dim = dims[i], dims[i+1]
            setattr(self, f'fc{i+1}', nn.Linear(in_dim, out_dim))
            setattr(self, f'batch_norm{i+1}', nn.BatchNorm1d(out_dim))
            setattr(self, f'gelu{i+1}', nn.GELU())
            # smaller dropout on last block
            dropout_rate = 0.1 if i == len(dims)-2 else 0.2
            setattr(self, f'dropout{i+1}', nn.Dropout(dropout_rate))
        # final layer
        self.fc7 = nn.Linear(16, 2)

    def forward(self, x):
        # sequentially apply each block
        for i in range(1, 7):
            x = getattr(self, f'fc{i}')(x)
            x = getattr(self, f'dropout{i}')(x)
            x = getattr(self, f'batch_norm{i}')(x)
            x = getattr(self, f'gelu{i}')(x)
        x = self.fc7(x)
        return x

tf = transforms.Compose([transforms.Resize((1024,1024)), transforms.ToTensor(), transforms.Normalize([.485,.456,.406],[.229,.224,.225])])

def load_imgs(fold,t):
    imgs,paths = [],[]
    for fn in sorted(os.listdir(fold)):
        if fn.endswith('.jpg') and t in fn:
            img = Image.open(os.path.join(fold,fn)).convert('RGB')
            imgs.append(tf(img)); paths.append(os.path.join(fold,fn))
    return torch.stack(imgs), paths

def haversine(c1,c2):
    R,lat1,lon1,lat2,lon2 = 6371,np.radians(c1[:,0]),np.radians(c1[:,1]),np.radians(c2[:,0]),np.radians(c2[:,1])
    dlat,dlon = lat2-lat1,lon2-lon1
    a = np.sin(dlat/2)**2+np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
    return R*2*np.arctan2(np.sqrt(a),np.sqrt(1-a))

def score(err): return 5000 if err < .15 else np.floor(5000*np.exp(-err/2000))

def main(t='Game'):
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if t not in types: raise ValueError("Invalid test type")
    imgs,_ = load_imgs('Roy/Test_Images', t); imgs = imgs.to(dev)
    coords = {
    'Game': [[59.2642,10.4276],[1.4855,103.8676],[54.9927,-1.6732],[19.4745,-99.1974],[58.6133,49.6275]],
    'Validation': [[43.3219,-5.5784],[23.0376,72.5819],[55.93,-3.2679],[51.9187,4.4957],[40.6,-74.3125]],
    'Verification': [[48.1787,16.4149],[39.3544,-76.4284],[12.6546,77.4269],[53.5362,-113.4709],[65.9409,12.2172]],
    'Super': [[47.0676,12.5319],[45.8186,-63.4844],[41.861,12.5368],[-6.3321,106.8518],[35.6062,-77.3732]],
    'Ultra': [[45.5098,27.8273],[41.7107,-93.7364],[-31.3876,-57.9646],[-37.8981,144.626],[54.3667,-6.7719]],
    'Extreme': [[8.6522,81.0068],[46.2477,-80.4337],[60.3055,56.9714],[40.8338,-74.0826],[43.1914,17.3802]],
    'Chrome': [[34.5231,-86.97],[52.293,4.6669],[52.5859,-0.2502],[32.5222,-82.9127],[39.7692,30.5314]],
    'World': [[-6.8147,-38.6534],[12.1392,-68.949],[59.4228,15.8038],[51.553,-0.4759],[14.333,99.6477]],
    'Task': [[34.2469,-82.2092],[49.9352,5.4581],[43.9436,12.4477],[48.0833,-0.6451],[53.356,55.9645]],
    'Enlarged': [[-34.8295,-58.8708],[40.437,-3.6859],[-54.1258,-68.0709],[48.9828,12.6387],[45.9313,-82.4707]],
    'Exam': [[-4.1237,-38.3706],[40.1162,-75.1249],[35.1362,136.7419],[41.6557,-91.5466],[-47.0777,-72.1647]],
    'Google': [[59.4073,15.4157],[52.5644,-110.8206],[-36.87,174.6481],[37.9271,-122.5303],[28.6397,77.293]],
    'Zurich': [[29.959,-95.3912],[62.6314,23.6289],[34.9733,-84.0204],[4.3001,117.8595],[55.7863,-3.923]],
    'Moscow': [[-34.5219,-58.5367],[51.2135,45.9191],[53.1024,-6.064],[37.7153,126.7598],[47.5224,-111.27]],
    'Friends': [[38.9813,-76.9782],[59.8716,30.2994],[-1.5005,29.6217],[59.0596,-3.0761],[1.717,103.4523]],
    'Full': [[41.103,40.7493],[52.5649,-0.2828],[47.2319,38.8685],[41.8301,-70.8728],[23.1109,72.5172]],
    'Entire': [[34.6289,136.5106],[-22.1921,-48.4043],[51.0732,17.7593],[36.5756,-79.8418],[38.1897,15.2438]]}[t]
    real = np.array(coords)
    emb = GeoEmbed().to(dev).eval()
    emb.load_state_dict(torch.load('Roy/ML/Saved_Models/geo_embedding_model_r152_normal.pth', map_location=dev))
    with torch.no_grad(): embs = emb(imgs).cpu()
    weights = [torch.load(f'Roy/ML/Saved_Models/'+f, map_location=dev) for f in sorted(os.listdir('Roy/ML/Saved_Models')) if f.endswith('.pth') and 'embedding' not in f and 'lowest' not in f]
    avg_w = {k: sum(w[k] for w in weights)/len(weights) for k in weights[0].keys()}
    model = GeoPredictorNN().to(dev).eval(); model.load_state_dict(avg_w)
    with torch.no_grad():
        pred = model(embs.to(dev)).cpu().numpy()
        pred[:,0] = (pred[:,0]+90)%180 - 90; pred[:,1] = (pred[:,1]+180)%360 - 180
    errs = haversine(real, pred); pts = [score(e) for e in errs]
    print(f"{t}: Total Points: {sum(pts):.1f}, Errors: {errs}")
    return sum(pts), sum([max(p,0) for p in pts]), sum([1-(p/5000) for p in pts])

if __name__ == '__main__':
    t0 = time.time(); t = 'Game'; main(t) if t != 'All' else [main(tt) for tt in types]; print(f"Time: {time.time()-t0:.1f}s")