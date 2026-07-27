import argparse, csv, json
from pathlib import Path
import joblib, numpy as np, torch
from common.data_utils import FORMAL_SCALER_PATH, FORMAL_SPLIT_PATH, PROJECT_ROOT, load_formal_scaler, load_formal_split
from common.noise import add_measurement_noise
from high_resolution_clean.pytorch.model import HighResGeneratorTorch
from .data import build_selected_features, load_raw_1024


DEFAULT_CLEAN=PROJECT_ROOT/'models/pytorch/best_main_pytorch_1024_no_norm_dropout_000_v5_continue150.pt'
DEFAULT_NOISE=PROJECT_ROOT/'models/formal_experiments/best_noise_augmented_m6_s7_v1.pt'


def load_model(path,device,noise_model):
    c=torch.load(path,map_location=device,weights_only=False); dim=int(c.get('input_dimension',132)); model=HighResGeneratorTorch(dim,1024,256,0).to(device); model.load_state_dict(c['model_state_dict']); model.eval()
    scaler=joblib.load(PROJECT_ROOT/c['scaler_path'])['scaler'] if noise_model else load_formal_scaler()
    return model,scaler,c


def evaluate(model,scaler,modal,freq,masks,indices,mode_level,freq_level,seed,batch_size,device):
    metric_values={key:[] for key in ('damage_iou','damage_dice','damage_precision','damage_recall')}
    for start in range(0,len(indices),batch_size):
        ids=indices[start:start+batch_size]; noisy_m=[]; noisy_f=[]
        for g in ids:
            rng=np.random.default_rng(seed+int(g)); m,f=add_measurement_noise(modal[g],freq[g],mode_level,freq_level,rng); noisy_m.append(m); noisy_f.append(f)
        x=scaler.transform(build_selected_features(np.asarray(noisy_m),np.asarray(noisy_f))).astype(np.float32); y=torch.from_numpy(masks[ids,None]).to(device)
        with torch.no_grad(): pred=model(torch.from_numpy(x).to(device))<.5
        truth=y<.5
        axes=tuple(range(1,truth.ndim)); tp=(pred&truth).sum(dim=axes).cpu().numpy(); fp=(pred&~truth).sum(dim=axes).cpu().numpy(); fn=(~pred&truth).sum(dim=axes).cpu().numpy()
        counts={'damage_iou':(tp,tp+fp+fn),'damage_dice':(2*tp,2*tp+fp+fn),'damage_precision':(tp,tp+fp),'damage_recall':(tp,tp+fn)}
        for key,(numerator,denominator) in counts.items():
            metric_values[key].extend(np.divide(numerator,denominator,out=np.zeros_like(numerator,dtype=np.float64),where=denominator!=0).tolist())
    return {key:float(np.mean(values)) for key,values in metric_values.items()}


def main():
    p=argparse.ArgumentParser(); p.add_argument('--clean-model',type=Path,default=DEFAULT_CLEAN); p.add_argument('--noise-model',type=Path,default=DEFAULT_NOISE); p.add_argument('--models',choices=('clean','noise','both'),default='clean'); p.add_argument('--mode-noise-levels',type=float,nargs='+',default=(0,.01,.03,.05)); p.add_argument('--frequency-noise',type=float,default=.01,help='Paired nonzero frequency noise used when --frequency-noise-levels is omitted.'); p.add_argument('--frequency-noise-levels',type=float,nargs='+',help='If supplied, evaluate the Cartesian product with all mode-noise levels.'); p.add_argument('--repeats',type=int,default=5); p.add_argument('--seed',type=int,default=2026); p.add_argument('--batch-size',type=int,default=8); p.add_argument('--output-name',default='baseline_clean_v1'); a=p.parse_args()
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); modal,freq,masks=load_raw_1024(); indices=np.asarray(load_formal_split(FORMAL_SPLIT_PATH,verify_dataset=True)['test_indices']); rows=[]
    candidates=[]
    if a.models in ('clean','both'): candidates.append(('clean_trained',a.clean_model,False))
    if a.models in ('noise','both'): candidates.append(('noise_augmented',a.noise_model,True))
    noise_pairs=([(mode,frequency) for mode in sorted(set(a.mode_noise_levels)) for frequency in sorted(set(a.frequency_noise_levels))] if a.frequency_noise_levels is not None else [(level,0.0 if level==0 else a.frequency_noise) for level in a.mode_noise_levels])
    for label,path,is_noise in candidates:
        if not path.is_file(): raise FileNotFoundError(f'Checkpoint not found: {path}')
        model,scaler,c=load_model(path.resolve(),device,is_noise)
        for level,frequency_level in noise_pairs:
            repeat_count=1 if level==0 and frequency_level==0 else a.repeats
            for repeat in range(repeat_count):
                metrics=evaluate(model,scaler,modal,freq,masks,indices,level,frequency_level,a.seed+repeat*1_000_003,a.batch_size,device); rows.append({'model':label,'mode_noise':level,'frequency_noise':frequency_level,'repeat':repeat,**metrics}); print(label,level,repeat,metrics['damage_iou'],flush=True)
    summary=[]
    for label,_,_ in candidates:
        for level,frequency_level in noise_pairs:
            subset=[r for r in rows if r['model']==label and r['mode_noise']==level and r['frequency_noise']==frequency_level]; row={'model':label,'mode_noise':level,'frequency_noise':frequency_level,'repeats':len(subset)}
            for key in ('damage_iou','damage_dice','damage_precision','damage_recall'):
                values=np.array([r[key] for r in subset]); row[key+'_mean']=float(values.mean()); row[key+'_std']=float(values.std(ddof=1)) if len(values)>1 else 0.0
            summary.append(row)
    out=PROJECT_ROOT/'outputs/formal_experiments/noise_robustness'/a.output_name; out.mkdir(parents=True,exist_ok=True)
    for name,data in [('repeats.csv',rows),('summary.csv',summary)]:
        with (out/name).open('w',newline='',encoding='utf-8') as f: w=csv.DictWriter(f,fieldnames=data[0].keys()); w.writeheader(); w.writerows(data)
    (out/'protocol.json').write_text(json.dumps({**vars(a),'noise_pairs':noise_pairs},default=str,indent=2),encoding='utf-8')
if __name__=='__main__': main()
