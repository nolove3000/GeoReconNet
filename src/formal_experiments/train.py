import argparse, json, time
from pathlib import Path
import numpy as np, torch
from torch.utils.data import DataLoader

from common.data_utils import FORMAL_SPLIT_PATH, PROJECT_ROOT, load_formal_split
from high_resolution_clean.pytorch.model import HighResGeneratorTorch, HybridLossTorch, TORCH_ARCHITECTURE_VERSION, TORCH_INITIALIZATION_VERSION
from high_resolution_clean.pytorch.train import build_lr_scheduler, run_epoch
from .data import FixedFeatureDataset, NoiseAugmentedDataset, artifact_version, build_selected_features, fit_or_load_scaler, load_raw_1024, scaler_path, select_raw_inputs


def arguments():
    p=argparse.ArgumentParser(); p.add_argument('--kind',choices=('sensitivity','noise_augmented'),required=True)
    p.add_argument('--num-modes',type=int,choices=(1,2,3,4,5,6),default=6); p.add_argument('--num-sensors',type=int,choices=tuple(range(8)),default=7)
    p.add_argument('--epochs',type=int,default=200); p.add_argument('--batch-size',type=int,default=8); p.add_argument('--seed',type=int,default=42)
    p.add_argument('--max-mode-noise',type=float,default=.05); p.add_argument('--max-frequency-noise',type=float,default=.01)
    return p.parse_args()


def main():
    a=arguments(); np.random.seed(a.seed); torch.manual_seed(a.seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(a.seed)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    modal,frequency,masks=load_raw_1024(); modal,frequency=select_raw_inputs(modal,frequency,a.num_modes,a.num_sensors)
    split=load_formal_split(FORMAL_SPLIT_PATH,verify_dataset=True); tr=split['train_indices']; va=split['validation_indices']
    clean=build_selected_features(modal,frequency); scaler=fit_or_load_scaler(clean,tr,a.num_modes,a.num_sensors)
    if a.kind=='noise_augmented':
        if (a.num_modes,a.num_sensors)!=(6,7): raise ValueError('Formal noise augmentation uses the full 6-mode/7-sensor input')
        train_data=NoiseAugmentedDataset(modal[tr],frequency[tr],masks[tr],scaler,tr,a.seed,a.max_mode_noise,a.max_frequency_noise)
    else: train_data=FixedFeatureDataset(scaler.transform(clean[tr]).astype(np.float32),masks[tr])
    val_data=FixedFeatureDataset(scaler.transform(clean[va]).astype(np.float32),masks[va])
    noisy_val_data=(NoiseAugmentedDataset(modal[va],frequency[va],masks[va],scaler,va,a.seed+20_000,a.max_mode_noise,a.max_frequency_noise,randomize_levels=False) if a.kind=='noise_augmented' else None)
    opts=dict(batch_size=a.batch_size,num_workers=0,pin_memory=device.type=='cuda')
    train_loader=DataLoader(train_data,shuffle=True,generator=torch.Generator().manual_seed(a.seed),**opts); val_loader=DataLoader(val_data,shuffle=False,**opts)
    noisy_val_loader=DataLoader(noisy_val_data,shuffle=False,**opts) if noisy_val_data is not None else None
    model=HighResGeneratorTorch(input_dimension=clean.shape[1],out_size=1024,base_channels=256,dropout=0).to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=2e-4,weight_decay=1e-5,eps=1e-7); scheduler=build_lr_scheduler(optimizer); loss=HybridLossTorch()
    run=f"{a.kind}_m{a.num_modes}_s{a.num_sensors}_{artifact_version(a.num_sensors)}"; model_dir=PROJECT_ROOT/'models/formal_experiments'; out=PROJECT_ROOT/'outputs/formal_experiments'/run/'training'
    model_dir.mkdir(parents=True,exist_ok=True); out.mkdir(parents=True,exist_ok=True); checkpoint_path=model_dir/f'best_{run}.pt'; history=[]; best=-1
    print(f'run={run}; device={device}; input={clean.shape[1]}; parameters={sum(p.numel() for p in model.parameters()):,}')
    for epoch in range(1,a.epochs+1):
        if isinstance(train_data,NoiseAugmentedDataset): train_data.set_epoch(epoch)
        start=time.perf_counter(); lr=optimizer.param_groups[0]['lr']; tl,tm=run_epoch(model,train_loader,loss,device,optimizer); vl,vm=run_epoch(model,val_loader,loss,device)
        noisy_vl,noisy_vm=run_epoch(model,noisy_val_loader,loss,device) if noisy_val_loader is not None else (vl,vm); selection_vm=noisy_vm if noisy_val_loader is not None else vm; scheduler.step(noisy_vl if noisy_val_loader is not None else vl)
        rec={'epoch':epoch,'learning_rate':lr,'next_learning_rate':optimizer.param_groups[0]['lr'],'training_loss':tl,'validation_loss':vl,'noisy_validation_loss':noisy_vl,**{f'training_{k}':v for k,v in tm.items()},**{f'validation_{k}':v for k,v in vm.items()},**{f'noisy_validation_{k}':v for k,v in noisy_vm.items()},'seconds':time.perf_counter()-start}; history.append(rec)
        if selection_vm['damage_iou']>best:
            best=selection_vm['damage_iou']; torch.save({'model_state_dict':model.state_dict(),'epoch':epoch,'best_validation_iou':best,'checkpoint_selection':'fixed_max_noise_validation_iou' if noisy_val_loader is not None else 'clean_validation_iou','run_name':run,'kind':a.kind,'num_modes':a.num_modes,'num_sensors':a.num_sensors,'input_dimension':clean.shape[1],'base_channels':256,'out_size':1024,'seed':a.seed,'architecture_version':TORCH_ARCHITECTURE_VERSION,'initialization_version':TORCH_INITIALIZATION_VERSION,'scaler_path':str(scaler_path(a.num_modes,a.num_sensors).relative_to(PROJECT_ROOT)),'split_path':str(FORMAL_SPLIT_PATH.relative_to(PROJECT_ROOT)),'train_indices':np.asarray(tr).tolist(),'validation_indices':np.asarray(va).tolist(),'test_indices':np.asarray(split['test_indices']).tolist(),'max_mode_noise':a.max_mode_noise if a.kind=='noise_augmented' else 0.0,'max_frequency_noise':a.max_frequency_noise if a.kind=='noise_augmented' else 0.0},checkpoint_path)
        (out/'metrics.json').write_text(json.dumps(history,indent=2),encoding='utf-8')
        print(f"Epoch {epoch:03d}: loss={tl:.6f}; val_loss={vl:.6f}; noisy_val_loss={noisy_vl:.6f}; IoU={tm['damage_iou']:.6f}; val_IoU={vm['damage_iou']:.6f}; noisy_val_IoU={noisy_vm['damage_iou']:.6f}; best={best:.6f}; s={rec['seconds']:.2f}",flush=True)
    print(f'Best checkpoint: {checkpoint_path}')
if __name__=='__main__': main()
