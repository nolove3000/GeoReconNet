import argparse, json
from pathlib import Path
import joblib, numpy as np, torch
from torch.utils.data import DataLoader
from common.data_utils import FORMAL_SPLIT_PATH, PROJECT_ROOT, load_formal_split
from high_resolution_clean.pytorch.evaluate import METRIC_NAMES, metrics_from_counts, summarize_records, write_csv
from high_resolution_clean.pytorch.model import HighResGeneratorTorch
from .data import FixedFeatureDataset, build_selected_features, load_raw_1024, select_raw_inputs


def main():
    p=argparse.ArgumentParser(); p.add_argument('--model',type=Path,required=True); p.add_argument('--split',choices=('validation','test'),default='test'); p.add_argument('--batch-size',type=int,default=8); p.add_argument('--bootstrap-resamples',type=int,default=10000); p.add_argument('--bootstrap-seed',type=int,default=2026); a=p.parse_args()
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); path=a.model.resolve(); c=torch.load(path,map_location=device,weights_only=False)
    model=HighResGeneratorTorch(c['input_dimension'],1024,256,0).to(device); model.load_state_dict(c['model_state_dict']); model.eval()
    modal,freq,masks=load_raw_1024(); modal,freq=select_raw_inputs(modal,freq,c['num_modes'],c['num_sensors']); features=build_selected_features(modal,freq)
    scaler=joblib.load(PROJECT_ROOT/c['scaler_path'])['scaler']; features=scaler.transform(features).astype(np.float32)
    split=load_formal_split(FORMAL_SPLIT_PATH,verify_dataset=True); indices=np.asarray(split['validation_indices' if a.split=='validation' else 'test_indices'])
    loader=DataLoader(FixedFeatureDataset(features[indices],masks[indices]),batch_size=a.batch_size,shuffle=False,pin_memory=device.type=='cuda')
    records=[]; micro=np.zeros(3,np.int64); offset=0
    with torch.no_grad():
        for x,y in loader:
            pred=model(x.to(device))<.5; truth=y.to(device)<.5; axes=tuple(range(1,truth.ndim)); tp=(pred&truth).sum(dim=axes).cpu().numpy(); fp=(pred&~truth).sum(dim=axes).cpu().numpy(); fn=(~pred&truth).sum(dim=axes).cpu().numpy(); metric=metrics_from_counts(tp,fp,fn); micro+=np.array([tp.sum(),fp.sum(),fn.sum()])
            for j in range(len(x)):
                g=int(indices[offset+j]); row={'split_position':offset+j,'global_index':g,'sample_id':str(split['sample_ids'][g]),'source_file':str(split['source_files'][g]),'geometry_type':str(split['geometry_types'][g]),'true_positive_pixels':int(tp[j]),'false_positive_pixels':int(fp[j]),'false_negative_pixels':int(fn[j])}; row.update({k:float(metric[k][j]) for k in METRIC_NAMES}); records.append(row)
            offset+=len(x)
    summary=summarize_records(records,a.bootstrap_resamples,a.bootstrap_seed); micro_metrics=metrics_from_counts(*micro); out=PROJECT_ROOT/'outputs/formal_experiments'/c['run_name']/'evaluation'/a.split; write_csv(out/'per_sample_metrics.csv',records); write_csv(out/'summary_metrics.csv',summary); (out/'evaluation_metadata.json').write_text(json.dumps({'model':str(path.relative_to(PROJECT_ROOT)),'checkpoint_epoch':c['epoch'],'best_validation_iou':c['best_validation_iou'],'num_modes':c['num_modes'],'num_sensors':c['num_sensors'],'input_dimension':c['input_dimension'],'split':a.split,'sample_count':len(records),'micro_metrics':{k:float(v) for k,v in micro_metrics.items()}},indent=2),encoding='utf-8')
    for row in summary: print(f"{row['group']}: n={row['n']}; IoU={row['damage_iou_mean']:.6f}; Dice={row['damage_dice_mean']:.6f}; Precision={row['damage_precision_mean']:.6f}; Recall={row['damage_recall_mean']:.6f}")
if __name__=='__main__': main()
