#!/usr/bin/env python3
"""Train PSMAF-YOLOv8 on paired M3FD images with diagnostics."""
import argparse, json, os, platform, sys, time, warnings
from contextlib import nullcontext
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path: sys.path.insert(0, str(REPO_ROOT))
import torch, torchvision, yaml
from torch.utils.data import DataLoader
from detection.datasets import M3FDPairedDataset, paired_collate_fn
from detection.models.psmaf_yolov8 import PSMAFYOLOv8, load_yolov8s_weights
from detection.scripts.psmaf_yolov8_utils import (ModelEMA, WarmupCosineScheduler, _progress,
    cuda_memory_metrics, evaluate_yolov8, resolve_resume_path, set_backbones_trainable,
    strict_device_check, tensor_devices, yolov8_detection_loss)
from detection.scripts.psmaf_yolo_utils import limit_dataset, save_metrics, save_train_log_row, seed_everything


def parser():
    p=argparse.ArgumentParser(); p.add_argument('--dataset-root'); p.add_argument('--data',default='detection/configs/m3fd_psmaf_yolov8.yaml')
    p.add_argument('--epochs',type=int,default=100); p.add_argument('--batch',type=int,default=8); p.add_argument('--imgsz',type=int,default=640); p.add_argument('--device',default='cpu')
    p.add_argument('--weights'); p.add_argument('--project',default='runs/detect'); p.add_argument('--name',default='psmaf-yolov8'); p.add_argument('--seed',type=int,default=0)
    p.add_argument('--amp',action=argparse.BooleanOptionalAction,default=True); p.add_argument('--ema',action=argparse.BooleanOptionalAction,default=True)
    p.add_argument('--progress',action=argparse.BooleanOptionalAction,default=None); p.add_argument('--log-interval',type=int,default=20)
    p.add_argument('--strict-device-check',action='store_true'); p.add_argument('--debug-memory',action='store_true'); p.add_argument('--debug-shapes',action='store_true'); p.add_argument('--save-vis',type=int,default=0)
    p.add_argument('--resume',nargs='?',const='auto'); p.add_argument('--workers',type=int,default=4); p.add_argument('--save-period',type=int,default=10)
    p.add_argument('--fusion-mode',choices=('psmaf','add','concat'),default='psmaf'); p.add_argument('--no-psg',action='store_true'); p.add_argument('--no-msaf',action='store_true')
    p.add_argument('--conf-thres',type=float,default=.25); p.add_argument('--nms-iou',type=float,default=.45); p.add_argument('--eval-train',action='store_true')
    p.add_argument('--lr0',type=float,default=1e-3); p.add_argument('--lrf',type=float,default=.01); p.add_argument('--warmup-epochs',type=float,default=3.)
    p.add_argument('--backbone-lr-mult',type=float,default=.1); p.add_argument('--freeze-backbone-epochs',type=int,default=0); p.add_argument('--debug-num-images',type=int,default=0)
    return p

def append_jsonl(path,value):
    with path.open('a') as f: f.write(json.dumps(value)+'\n')
def counts(model):
    trainable=sum(p.numel() for p in model.parameters() if p.requires_grad); total=sum(p.numel() for p in model.parameters()); return total,trainable,total-trainable
def memory_print(label,device): print('  memory '+label+': '+', '.join(f'{k}={v:.1f}' for k,v in cuda_memory_metrics(device).items()))
def shape_debug(model,rgb,ir):
    result=model.forward_debug(rgb,ir); expected=[(rgb.shape[-2]//s,rgb.shape[-1]//s) for s in model.strides]
    print('Shape diagnostics:\n  RGB input:',tuple(rgb.shape),'\n  IR input:',tuple(ir.shape))
    for key in ('rgb_features','ir_features','fused_features','outputs'):
        shapes=[tuple(x.shape) for x in result[key]]; print(f'  {key}: {shapes}')
        if [x.shape[-2:] for x in result[key]] != [torch.Size(x) for x in expected]: warnings.warn(f'{key} spatial shapes differ from expected {expected}')
    return result['outputs']

def main(argv=None):
    args=parser().parse_args(argv); seed_everything(args.seed); progress=sys.stderr.isatty() if args.progress is None else args.progress
    cfg=yaml.safe_load(Path(args.data).read_text()); root=args.dataset_root or cfg['path']; device=torch.device(args.device); output=Path(args.project)/args.name; output.mkdir(parents=True,exist_ok=True)
    train=limit_dataset(M3FDPairedDataset(root,cfg['train'],args.imgsz),args.debug_num_images); val=limit_dataset(M3FDPairedDataset(root,cfg['val'],args.imgsz),args.debug_num_images)
    kw=dict(batch_size=args.batch,num_workers=args.workers,collate_fn=paired_collate_fn); train_loader=DataLoader(train,shuffle=True,**kw); val_loader=DataLoader(val,**kw)
    model=PSMAFYOLOv8(cfg['nc'],fusion_mode=args.fusion_mode,use_psg=not args.no_psg,use_msaf=not args.no_msaf).to(device)
    if args.resume and args.weights: warnings.warn('--resume takes precedence over --weights')
    checkpoint=resolve_resume_path(args.resume,output) if args.resume else None; start=0; best_map=-1.; best_epoch=0; state=None
    backbone=list(model.rgb_backbone.parameters())+list(model.ir_backbone.parameters()); ids={id(p) for p in backbone}; base=[p for p in model.parameters() if id(p) not in ids]
    optimizer=torch.optim.AdamW([{'params':backbone,'lr':args.lr0*args.backbone_lr_mult,'initial_lr':args.lr0*args.backbone_lr_mult,'name':'backbone'},{'params':base,'lr':args.lr0,'initial_lr':args.lr0,'name':'fusion_neck_head'}])
    scaler=torch.amp.GradScaler('cuda',enabled=args.amp and device.type=='cuda')
    if checkpoint:
        print('Resuming checkpoint:',checkpoint); state=torch.load(checkpoint,map_location=device,weights_only=False); model.load_state_dict(state['model']); start=state.get('epoch',-1)+1; best_map=state.get('best_map',-1.); best_epoch=state.get('best_epoch',0)
        if 'optimizer' in state: optimizer.load_state_dict(state['optimizer'])
        if 'scaler' in state: scaler.load_state_dict(state['scaler'])
    elif args.weights: load_yolov8s_weights(model,args.weights)
    scheduler=WarmupCosineScheduler(optimizer,max(args.epochs*max(len(train_loader),1),1),round(args.warmup_epochs*max(len(train_loader),1)),args.lrf)
    if state and 'scheduler' in state: scheduler.load_state_dict(state['scheduler'])
    ema=ModelEMA(model) if args.ema else None
    if ema and state and state.get('ema'): ema.load_state_dict(state['ema'])
    total,trn,frz=counts(model); cuda_ok=torch.cuda.is_available() and device.type=='cuda'; props=torch.cuda.get_device_properties(device) if cuda_ok else None
    print(f'''Environment:\n  python version: {platform.python_version()}\n  torch version: {torch.__version__}\n  torchvision version: {torchvision.__version__}\n  cuda available: {torch.cuda.is_available()}\n  torch cuda version: {torch.version.cuda}\n  CUDA_VISIBLE_DEVICES: {os.getenv('CUDA_VISIBLE_DEVICES')}\n  selected device: {device}\n  GPU name: {props.name if props else None}\n  GPU total memory MiB: {props.total_memory/1024**2 if props else 0}\nRun configuration:\n  dataset_root: {root}\n  data config path: {args.data}\n  weights path: {args.weights}\n  imgsz: {args.imgsz}\n  batch size: {args.batch}\n  epochs: {args.epochs}\n  amp enabled: {scaler.is_enabled()}\n  ema enabled: {ema is not None}\n  fusion_mode: {args.fusion_mode}\n  use_psg: {not args.no_psg}\n  use_msaf: {not args.no_msaf}\n  freeze_backbone_epochs: {args.freeze_backbone_epochs}\n  lr0: {args.lr0}\n  lrf: {args.lrf}\n  warmup_epochs: {args.warmup_epochs}\n  backbone_lr_mult: {args.backbone_lr_mult}\n  workers: {args.workers}\nModel:\n  total parameter count: {total}\n  trainable parameter count: {trn}\n  frozen parameter count: {frz}\n  first model parameter device: {next(model.parameters()).device}''')
    for epoch in range(start,args.epochs):
        if cuda_ok: torch.cuda.reset_peak_memory_stats(device)
        began=time.perf_counter(); trainable=epoch>=args.freeze_backbone_epochs; set_backbones_trainable(model,trainable); model.train()
        if not trainable: model.rgb_backbone.eval(); model.ir_backbone.eval()
        total,trn,frz=counts(model); print(f'Epoch {epoch+1}/{args.epochs}: backbones are {"trainable" if trainable else "frozen"} (trainable_params={trn}, frozen_params={frz})')
        sums={k:0. for k in ('loss','obj_loss','box_loss','cls_loss','num_pos')}; batches=images=0; iterator=_progress(train_loader,progress,f'train {epoch+1}/{args.epochs}',len(train_loader))
        for bi,batch in enumerate(iterator):
            if args.debug_memory and bi==0: memory_print('before moving data to device',device)
            rgb=batch['rgb'].to(device); ir=batch['ir'].to(device); labels=batch['labels'].to(device); images+=len(rgb)
            if args.debug_memory and bi==0: memory_print('after moving RGB/IR/targets to device',device)
            optimizer.zero_grad(set_to_none=True); context=torch.autocast(device.type) if scaler.is_enabled() else nullcontext()
            with context: outputs=shape_debug(model,rgb,ir) if args.debug_shapes and epoch==start and bi==0 else model(rgb,ir)
            if args.debug_memory and bi==0: memory_print('after forward (first forward peak)',device)
            if bi==0:
                print(f'First batch devices: model={next(model.parameters()).device}, RGB={rgb.device}, IR={ir.device}, labels/targets={labels.device}, outputs={tensor_devices(outputs)}')
            if args.strict_device_check: strict_device_check(model,device,rgb=rgb,ir=ir,labels=labels,outputs=outputs)
            components=yolov8_detection_loss(outputs,labels,cfg['nc']); loss=components['loss']
            if args.debug_memory and bi==0: memory_print('after loss computation',device)
            scaler.scale(loss).backward()
            if args.debug_memory and bi==0: memory_print('after backward (first backward peak)',device)
            scaler.step(optimizer); scaler.update()
            if args.debug_memory and bi==0: memory_print('after optimizer step',device)
            if ema: ema.update(model)
            if args.debug_memory and bi==0: memory_print('after EMA update',device)
            scheduler.step(); batches+=1
            for k in sums: sums[k]+=float(components[k].detach())
            if hasattr(iterator,'set_postfix') and bi%max(args.log_interval,1)==0:
                m=cuda_memory_metrics(device); iterator.set_postfix(batch=f'{bi+1}/{len(train_loader)}',total_loss=f'{loss.item():.4g}',box_loss=f'{components["box_loss"].item():.4g}',cls_loss=f'{components["cls_loss"].item():.4g}',obj_loss=f'{components["obj_loss"].item():.4g}',num_pos=int(components['num_pos']),lr=f'{optimizer.param_groups[1]["lr"]:.3g}',backbone_lr=f'{optimizer.param_groups[0]["lr"]:.3g}',base_lr=f'{optimizer.param_groups[1]["lr"]:.3g}',alloc=f'{m["cuda_allocated_mib"]:.0f}MiB',reserved=f'{m["cuda_reserved_mib"]:.0f}MiB',peak_alloc=f'{m["cuda_peak_allocated_mib"]:.0f}MiB',peak_reserved=f'{m["cuda_peak_reserved_mib"]:.0f}MiB')
        evaluation_model=ema.ema if ema else model
        metrics=evaluate_yolov8(evaluation_model,val_loader,device,args.conf_thres,args.nms_iou,output/'eval_diagnostics.json',progress,args.log_interval,args.save_vis,args.strict_device_check)
        if args.debug_memory: memory_print('after validation',device)
        elapsed=time.perf_counter()-began; mem=cuda_memory_metrics(device)
        row={'epoch':epoch+1,'avg_total_loss':sums['loss']/max(batches,1),'avg_obj_loss':sums['obj_loss']/max(batches,1),'avg_box_loss':sums['box_loss']/max(batches,1),'avg_cls_loss':sums['cls_loss']/max(batches,1),'num_pos':sums['num_pos']/max(batches,1),'learning_rate':optimizer.param_groups[1]['lr'],'backbone_learning_rate':optimizer.param_groups[0]['lr'],'base_learning_rate':optimizer.param_groups[1]['lr'],'backbone_frozen':not trainable,'trainable_params':trn,'frozen_params':frz,'epoch_time_sec':elapsed,'images_per_sec':images/max(elapsed,1e-9),**mem,'val_precision':metrics['precision'],'val_recall':metrics['recall'],'val_AP50':metrics['AP50'],'val_mAP50_95':metrics['mAP50_95']}
        save_train_log_row(row,output); append_jsonl(output/'epoch_metrics.jsonl',{**row,'per_class_AP':metrics['per_class_ap']}); append_jsonl(output/'memory_metrics.jsonl',{'epoch':epoch+1,**mem,'epoch_time_sec':elapsed,'images_per_sec':row['images_per_sec']}); (output/'latest_metrics.json').write_text(json.dumps({**row,'per_class_AP':metrics['per_class_ap']},indent=2))
        print(' '.join(f'{k}={v:.6g}' if isinstance(v,float) else f'{k}={v}' for k,v in row.items()))
        improved=metrics['mAP50_95']>best_map
        if improved: best_map=metrics['mAP50_95']; best_epoch=epoch+1; best_snapshot=metrics.copy()
        elif not 'best_snapshot' in locals(): best_snapshot=metrics.copy()
        state={'model':model.state_dict(),'ema':ema.state_dict() if ema else None,'optimizer':optimizer.state_dict(),'scheduler':scheduler.state_dict(),'scaler':scaler.state_dict(),'epoch':epoch,'args':vars(args),'metrics':metrics,'best_map':best_map,'best_epoch':best_epoch}; torch.save(state,output/'last.pt')
        if improved: torch.save(state,output/'best.pt')
        if args.save_period>0 and (epoch+1)%args.save_period==0: torch.save(state,output/f'epoch{epoch+1}.pt')
        save_metrics(metrics,output,'metrics')
        best={'best_epoch':best_epoch,'best_metric_name':'mAP50_95','best_metric_value':best_map,'best_precision':best_snapshot['precision'],'best_recall':best_snapshot['recall'],'best_AP50':best_snapshot['AP50'],'best_mAP50_95':best_snapshot['mAP50_95'],'checkpoint_path':str(output/'best.pt'),'used_ema_for_eval':ema is not None,'imgsz':args.imgsz,'batch':args.batch,'epochs':args.epochs,'lr0':args.lr0,'lrf':args.lrf,'warmup_epochs':args.warmup_epochs,'backbone_lr_mult':args.backbone_lr_mult,'freeze_backbone_epochs':args.freeze_backbone_epochs,'amp_enabled':scaler.is_enabled(),'ema_enabled':ema is not None}; (output/'best_metrics.json').write_text(json.dumps(best,indent=2))
        print(f'''Epoch {epoch+1}/{args.epochs} summary:\n  train:\n    total_loss: {row['avg_total_loss']:.6g}\n    box_loss: {row['avg_box_loss']:.6g}\n    cls_loss: {row['avg_cls_loss']:.6g}\n    obj_loss: {row['avg_obj_loss']:.6g}\n    num_pos: {row['num_pos']:.3f}\n    lr: {row['learning_rate']:.6g}\n    backbone_lr: {row['backbone_learning_rate']:.6g}\n    base_lr: {row['base_learning_rate']:.6g}\n    epoch_time_sec: {elapsed:.2f}\n    images_per_sec: {row['images_per_sec']:.2f}\n    cuda_allocated_mib: {mem['cuda_allocated_mib']:.1f}\n    cuda_reserved_mib: {mem['cuda_reserved_mib']:.1f}\n    cuda_peak_allocated_mib: {mem['cuda_peak_allocated_mib']:.1f}\n    cuda_peak_reserved_mib: {mem['cuda_peak_reserved_mib']:.1f}\n  val:\n    precision: {metrics['precision']:.6g}\n    recall: {metrics['recall']:.6g}\n    AP50: {metrics['AP50']:.6g}\n    mAP50_95: {metrics['mAP50_95']:.6g}\n    per_class_AP: {metrics['per_class_ap']}\n  best:\n    best_epoch: {best_epoch}\n    best_metric_name: mAP50_95\n    best_metric_value: {best_map:.6g}''')
if __name__=='__main__': main()
