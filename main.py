import json
import argparse
import os
import datetime
from pathlib import Path
import glob
from dataset import FeatureDataset
from model import NeuralDecisionForest

def parse_args():
    parser = argparse.ArgumentParser(description='..')
    parser.add_argument('--non_max_routing', action='store_true')
    parser.add_argument('--n_tree', type=int, default=5)
    parser.add_argument('--depth', type=int, default=10)
    parser.add_argument('--used_feature_rate', type=float, default=0.5)
    parser.add_argument('--num_classes', type=int, default=8)
    parser.add_argument('--val_file', type=str, default='val.json')
    parser.add_argument('--test_file', type=str, default='unseen.json')
    parser.add_argument('--file_prefix', type=str, default='/')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--input_dim', type=int, default=2048)
    parser.add_argument('--device', type=str, choices=['cpu', 'cuda', 'cuda:0', 'cuda:1'], default='cuda:1')
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--out_dir', type=str, default='output')
    parser.add_argument('--output_csv', type=str, default='output_csv')
    parser.add_argument('--model_path', type=str, default='weights/model.pth')

    args = parser.parse_args()
    
   if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir, exist_ok=True)
    args.train_csv = f'{output_csv_dir}/train.csv'
    args.val_csv = f'{output_csv_dir}/val.csv'
    args.test_csv = f'{output_csv_dir}/test_{Path(args.test_file).stem}.csv'

    return args

if __name__ == '__main__':
    args = parse_args()

    model = NeuralDecisionForest(args=args)
    model.to(args.device)
    model.eval()
    val_dataset = FeatureDataset(args, 'val')
    val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=1, shuffle=False,num_workers=0, pin_memory=True)
    tbar = tqdm(loader)
    f = open(file_path, 'w')
    f.write('img,gt,orig_pred,score_orig,score_avg,leafs,max,min,std,mean,ent_e,ent_orig\n')
    with torch.no_grad():
        for i, data in enumerate(tbar):
            input_ = data['x'].to(args.device)  # batch, 3, 256, 256
            output_avg = model(input_)
            score_avg, _ = output_avg.squeeze(0).max(0)
            output_orig, leafs = model.evaluate(input_, self.args.non_max_routing)
            score_orig, pred = output_orig.squeeze(0).max(0)
            max_str, min_str, std_str, mean_str, ent_e, ent_orig = model.evaluate_entropy(input_)
            f.write(f"{data['file'][0]},{data['y'].item()},{pred.item()}"
                    f",{score_orig.item()},{score_avg.item()},{leafs}"
                    f",{max_str},{min_str},{std_str},{mean_str},{ent_e},{ent_orig}\n")
    f.close()



