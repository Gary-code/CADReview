
import torch
import argparse
import os
import numpy as np
from tqdm import tqdm
import json
import time
import random
import glob
import warnings
from scipy.stats import entropy
from sklearn.neighbors import NearestNeighbors
from plyfile import PlyData, PlyElement
import pandas as pd

N_POINTS = 2000

random.seed(1234)

PC_ROOT = "../data/pc_cad"
RECORD_FILE = "../data/train_val_test_split.json"

def find_files(directory, postfix=""):
    output_files = []
    # file_list = os.listdir(directory.replace("stl","clean"))
    for root, dirs, files in os.walk(directory):
        for file in files:
            # name = root.split('/')[-1] + ".scad" 
            if file.endswith(postfix):
            # if file.endswith(".stl") and name in file_list:
                output_files.append(os.path.join(root, file))
    return output_files

def read_ply(path):
    with open(path, 'rb') as f:
        plydata = PlyData.read(f)
        x = np.array(plydata['vertex']['x'])
        y = np.array(plydata['vertex']['y'])
        z = np.array(plydata['vertex']['z'])
        vertex = np.stack([x, y, z], axis=1)
    return vertex


def distChamfer(a, b):
    x, y = a, b
    bs, num_points, points_dim = x.size()
    xx = torch.bmm(x, x.transpose(2, 1))
    yy = torch.bmm(y, y.transpose(2, 1))
    zz = torch.bmm(x, y.transpose(2, 1))
    diag_ind = torch.arange(0, num_points).to(a).long()
    rx = xx[:, diag_ind, diag_ind].unsqueeze(1).expand_as(xx)
    ry = yy[:, diag_ind, diag_ind].unsqueeze(1).expand_as(yy)
    P = (rx.transpose(2, 1) + ry - 2 * zz)
    return P.min(1)[0], P.min(2)[0]


def _pairwise_CD(sample_pcs, ref_pcs, batch_size):
    N_sample = sample_pcs.shape[0]
    N_ref = ref_pcs.shape[0]
    all_cd = []
    all_emd = []
    iterator = range(N_sample)
    matched_gt = []
    pbar = iterator
    # pbar = tqdm(iterator)
    for sample_b_start in pbar:
        sample_batch = sample_pcs[sample_b_start]

        cd_lst = []
        emd_lst = []
        for ref_b_start in range(0, N_ref, batch_size):
            ref_b_end = min(N_ref, ref_b_start + batch_size)
            ref_batch = ref_pcs[ref_b_start:ref_b_end]

            batch_size_ref = ref_batch.size(0)
            sample_batch_exp = sample_batch.view(1, -1, 3).expand(batch_size_ref, -1, -1)
            sample_batch_exp = sample_batch_exp.contiguous()

            dl, dr = distChamfer(sample_batch_exp, ref_batch)
            cd_lst.append((dl.mean(dim=1) + dr.mean(dim=1)).view(1, -1))

        cd_lst = torch.cat(cd_lst, dim=1)
        all_cd.append(cd_lst)

        hit = np.argmin(cd_lst.detach().cpu().numpy()[0])
        matched_gt.append(hit)
        # pbar.set_postfix({"cov": len(np.unique(matched_gt)) * 1.0 / N_ref})

    all_cd = torch.cat(all_cd, dim=0)  # N_sample, N_ref

    return all_cd


def compute_cov_mmd(sample_pcs, ref_pcs, batch_size):
    all_dist = _pairwise_CD(sample_pcs, ref_pcs, batch_size)

    N_sample, N_ref = all_dist.size(0), all_dist.size(1)
    min_val_fromsmp, min_idx = torch.min(all_dist, dim=1)
    min_val, _ = torch.min(all_dist, dim=0)
    mmd = min_val.mean()
    cov = float(min_idx.unique().view(-1).size(0)) / float(N_ref)
    cov = torch.tensor(cov).to(all_dist)

    return {
        'MMD-CD': round(mmd.item() * 1000, 2),
        # 'COV-CD': cov.item(),
        'Median-CD': round(min_val_fromsmp.median().item() * 1000, 2),
        'Mean-CD': round(min_val_fromsmp.mean().item() * 1000, 2),
    }


def jsd_between_point_cloud_sets(sample_pcs, ref_pcs, in_unit_sphere, resolution=28):
    '''Computes the JSD between two sets of point-clouds, as introduced in the paper ```Learning Representations And Generative Models For 3D Point Clouds```.
    Args:
        sample_pcs: (np.ndarray S1xR2x3) S1 point-clouds, each of R1 points.
        ref_pcs: (np.ndarray S2xR2x3) S2 point-clouds, each of R2 points.
        resolution: (int) grid-resolution. Affects granularity of measurements.
    '''
    sample_grid_var = entropy_of_occupancy_grid(sample_pcs, resolution, in_unit_sphere)[1]
    ref_grid_var = entropy_of_occupancy_grid(ref_pcs, resolution, in_unit_sphere)[1]
    return jensen_shannon_divergence(sample_grid_var, ref_grid_var)


def entropy_of_occupancy_grid(pclouds, grid_resolution, in_sphere=False):
    '''Given a collection of point-clouds, estimate the entropy of the random variables
    corresponding to occupancy-grid activation patterns.
    Inputs:
        pclouds: (numpy array) #point-clouds x points per point-cloud x 3
        grid_resolution (int) size of occupancy grid that will be used.
    '''
    epsilon = 10e-4
    bound = 1 + epsilon
    if abs(np.max(pclouds)) > bound or abs(np.min(pclouds)) > bound:
        print(abs(np.max(pclouds)), abs(np.min(pclouds)))
        warnings.warn('Point-clouds are not in unit cube.')

    if in_sphere and np.max(np.sqrt(np.sum(pclouds ** 2, axis=2))) > bound:
        warnings.warn('Point-clouds are not in unit sphere.')

    grid_coordinates, _ = unit_cube_grid_point_cloud(grid_resolution, in_sphere)
    grid_coordinates = grid_coordinates.reshape(-1, 3)
    grid_counters = np.zeros(len(grid_coordinates))
    grid_bernoulli_rvars = np.zeros(len(grid_coordinates))
    nn = NearestNeighbors(n_neighbors=1).fit(grid_coordinates)

    for pc in pclouds:
        _, indices = nn.kneighbors(pc)
        indices = np.squeeze(indices)
        for i in indices:
            grid_counters[i] += 1
        indices = np.unique(indices)
        for i in indices:
            grid_bernoulli_rvars[i] += 1

    acc_entropy = 0.0
    n = float(len(pclouds))
    for g in grid_bernoulli_rvars:
        p = 0.0
        if g > 0:
            p = float(g) / n
            acc_entropy += entropy([p, 1.0 - p])

    return acc_entropy / len(grid_counters), grid_counters


def unit_cube_grid_point_cloud(resolution, clip_sphere=False):
    '''Returns the center coordinates of each cell of a 3D grid with resolution^3 cells,
    that is placed in the unit-cube.
    If clip_sphere it True it drops the "corner" cells that lie outside the unit-sphere.
    '''
    grid = np.ndarray((resolution, resolution, resolution, 3), np.float32)
    spacing = 1.0 / float(resolution - 1) * 2
    for i in range(resolution):
        for j in range(resolution):
            for k in range(resolution):
                grid[i, j, k, 0] = i * spacing - 0.5 * 2
                grid[i, j, k, 1] = j * spacing - 0.5 * 2
                grid[i, j, k, 2] = k * spacing - 0.5 * 2

    if clip_sphere:
        grid = grid.reshape(-1, 3)
        grid = grid[np.linalg.norm(grid, axis=1) <= 0.5]

    return grid, spacing


def jensen_shannon_divergence(P, Q):
    if np.any(P < 0) or np.any(Q < 0):
        raise ValueError('Negative values.')
    if len(P) != len(Q):
        raise ValueError('Non equal size.')

    P_ = P / np.sum(P)  # Ensure probabilities.
    Q_ = Q / np.sum(Q)

    e1 = entropy(P_, base=2)
    e2 = entropy(Q_, base=2)
    e_sum = entropy((P_ + Q_) / 2.0, base=2)
    res = e_sum - ((e1 + e2) / 2.0)

    res2 = _jsdiv(P_, Q_)

    if not np.allclose(res, res2, atol=10e-5, rtol=0):
        warnings.warn('Numerical values of two JSD methods don\'t agree.')

    return res


def _jsdiv(P, Q):
    '''another way of computing JSD'''

    def _kldiv(A, B):
        a = A.copy()
        b = B.copy()
        idx = np.logical_and(a > 0, b > 0)
        a = a[idx]
        b = b[idx]
        return np.sum([v for v in a * np.log2(a / b)])

    P_ = P / np.sum(P)
    Q_ = Q / np.sum(Q)

    M = 0.5 * (P_ + Q_)

    return 0.5 * (_kldiv(P_, M) + _kldiv(Q_, M))


def downsample_pc(points, n):
    sample_idx = random.sample(list(range(points.shape[0])), n)
    return points[sample_idx]


def normalize_pc(points):
    scale = np.max(np.abs(points))
    points = points / scale
    return points

def compute_compile_success_rate(args, category):
    # all_plys = find_files(os.path.join(args.src, "output/ply/"+category), '_pred.ply')
    # all_plys = find_files(args.src, '_pred.ply')
    # all_scad = find_files(args.src, '_gt.scad')
    if "cube" == category:
        all_plys = find_files(os.path.join(args.src, "output/ply/"+category), '_pred.ply')
        success_rate = len(all_plys)/1076
    elif "real" == category:
        all_plys = find_files(os.path.join(args.src, "output/ply/"+category), '_pred.ply')
        success_rate = len(all_plys)/539
    else:
        all_plys = find_files(args.src, '_pred.ply')
        success_rate = len(all_plys)/1615
    # print("Compile success rate: ", success_rate)
    return success_rate


def collect_test_set_pcs(args):
    start = time.time()

    # with open(os.path.join(RECORD_FILE), "r") as fp:
    #     all_data = json.load(fp)['test']
    # select_idx = random.sample(list(range(len(all_data))), args.n_test + 5)
    # all_data = [all_data[x] for x in select_idx]
    all_data = find_files(args.src, '_gt.ply')

    print("ref ply data length: ", len(all_data))
    ref_pcs = []
    ref_data = []
    for data_id in tqdm(all_data):
        pc_path = data_id
        if not os.path.exists(pc_path):
            continue
        pc = read_ply(pc_path)
        ref_data.append(pc)
        if pc.shape[0] < N_POINTS:
            continue
        if pc.shape[0] > N_POINTS:
            pc = downsample_pc(pc, N_POINTS)

        pc = normalize_pc(pc)
        ref_pcs.append(pc)
    min_point = min([pc.shape[0] for pc in ref_data])
    max_point = max([pc.shape[0] for pc in ref_data])
    print("min point: ", min_point)
    print("max point: ", max_point)
    ref_pcs = np.stack(ref_pcs, axis=0)
    print("reference point clouds: {}".format(ref_pcs.shape))
    print("time: {:.2f}s".format(time.time() - start))
    return ref_pcs


def collect_src_pcs(args):
    start = time.time()

    if args.eval_type == 'pred':
        all_paths = find_files(args.src, '_pred.ply')
    else:
        all_paths = find_files(args.src, '_error.ply')
    print("pred ply data length: ", len(all_paths))
    gen_pcs = []
    for path in tqdm(all_paths):
        pc = read_ply(path)
        if pc.shape[0] < N_POINTS:
            continue
        if pc.shape[0] > N_POINTS:
            # print(path)
            pc = downsample_pc(pc, N_POINTS)

        # if np.max(np.abs(pc)) > 1:
        pc = normalize_pc(pc)
        gen_pcs.append(pc)

    gen_pcs = np.stack(gen_pcs, axis=0)
    print("generated point clouds: {}".format(gen_pcs.shape))
    print("time: {:.2f}s".format(time.time() - start))
    return gen_pcs


def Compute_CD(args):
    if args.eval_type == 'pred':
        pred_paths = find_files(args.src, '_pred.ply')
    else:
        pred_paths = find_files(args.src, '_error.ply')
    CDs = []
    for path in tqdm(pred_paths):
        gt_path = path.replace('_pred.ply', '_gt.ply').replace('_error.ply', '_gt.ply')
        if not os.path.exists(gt_path):
            continue
        pred_pc = read_ply(path)
        gt_pc = read_ply(gt_path)
        N_POINTS = min(pred_pc.shape[0], gt_pc.shape[0])
        pred_pc_sample = downsample_pc(pred_pc, N_POINTS)
        gt_pc_sample = downsample_pc(gt_pc, N_POINTS)
        pred_pc_sample = normalize_pc(pred_pc_sample)
        gt_pc_sample = normalize_pc(gt_pc_sample)
        pred_pc_sample = np.stack([pred_pc_sample], axis=0)
        gt_pc_sample = np.stack([gt_pc_sample], axis=0)
        pred_pc_sample = torch.tensor(pred_pc_sample).cuda()
        gt_pc_sample = torch.tensor(gt_pc_sample).cuda()
        CD = _pairwise_CD(pred_pc_sample, gt_pc_sample, batch_size=args.batch_size).cpu().numpy()
        CDs.append(CD[0])
        torch.cuda.empty_cache()
    

    Median_CD = np.median(CDs)
    Mean_CD = np.mean(CDs)

        
    
    return {
        # 'COV-CD': cov.item(),
        'Median-CD': float(Median_CD),
        'Mean-CD': float(Mean_CD),
    }
    
def save_to_csv(results, args):
    csv_path = os.path.join("./evaluate", '3D_metric.csv')
    experiment_name = os.path.basename(os.path.normpath(args.src))
    
    new_row = {'experiment': experiment_name}
    for category in ['cube', 'real', 'all']:
        if category in results:
            metrics = results[category]
            for metric_name, value in metrics.items():

                base_metric = '-'.join(metric_name.split('-')[2:]) if 'avg' in metric_name else metric_name.split('-', 1)[1]

                column_name = f"{category}_{base_metric}"
                new_row[column_name] = value
    

    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        df = pd.DataFrame()
    

    new_df = pd.DataFrame([new_row])
    df = pd.concat([df, new_df], ignore_index=True)
    

    df.to_csv(csv_path, index=False)
    print(f"\nResults saved to: {csv_path}")   

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", type=str, default="")
    parser.add_argument('-g', '--gpu_ids', type=str, default="0,1,2,3,4,5,6,7", help="gpu to use, e.g. 0  0,1,2. CPU not supported.")
    parser.add_argument("--eval_type", type=str, default="pred")
    parser.add_argument("--n_test", type=int, default=1000)
    parser.add_argument("--multi", type=int, default=1)
    parser.add_argument("--times", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("-o", "--output", type=str)
    args = parser.parse_args()
    print("n_test: {}, multiplier: {}, repeat times: {}".format(args.n_test, args.multi, args.times))
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_ids)

    if args.output is None:
        args.output = os.path.join(args.src, 'eval_gen.txt')

    fp = open(args.output, "w")

    categories = ['cube', 'real', 'all']
    all_results = {}

    for category in categories:
        print(f"\nComputing metrics for {category}...")
        result_list = []
        
        for i in range(args.times):
            print(f"iteration {i}...")
            
            if category == 'all':
                ref_pcs = collect_test_set_pcs(args)
                sample_pcs = collect_src_pcs(args)
            else:
                original_src = args.src
                args.src = os.path.join(original_src, "output/ply/"+category)
                ref_pcs = collect_test_set_pcs(args)
                sample_pcs = collect_src_pcs(args)
                args.src = original_src

            if len(sample_pcs) == 0 or len(ref_pcs) == 0:
                print(f"No data found for category: {category}")
                continue

            jsd = jsd_between_point_cloud_sets(sample_pcs, ref_pcs, in_unit_sphere=False)

            sample_pcs = torch.tensor(sample_pcs).cuda()
            ref_pcs = torch.tensor(ref_pcs).cuda()
            result = compute_cov_mmd(sample_pcs, ref_pcs, batch_size=args.batch_size)
            result.update({"JSD": round(jsd * 1000, 2)})
            
            print(f"{category} result:", result)
            print(f"{category} result:", result, file=fp)
            result_list.append(result)

        if result_list:
            avg_result = {}
            for k in result_list[0].keys():
                avg_result.update({f"{category}-avg-{k}": np.mean([x[k] for x in result_list])})
            
            compile_success_rate = compute_compile_success_rate(args, category)
            avg_result.update({f"{category}-compile_success_rate": round(compile_success_rate * 100, 2)})
            
            all_results[category] = avg_result
            print(f"\n{category} average result:")
            print(avg_result)
            print(avg_result, file=fp)

    print("\nFinal results for all categories:", file=fp)
    print(all_results, file=fp)
    fp.close()

    save_to_csv(all_results, args)

if __name__ == '__main__':
    main()
