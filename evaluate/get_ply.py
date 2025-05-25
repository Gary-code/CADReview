import trimesh
import argparse
import subprocess
import os
import json
from tqdm import tqdm
from scale_nums import scale_code
from scale_stl import scale_and_refine_stl
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor

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

def export_scad_to_stl(scad_file, output_stl_path, openscad_path='openscad'):

    if not os.path.exists(os.path.dirname(output_stl_path)):
        os.makedirs(os.path.dirname(output_stl_path))
    try:
        subprocess.run(
            [openscad_path, "-o", output_stl_path, scad_file],
            check=True,
            timeout=10
        )
        return f"{output_stl_path}"
    except subprocess.CalledProcessError as e:
        return f"{scad_file}: {e}"
    except subprocess.TimeoutExpired:
        return f"{scad_file} (>{10}s)."

def stl_to_ply(input_file, output_file):
    """
    Convert an STL file to PLY format.
    
    Args:
        input_file (str): Path to the input STL file.
        output_file (str): Path to the output PLY file.
    """
    try:
        # Load the STL file using trimesh
        mesh = trimesh.load_mesh(input_file)
        if not isinstance(mesh, trimesh.Trimesh):
            raise ValueError("The input file does not represent a valid 3D mesh.")
        
        # Export the mesh to PLY format
        mesh.export(output_file)
        # print(f"Successfully converted {input_file} to {output_file}.")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", type=str, default="")
    args = parser.parse_args()
    if 'baseline' in args.src:
        json_path = find_files(args.src, ".json")[0]
        with open(json_path, "r") as f:
            data = json.load(f)     
    else:    
        jsonl_paths = find_files(args.src, ".jsonl")
        for path in jsonl_paths:
            if 'editor' in path:
                jsonl_path = path
                break
        with open(jsonl_path, "r") as f:
            data = [json.loads(line) for line in f.readlines()]
        
    output_dir = os.path.join(args.src, "output")
    scad_dir = os.path.join(output_dir, "scad")
    id = 0
    for i in range(len(data)):
        # if "No error" not in data[i]["feedback_messages"][-1]["content"]:
            if not os.path.exists(scad_dir):
                os.makedirs(scad_dir)
            if 'baseline' in args.src:
                data_type = data[i]['data_type']
                file_name = ""
                pred_code = data[i]["pred"]["correct_code"].strip("()")
                gt_code = data[i]["gt"]["correct_code"]
            else:
                file_name = data[i]["images"][-1].split("_yellow/")[-1].split("/render_0.png")[0].replace("/" ,"_")
                if "cube" in file_name:
                    data_type = "cube"
                else:
                    data_type = "real"
                pred_code = data[i]["prediction"]
                gt_code = data[i]["messages"][-1]["content"]
            scad_path_pred = os.path.join(scad_dir, f"{data_type}/{id}_{file_name}_pred.scad")
            scad_path_gt = os.path.join(scad_dir, f"{data_type}/{id}_{file_name}_gt.scad")
            if not os.path.exists(os.path.dirname(scad_path_gt)):
                os.makedirs(os.path.dirname(scad_path_gt))
            if not os.path.exists(os.path.dirname(scad_path_pred)):
                os.makedirs(os.path.dirname(scad_path_pred))
            with open(scad_path_pred, "w") as f:
                f.write(pred_code)
            with open(scad_path_gt, "w") as f:
                f.write(gt_code)
            
            id += 1
    scad_files = find_files(scad_dir, ".scad")
    print(len(scad_files))

    with ThreadPoolExecutor(max_workers=100) as executor:  
        future_to_scad = {
            executor.submit(
                export_scad_to_stl, 
                scad_file, 
                scad_file.replace(".scad", ".stl").replace("scad/", "stl/")
            ): scad_file for scad_file in scad_files if not os.path.exists(
                scad_file.replace(".scad", ".stl").replace("scad/", "stl/")
            )
        }
        
        for future in tqdm(concurrent.futures.as_completed(future_to_scad), total=len(future_to_scad)):
            scad_file = future_to_scad[future]
            try:
                result = future.result()
                if result:
                    print(result)
            except Exception as e:
                print(f"处理文件 {scad_file} 时发生错误: {e}")
        
        
    stl_files = find_files(os.path.join(output_dir, "stl"), ".stl")
    print(len(scad_files))
    # nomalize stl
    for stl_file in tqdm(stl_files):
        try:
            scale_and_refine_stl(stl_file, stl_file, target_min=0, target_max=10)
        except Exception as e:
            print(f"Error: {e}")
            
    # Convert all STL files to PLY format
    for stl_file in tqdm(stl_files):
        output_ply_path = stl_file.replace(".stl", ".ply").replace("stl/", "ply/")
        if not os.path.exists(os.path.dirname(output_ply_path)):
            os.makedirs(os.path.dirname(output_ply_path))
        stl_to_ply(stl_file, output_ply_path)


