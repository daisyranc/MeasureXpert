import argparse
import importlib
import numpy as np
import tensorflow as tf
import open3d as o3d
import os
import shutil
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
import csv
import pandas as pd

os.environ["CUDA_VISIBLE_DEVICES"]="1"

def pcd2ply(pcd):
    mesh_file_path = '/home/pm/5117.ply'
    ply_input = o3d.io.read_triangle_mesh(mesh_file_path)
    input = pcd
    ply_input.vertices = input.points
    ply_input.compute_vertex_normals()
    return ply_input

def xyz2pcd(xyz):
 pcd = o3d.geometry.PointCloud()
 pcd.points = o3d.utility.Vector3dVector(xyz)
 return pcd

def point2pcd(pts):
    recon_body_pcd = o3d.geometry.PointCloud()
    recon_body_pcd.points = o3d.utility.Vector3dVector(pts)
    return recon_body_pcd

def point2ply(pts):
    pcd=point2pcd(pts)
    ply=pcd2ply(pcd)
    return ply

def compare_folders(folder_a, folder_b):
    # List all files in folder A
    files_a = set(os.path.splitext(file)[0] for file in os.listdir(folder_a))
    # List all files in folder B
    files_b = set(os.path.splitext(file)[0] for file in os.listdir(folder_b))
    # Find files that are in A but not in B
    difference = list(files_a - files_b)
    return difference

def chamfer_distance(cloud1, cloud2):
    # Create trees for fast nearest-neighbor search
    tree1 = cKDTree(cloud1)
    tree2 = cKDTree(cloud2)
    # Find shortest distances from each point in cloud1 to the nearest neighbor in cloud2
    dist1, _ = tree1.query(cloud2)
    dist2, _ = tree2.query(cloud1)
    # Calculate the Chamfer distance
    return np.mean(dist1) + np.mean(dist2)

def l2_distance(cloud1, cloud2):
    # Ensure the point clouds are arrays and have the same shape
    assert cloud1.shape == cloud2.shape, "Point clouds must have the same number of points and dimensions"
    # Compute the squared differences along each dimension
    diff = cloud1 - cloud2
    # Calculate the L2 distance for each pair of points
    distances = np.sqrt(np.sum(diff ** 2, axis=1))
    # Optionally return the total distance (sum) or mean distance
    return np.sum(distances), np.mean(distances)

def l1_distance(list1, list2):
    # Convert lists to numpy arrays
    arr1 = np.array(list1)
    arr2 = np.array(list2)
    # Ensure both lists have the same length
    assert arr1.shape == arr2.shape, "Both lists must have the same length"
    # Calculate the L1 distance
    return np.abs(arr1 - arr2)

def demo(model_type,checkpoints,input_f_root,demo_type="generation",data_type="test",vis=True,save=False):
    gt_root="/media/pm/Elements/partial2complete/centered_data/test/measurement/"
    waist_root=input_f_root

    #file_list = os.listdir(waist_root)
    file_list =os.listdir(waist_root)
    num=len(file_list)
    print(num)

    with open('offset_1_3norm_value.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        # 写入标题行
        if demo_type=="generation":
            writer.writerow(
                ["bust_value", "under_bust_value", "hip_value", "waist1_value", "waist2_value", "waist3_value",
                 "waist4_value", "waist5_value", "waist6_value",
                 "right_middle_value", "left_middle_value", "right_knee_value", "left_knee_value", "right_calf_value",
                 "left_calf_value", "right_upper_value",
                 "left_upper_value", "right_elbow_value", "left_elbow_value", "right_wrist_value", "left_wrist_value",
                 "cd", "id"])
        elif demo_type=="only_value":
            writer.writerow(
                ["bust_value", "under_bust_value", "hip_value", "waist1_value", "waist2_value", "waist3_value",
                 "waist4_value", "waist5_value", "waist6_value",
                 "right_middle_value", "left_middle_value", "right_knee_value", "left_knee_value", "right_calf_value",
                 "left_calf_value", "right_upper_value",
                 "left_upper_value", "right_elbow_value", "left_elbow_value", "right_wrist_value", "left_wrist_value",
                  "id"])
        for i in range(0, num):
            tf.reset_default_graph()
            print('No.', i)
            print('processing ', file_list[i])
            file_name=file_list[i].split(".")[0]

            parser = argparse.ArgumentParser()
            parser.add_argument('--model_type', default=model_type)
            parser.add_argument('--checkpoint', default=checkpoints)
            parser.add_argument('--batch_size', type=int, default=16)
            parser.add_argument('--num_input_points', type=int, default=2048)
            parser.add_argument('--num_head_points', type=int, default=1315)
            parser.add_argument('--num_right_arm_points', type=int, default=1301)
            parser.add_argument('--num_left_arm_points', type=int, default=1275)
            parser.add_argument('--num_right_leg_points', type=int, default=607)
            parser.add_argument('--num_left_leg_points', type=int, default=612)
            parser.add_argument('--num_body_points', type=int, default=1898)
            parser.add_argument('--num_gt_points', type=int, default=6890)
            parser.add_argument('--num_gt_landmark_points', type=int, default=10363)
            parser.add_argument('--num_value_points', type=int, default=36)

            # =====================read point cloud==========================================================
            input_waist_path = waist_root + "/" + file_list[i].split(".")[0] + ".pcd"

            waist_pcd = o3d.io.read_point_cloud(input_waist_path)

            waist_points = np.array(waist_pcd.points)

            args = parser.parse_args()

            input_t = tf.placeholder(tf.float32, (1, args.num_gt_landmark_points, 3))
            gt = tf.placeholder(tf.float32, (1, args.num_gt_landmark_points, 3))
            gt_value = tf.placeholder(tf.float32, (1, args.num_value_points, 3))

            model_module = importlib.import_module('.%s' % args.model_type, 'models')
            # inputs_front, inputs_back,

            model = model_module.Model(input_t, gt, gt_value, tf.constant(1.0))

            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            config.allow_soft_placement = True
            sess = tf.Session(config=config)

            saver = tf.train.Saver()
            saver.restore(sess, args.checkpoint)

            if demo_type == "generation":
                pred_points_t = sess.run(model.outputs_t, feed_dict={input_t: [waist_points]})[0]
                pcd_t = point2pcd(pred_points_t)
                bust_value = sess.run(model.bust_value, feed_dict={input_t: [waist_points]})[0]
                under_bust_value = sess.run(model.under_bust_value, feed_dict={input_t: [waist_points]})[0]
                hip_value = sess.run(model.hip_value, feed_dict={input_t: [waist_points]})[0]
                waist1_value = sess.run(model.waist1_value, feed_dict={input_t: [waist_points]})[0]
                waist2_value = sess.run(model.waist2_value, feed_dict={input_t: [waist_points]})[0]
                waist3_value = sess.run(model.waist3_value, feed_dict={input_t: [waist_points]})[0]
                waist4_value = sess.run(model.waist4_value, feed_dict={input_t: [waist_points]})[0]
                waist5_value = sess.run(model.waist5_value, feed_dict={input_t: [waist_points]})[0]
                waist6_value = sess.run(model.waist6_value, feed_dict={input_t: [waist_points]})[0]
                right_middle_value = sess.run(model.right_middle_value, feed_dict={input_t: [waist_points]})[0]
                left_middle_value = sess.run(model.left_middle_value, feed_dict={input_t: [waist_points]})[0]
                right_knee_value = sess.run(model.right_knee_value, feed_dict={input_t: [waist_points]})[0]
                left_knee_value = sess.run(model.left_knee_value, feed_dict={input_t: [waist_points]})[0]
                right_calf_value = sess.run(model.right_calf_value, feed_dict={input_t: [waist_points]})[0]
                left_calf_value = sess.run(model.left_calf_value, feed_dict={input_t: [waist_points]})[0]
                right_upper_value = sess.run(model.right_upper_value, feed_dict={input_t: [waist_points]})[0]
                left_upper_value = sess.run(model.left_upper_value, feed_dict={input_t: [waist_points]})[0]
                right_elbow_value = sess.run(model.right_elbow_value, feed_dict={input_t: [waist_points]})[0]
                left_elbow_value = sess.run(model.left_elbow_value, feed_dict={input_t: [waist_points]})[0]
                right_wrist_value = sess.run(model.right_wrist_value, feed_dict={input_t: [waist_points]})[0]
                left_wrist_value = sess.run(model.left_wrist_value, feed_dict={input_t: [waist_points]})[0]
                value_list = [bust_value, under_bust_value, hip_value, waist1_value, waist2_value, waist3_value,
                                waist4_value,
                                waist5_value, waist6_value, right_middle_value, left_middle_value
                    , right_knee_value, left_knee_value, right_calf_value, left_calf_value,right_upper_value, left_upper_value, right_elbow_value, left_elbow_value,
                                right_wrist_value, left_wrist_value]

                gt_value_path = gt_root + "girth/" + file_name + ".npy"
                gt_value = np.load(gt_value_path)
                gt_landmark_path = "/media/pm/Elements/partial2complete/centered_data/test/T/" + file_name + ".ply"
                gt_landmark = o3d.io.read_point_cloud(gt_landmark_path)
                error_value = l1_distance(value_list, gt_value).tolist()

                pcd = point2pcd(pred_points_t[-6890:, :])
                cd = chamfer_distance(np.asarray(pcd.points).reshape(-1, 3),
                                      np.asarray(gt_landmark.points).reshape(-1, 3))
                error_value.append(cd)
                error_value.append(int(file_name))
                writer.writerow(error_value)

            elif demo_type=="only_value":
                bust_value = sess.run(model.bust_value, feed_dict={input_t: [waist_points]})[0]
                under_bust_value = sess.run(model.under_bust_value, feed_dict={input_t: [waist_points]})[0]
                hip_value = sess.run(model.hip_value, feed_dict={input_t: [waist_points]})[0]
                waist1_value = sess.run(model.waist1_value, feed_dict={input_t: [waist_points]})[0]
                waist2_value = sess.run(model.waist2_value, feed_dict={input_t: [waist_points]})[0]
                waist3_value = sess.run(model.waist3_value, feed_dict={input_t: [waist_points]})[0]
                waist4_value = sess.run(model.waist4_value, feed_dict={input_t: [waist_points]})[0]
                waist5_value = sess.run(model.waist5_value, feed_dict={input_t: [waist_points]})[0]
                waist6_value = sess.run(model.waist6_value, feed_dict={input_t: [waist_points]})[0]
                right_middle_value = sess.run(model.right_middle_value, feed_dict={input_t: [waist_points]})[0]
                left_middle_value = sess.run(model.left_middle_value, feed_dict={input_t: [waist_points]})[0]
                right_knee_value = sess.run(model.right_knee_value, feed_dict={input_t: [waist_points]})[0]
                left_knee_value = sess.run(model.left_knee_value, feed_dict={input_t: [waist_points]})[0]
                right_calf_value = sess.run(model.right_calf_value, feed_dict={input_t: [waist_points]})[0]
                left_calf_value = sess.run(model.left_calf_value, feed_dict={input_t: [waist_points]})[0]
                right_upper_value = sess.run(model.right_upper_value, feed_dict={input_t: [waist_points]})[0]
                left_upper_value = sess.run(model.left_upper_value, feed_dict={input_t: [waist_points]})[0]
                right_elbow_value = sess.run(model.right_elbow_value, feed_dict={input_t: [waist_points]})[0]
                left_elbow_value = sess.run(model.left_elbow_value, feed_dict={input_t: [waist_points]})[0]
                right_wrist_value = sess.run(model.right_wrist_value, feed_dict={input_t: [waist_points]})[0]
                left_wrist_value = sess.run(model.left_wrist_value, feed_dict={input_t: [waist_points]})[0]
                value_list = [bust_value, under_bust_value, hip_value, waist1_value, waist2_value, waist3_value,
                                waist4_value,
                                waist5_value, waist6_value, right_middle_value, left_middle_value
                    , right_knee_value, left_knee_value, right_calf_value, left_calf_value,right_upper_value, left_upper_value, right_elbow_value, left_elbow_value,
                                right_wrist_value, left_wrist_value]

                gt_value_path = gt_root + "girth/" + file_name + ".npy"
                gt_value = np.load(gt_value_path)
                error_value = l1_distance(value_list, gt_value).tolist()
                error_value.append(int(file_name))
                writer.writerow(error_value)

            # ==================read GT============================================================
            if vis == True:
                if demo_type == "generation":
                    t_gt_root = "/media/pm/Elements/partial2complete/centered_data/" + data_type + "/T/" + \
                                file_list[i].split(".")[
                                    0] + ".ply"
                    t_gt_ply = o3d.io.read_point_cloud(t_gt_root)
                    t_gt_ply.paint_uniform_color([0, 1, 0])
                    print("=======T visulization==========")
                    o3d.visualization.draw_geometries([pcd_t] + [t_gt_ply])
                    o3d.visualization.draw_geometries([pcd_t])

            if save == True:
                if demo_type == "generation":
                    save_t_path = "/media/pm/Elements/shape_estimation/refine_T/" + "offset_1norm" + "/" + \
                                  file_list[i].split('.')[0] + ".pcd"
                    #   pcd_t=coarse_t_processing(pcd_t)
                    o3d.io.write_point_cloud(save_t_path, pcd_t)
            tf.get_default_graph().finalize()


if __name__ == '__main__':
    # root_w="/media/pm/Elements/waist2hip/centered_data/step2/waist/test_result/c2l_w/"
    # root_h = "/media/pm/Elements/waist2hip/centered_data/step2/hip/test_result/c2l_h/"
    model_type="fc_offset"
    data_type="test"
    checkpoint_path="/media/pm/Elements/shape_estimation/code_new/log/offset_1norm"+"/model-160000"
    # path_w=root_w+'fc_w'
    path_w="/media/pm/Elements/shape_estimation/coarse_T/"+data_type+"/"
    # path_h = root_h + 'fc_h'

    demo(model_type=model_type,checkpoints=checkpoint_path,input_f_root=path_w,demo_type="generation",data_type=data_type,vis=False,save=True)#,vis=False,save=T