import argparse
import importlib
import numpy as np
import tensorflow as tf
import open3d as o3d
import os
import shutil
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import pandas as pd

def pcd2ply(pcd):
    mesh_file_path = './body_seg/5117.ply'
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


def find_front_vertices(head,right_arm,left_arm,right_leg,left_leg, complete):
    head_pcd=o3d.io.read_point_cloud(head)
    right_arm=o3d.io.read_point_cloud(right_arm)
    left_arm=o3d.io.read_point_cloud(left_arm)
    right_leg=o3d.io.read_point_cloud(right_leg)
    left_leg=o3d.io.read_point_cloud(left_leg)
    complete_pcd=o3d.io.read_point_cloud(complete)

    complete_pcd.paint_uniform_color([0, 0, 1])
    threshold = 0.000001
    head_idx = []
    right_arm_idx = []
    left_arm_idx = []
    right_leg_idx = []
    left_leg_idx = []

    head_tree=o3d.geometry.KDTreeFlann(head_pcd)
    for i in range(len(complete_pcd.points)):
        [k, idx, dist] = head_tree.search_knn_vector_3d(complete_pcd.points[i], 1)
        if dist[0] < threshold:
            head_idx.append(i)
    head_idx = np.asarray(list(set(head_idx)))

    right_arm_tree=o3d.geometry.KDTreeFlann(right_arm)
    for i in range(len(complete_pcd.points)):
        [k, idx, dist] =right_arm_tree.search_knn_vector_3d(complete_pcd.points[i], 1)
        if dist[0] < threshold:
            right_arm_idx.append(i)
    right_arm_idx = np.asarray(list(set(right_arm_idx)))

    left_arm_tree=o3d.geometry.KDTreeFlann(left_arm)
    for i in range(len(complete_pcd.points)):
        [k, idx, dist] = left_arm_tree.search_knn_vector_3d(complete_pcd.points[i], 1)
        if dist[0] < threshold:
            left_arm_idx.append(i)
    left_arm_idx = np.asarray(list(set(left_arm_idx)))

    right_leg_tree=o3d.geometry.KDTreeFlann(right_leg)
    for i in range(len(complete_pcd.points)):
        [k, idx, dist] = right_leg_tree.search_knn_vector_3d(complete_pcd.points[i], 1)
        if dist[0] < threshold:
            right_leg_idx.append(i)
    right_leg_idx = np.asarray(list(set(right_leg_idx)))

    left_leg_tree=o3d.geometry.KDTreeFlann(left_leg)
    for i in range(len(complete_pcd.points)):
        [k, idx, dist] = left_leg_tree.search_knn_vector_3d(complete_pcd.points[i], 1)
        if dist[0] < threshold:
            left_leg_idx.append(i)
    left_leg_idx = np.asarray(list(set(left_leg_idx)))

    right_leg_idx=np.setdiff1d(right_leg_idx,left_leg_idx)
    #====check same id=====
    leg_idx=np.asarray(list(set(list(set(right_leg_idx))+list(set(left_leg_idx)))))

    indices = np.arange(len(complete_pcd.points))
    indices_outside_set = np.setdiff1d(indices,head_idx)
    indices_outside_set = np.setdiff1d(indices_outside_set, right_arm_idx)
    indices_outside_set = np.setdiff1d(indices_outside_set, left_arm_idx)
    indices_outside_set = np.setdiff1d(indices_outside_set, right_leg_idx)
    indices_outside_set = np.setdiff1d(indices_outside_set, left_leg_idx)

    selected_head = np.asarray(complete_pcd.points).reshape(-1, 3)[head_idx]
    selected_right_arm= np.asarray(complete_pcd.points).reshape(-1, 3)[right_arm_idx]
    selected_left_arm= np.asarray(complete_pcd.points).reshape(-1, 3)[left_arm_idx]
    selected_right_leg= np.asarray(complete_pcd.points).reshape(-1, 3)[right_leg_idx]
    selected_left_leg= np.asarray(complete_pcd.points).reshape(-1, 3)[left_leg_idx]
    unselected_body = np.asarray(complete_pcd.points).reshape(-1, 3)[indices_outside_set]

    seleced_head_pcd = xyz2pcd(selected_head)
    seleced_right_arm = xyz2pcd(selected_right_arm)
    seleced_left_arm= xyz2pcd(selected_left_arm)
    seleced_right_leg = xyz2pcd(selected_right_leg)
    seleced_left_leg= xyz2pcd(selected_left_leg)
    unseleced_body_pcd = xyz2pcd(unselected_body)
    seleced_head_pcd.paint_uniform_color([0.5, 0.5, 0.5])
    seleced_right_arm.paint_uniform_color([0.5, 0.5, 0.5])
    seleced_left_arm.paint_uniform_color([0.5, 0.5, 0.5])
    seleced_right_leg.paint_uniform_color([0.5, 0.5, 0.5])
    seleced_left_leg.paint_uniform_color([0.5, 0.5, 0.5])
    unseleced_body_pcd.paint_uniform_color([0.5, 0.5, 0.5])
    return head_idx,right_arm_idx,left_arm_idx,right_leg_idx,left_leg_idx,indices_outside_set
def recombination(head_idx,right_arm_idx,left_arm_idx,right_leg_idx,left_leg_idx,body_idx,
head_b_idx,right_arm_b_idx,left_arm_b_idx,right_leg_b_idx,left_leg_b_idx,
                  head_array,right_arm_array,left_arm_array,right_leg_array,left_leg_array,body_array):
    gt_path = "/home/pm/5117.ply"
    gt_pcd=o3d.io.read_point_cloud(gt_path)
    gt_np=np.asarray(gt_pcd.points).reshape(-1,3)
    new_array=np.empty_like(gt_np)
    new_array[head_idx]=head_array[29:]
    new_array[right_arm_idx]=right_arm_array[28:]
    new_array[left_arm_idx]=left_arm_array[26:]
    new_array[right_leg_idx]=right_leg_array[15:]
    new_array[left_leg_idx]=left_leg_array[20:]
    new_array[body_idx]=body_array[-1780:]

    new_array[head_b_idx] = (head_array[:29]+body_array[:29])/2
    new_array[right_arm_b_idx] = (right_arm_array[:28]+body_array[29:57])/2
    new_array[left_arm_b_idx] = (left_arm_array[:26]+body_array[57:83])/2
    new_array[right_leg_b_idx] = (right_leg_array[:15]+body_array[83:98])/2
    new_array[left_leg_b_idx] = (left_leg_array[:20]+body_array[98:118])/2
    return new_array

def demo(model_type,checkpoints,input_f_root,input_b_root,vis=True,save=False):
    root="./body_seg/"
    complete_path=root+"visualize_30.ply"
    head_path=root+"head_30.ply"
    right_arm_path=root+"right_arm_30.ply"
    left_arm_path=root+"left_arm_30.ply"
    right_leg_path=root+"right_leg_30.ply"
    left_leg_path=root+"left_leg_30.ply"

    # front_id_list,pts_front,pts_back=find_front_vertices(front_partial_path,gt_path)
    head_idx, right_arm_idx, left_arm_idx, right_leg_idx, left_leg_idx, body_idx = find_front_vertices(head=head_path,
                                                                                                                  right_arm=right_arm_path,left_arm=left_arm_path,
                                                                                                                  right_leg=right_leg_path,left_leg=left_leg_path,
                                                                                                                  complete=complete_path)
    waist_root=input_f_root
    hip_root=input_b_root

    file_list = os.listdir(waist_root)
    num=len(file_list)

    for i in range(0, num):
        tf.reset_default_graph()
        print('No.', i)
        print('processing ', file_list[i])

        parser = argparse.ArgumentParser()
        parser.add_argument('--model_type', default=model_type)
        parser.add_argument('--checkpoint', default=checkpoints)
        #parser.add_argument('--checkpoint', default='log/newfc_huber/model-300000')

        parser.add_argument('--num_input_points', type=int, default=2048)

        parser.add_argument('--num_head_points', type=int, default=1315)
        parser.add_argument('--num_right_arm_points', type=int, default=1301)
        parser.add_argument('--num_left_arm_points', type=int, default=1275)
        parser.add_argument('--num_right_leg_points', type=int, default=607)
        parser.add_argument('--num_left_leg_points', type=int, default=612)
        parser.add_argument('--num_body_points', type=int, default=1898)
        parser.add_argument('--num_gt_points', type=int, default=6890)
        # =====================read point cloud==========================================================
        input_waist_path=waist_root+"/"+file_list[i].split(".")[0]+"/0/0.pcd"
        input_hip_path=hip_root+"/"+file_list[i].split(".")[0]+"/1/1.pcd"

        waist_pcd=o3d.io.read_point_cloud(input_waist_path)
        hip_pcd=o3d.io.read_point_cloud(input_hip_path)
        waist_points=np.array(waist_pcd.points)
        hip_points=np.array(hip_pcd.points)

        args = parser.parse_args()

        partial_front = tf.placeholder(tf.float32, (1, None, 3))
        partial_back = tf.placeholder(tf.float32, (1, None, 3))
        posed_head_front = tf.placeholder(tf.float32, (1, args.num_head_points, 3))
        posed_right_arm_front = tf.placeholder(tf.float32, (1, args.num_right_arm_points, 3))
        posed_left_arm_front = tf.placeholder(tf.float32, (1, args.num_left_arm_points, 3))
        posed_right_leg_front = tf.placeholder(tf.float32, (1, args.num_right_leg_points, 3))
        posed_left_leg_front = tf.placeholder(tf.float32, (1, args.num_left_leg_points, 3))
        posed_body_front=tf.placeholder(tf.float32, (1, args.num_body_points, 3))
        posed_front=tf.placeholder(tf.float32, (1, args.num_gt_points, 3))
        posed_head_back = tf.placeholder(tf.float32, (1, args.num_head_points, 3))
        posed_right_arm_back = tf.placeholder(tf.float32, (1, args.num_right_arm_points, 3))
        posed_left_arm_back = tf.placeholder(tf.float32, (1, args.num_left_arm_points, 3))
        posed_right_leg_back = tf.placeholder(tf.float32, (1, args.num_right_leg_points, 3))
        posed_left_leg_back = tf.placeholder(tf.float32, (1, args.num_left_leg_points, 3))
        posed_body_back=tf.placeholder(tf.float32, (1, args.num_body_points, 3))
        posed_back=tf.placeholder(tf.float32, (1, args.num_gt_points, 3))
        t=tf.placeholder(tf.float32, (1, args.num_gt_points, 3))

        model_module = importlib.import_module('.%s' % args.model_type, 'models')

        model = model_module.Model(partial_front, partial_back,
                                   posed_head_front, posed_right_arm_front, posed_left_arm_front, posed_right_leg_front,
                                   posed_left_leg_front, posed_body_front, posed_front,
                                   posed_head_back, posed_right_arm_back, posed_left_arm_back, posed_right_leg_back,
                                   posed_left_leg_back, posed_body_back, posed_back,
                                   t,tf.constant(1.0))

        # model = model_module.Model(inputs_waist,inputs_hip, gt,gt1,gt2,gt3,gt4, tf.constant(1.0))

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        sess = tf.Session(config=config)

        saver = tf.train.Saver()
        saver.restore(sess, args.checkpoint)

        pred_points_t = sess.run(model.outputs_t, feed_dict={partial_front: [waist_points], partial_back: [hip_points]})[0]
        pred_points_front = \
        sess.run(model.outputs_front, feed_dict={partial_front: [waist_points], partial_back: [hip_points]})[0]
        pred_points_back = \
        sess.run(model.outputs_back, feed_dict={partial_front: [waist_points], partial_back: [hip_points]})[0]

        pred_points_head_front = sess.run(model.head_front, feed_dict={partial_front: [waist_points], partial_back: [hip_points]})[0]
        pred_points_right_arm_front = sess.run(model.right_arm_front, feed_dict={partial_front: [waist_points], partial_back: [hip_points]})[0]
        pred_points_left_arm_front = sess.run(model.left_arm_front, feed_dict={partial_front: [waist_points], partial_back: [hip_points]})[0]
        pred_points_right_leg_front = sess.run(model.right_leg_front, feed_dict={partial_front: [waist_points], partial_back: [hip_points]})[0]
        pred_points_left_leg_front = sess.run(model.left_leg_front, feed_dict={partial_front: [waist_points], partial_back: [hip_points]})[0]
        pred_points_body_front = sess.run(model.body_front, feed_dict={partial_front: [waist_points], partial_back: [hip_points]})[0]

        pred_points_head_back = sess.run(model.head_back, feed_dict={partial_front: [waist_points], partial_back: [hip_points]})[0]
        pred_points_right_arm_back = sess.run(model.right_arm_back, feed_dict={partial_front: [waist_points], partial_back: [hip_points]})[0]
        pred_points_left_arm_back = sess.run(model.left_arm_back, feed_dict={partial_front: [waist_points], partial_back: [hip_points]})[0]
        pred_points_right_leg_back = sess.run(model.right_leg_back, feed_dict={partial_front: [waist_points], partial_back: [hip_points]})[0]
        pred_points_left_leg_back = sess.run(model.left_leg_back, feed_dict={partial_front: [waist_points], partial_back: [hip_points]})[0]
        pred_points_body_back = sess.run(model.body_back, feed_dict={partial_front: [waist_points], partial_back: [hip_points]})[0]

        #==================read pts============================================================
        pcd_head_front=point2pcd(pred_points_head_front)
        pcd_right_arm_front = point2pcd(pred_points_right_arm_front)
        pcd_left_arm_front = point2pcd(pred_points_left_arm_front)
        pcd_right_leg_front = point2pcd(pred_points_right_leg_front)
        pcd_left_leg_front = point2pcd(pred_points_left_leg_front)
        pcd_body_front = point2pcd(pred_points_body_front[118:])

        pcd_head_back=point2pcd(pred_points_head_back)
        pcd_right_arm_back = point2pcd(pred_points_right_arm_back)
        pcd_left_arm_back = point2pcd(pred_points_left_arm_back)
        pcd_right_leg_back = point2pcd(pred_points_right_leg_back)
        pcd_left_leg_back = point2pcd(pred_points_left_leg_back)
        pcd_body_back = point2pcd(pred_points_body_back[118:])

        pcd_front=point2pcd(pred_points_front)
        pcd_back=point2pcd(pred_points_back)
        pcd_t=point2ply(pred_points_t)
        # ======recombine mesh====================
        head_idx = np.loadtxt("./body_seg/head.txt", dtype=int)
        print(head_idx.shape[0])
        ra_idx = np.loadtxt("./body_seg/ra.txt", dtype=int)
        la_idx = np.loadtxt("./body_seg/la.txt", dtype=int)
        rl_idx = np.loadtxt("./body_seg/rl.txt", dtype=int)
        ll_idx = np.loadtxt("./body_seg/ll.txt", dtype=int)

        head_b_idx = np.loadtxt("./body_seg/head_b.txt", dtype=int)
        ra_b_idx = np.loadtxt("./body_seg/ra_b.txt", dtype=int)
        la_b_idx = np.loadtxt("./body_seg/la_b.txt", dtype=int)
        rl_b_idx = np.loadtxt("./body_seg/rl_b.txt", dtype=int)
        ll_b_idx = np.loadtxt("./body_seg/ll_b.txt", dtype=int)
        new_body_front = recombination(head_idx=head_idx, right_arm_idx=ra_idx, left_arm_idx=la_idx, right_leg_idx=rl_idx,
                                 left_leg_idx=ll_idx,
                                 body_idx=body_idx, head_b_idx=head_b_idx, right_arm_b_idx=ra_b_idx,
                                 left_arm_b_idx=la_b_idx,
                                 right_leg_b_idx=rl_b_idx, left_leg_b_idx=ll_b_idx,
                                 head_array=pred_points_head_front, right_arm_array=pred_points_right_arm_front,
                                 left_arm_array=pred_points_left_arm_front,
                                 right_leg_array=pred_points_right_leg_front, left_leg_array=pred_points_left_leg_front,
                                 body_array=pred_points_body_front)
        new_body_back = recombination(head_idx=head_idx, right_arm_idx=ra_idx, left_arm_idx=la_idx, right_leg_idx=rl_idx,
                                 left_leg_idx=ll_idx,
                                 body_idx=body_idx, head_b_idx=head_b_idx, right_arm_b_idx=ra_b_idx,
                                 left_arm_b_idx=la_b_idx,
                                 right_leg_b_idx=rl_b_idx, left_leg_b_idx=ll_b_idx,
                                 head_array=pred_points_head_back, right_arm_array=pred_points_right_arm_back,
                                 left_arm_array=pred_points_left_arm_back,
                                 right_leg_array=pred_points_right_leg_back, left_leg_array=pred_points_left_leg_back,
                                 body_array=pred_points_body_back)
        body_ply_front = point2ply(new_body_front)
        body_ply_back = point2ply(new_body_back)

        #==================read GT============================================================
        if vis==True:
            pcd_head_front.paint_uniform_color([1,0,0])
            pcd_right_arm_front.paint_uniform_color([1,0,0.5])
            pcd_left_arm_front.paint_uniform_color([1,0,1])
            pcd_right_leg_front.paint_uniform_color([1,0.3,0.3])
            pcd_left_leg_front.paint_uniform_color([1,0.5,0.7])
            pcd_body_front.paint_uniform_color([1,0.5,0.2])

            pcd_head_back.paint_uniform_color([1,0,0])
            pcd_right_arm_back.paint_uniform_color([1,0,0.5])
            pcd_left_arm_back.paint_uniform_color([1,0,1])
            pcd_right_leg_back.paint_uniform_color([1,0.3,0.3])
            pcd_left_leg_back.paint_uniform_color([1,0.5,0.7])
            pcd_body_back.paint_uniform_color([1,0.5,0.2])

            front_gt_root ="/media/pm/Elements/partial2complete/centered_data/test/Posed_GT/"+file_list[i].split(".")[0]+"/0/0.ply"
            back_gt_root ="/media/pm/Elements/partial2complete/centered_data/test/Posed_GT/"+file_list[i].split(".")[0]+"/1/1.ply"
          #  t_gt_root="/media/pm/Elements/partial2complete/centered_data/test/TPosed_GT/"+file_list[i].split(".")[0]+"/1/1.ply"
            # t_gt_root="/media/pm/Elements/partial2complete/centered_data/test/TPosed_GT/"+file_list[i].split(".")[0]+"/0/0.ply"
            t_gt_root="/media/pm/Elements/partial2complete/centered_data/test/T/"+file_list[i].split(".")[0]+".ply"

            front_gt_ply=o3d.io.read_point_cloud(front_gt_root)
            front_gt_ply.paint_uniform_color([0,1,0])
            back_gt_ply=o3d.io.read_point_cloud(back_gt_root)
            back_gt_ply.paint_uniform_color([0,1,0])
            t_gt_ply=o3d.io.read_point_cloud(t_gt_root)
            t_gt_ply.paint_uniform_color([0,1,0])

            print("=======front visulization=======")
            o3d.visualization.draw_geometries(
                 [pcd_head_front] + [pcd_right_arm_front] + [pcd_left_arm_front] + [pcd_right_leg_front] + [pcd_left_leg_front] + [pcd_body_front]+[front_gt_ply])
            o3d.visualization.draw_geometries([body_ply_front])
            print("=======back visulization=======")
            o3d.visualization.draw_geometries(
                 [pcd_head_back] + [pcd_right_arm_back] + [pcd_left_arm_back] + [pcd_right_leg_back] + [pcd_left_leg_back] + [pcd_body_back]+[back_gt_ply])
            o3d.visualization.draw_geometries([body_ply_back])
            print("=======T visulization==========")
            o3d.visualization.draw_geometries([pcd_t]+[t_gt_ply])

            #o3d.visualization.draw_geometries([pcd_head]+[pcd_right_arm]+[pcd_left_arm]+[pcd_right_leg]+[pcd_left_leg]+[pcd_body])
            # o3d.visualization.draw_geometries(
            #      [pcd_head] + [pcd_right_arm] + [pcd_left_arm] + [pcd_right_leg] + [pcd_left_leg] + [pcd_body]+[gt])
            # o3d.visualization.draw_geometries([pcd_smpl])
            #
            #
            # o3d.visualization.draw_geometries([body_ply])

        if save==True:
            total_root="/media/pm/Elements/partial2complete/new_test/"
            # save_front_path=total_root+"front/"+ file_list[i].split('.')[0]+".ply"
            # save_back_path=total_root+"back/"+ file_list[i].split('.')[0]+".ply"
            save_front_path = total_root + "front/" + file_list[i].split('.')[0] + ".ply"
            save_back_path = total_root + "back/" + file_list[i].split('.')[0] + ".ply"
            save_t_path=total_root+"t/"+ file_list[i].split('.')[0]+".ply"

            o3d.io.write_triangle_mesh(save_front_path,body_ply_front)
            o3d.io.write_triangle_mesh(save_back_path,body_ply_back)
            o3d.io.write_triangle_mesh(save_t_path,pcd_t)
        tf.get_default_graph().finalize()

if __name__ == '__main__':
    model_type="step1"
    checkpoint_path="./log/step1"+"/model-300000"

    path_front="/media/pm/Elements/partial2complete/centered_data/test/input_pcd"
    path_back="/media/pm/Elements/partial2complete/centered_data/test/input_pcd"
    demo(model_type=model_type,checkpoints=checkpoint_path,input_f_root=path_w,input_b_root=path_h,vis=False,save=True)#,vis=False,save=True
