import numpy as np

import os
#import pcl
import random
import time

# usage: blender -b -P io_util.py

import open3d as o3d


def read_pcd(filename):
    pcd = o3d.io.read_point_cloud(filename)
    return np.array(pcd.points)

def getSample(data,sample_num):
    total_num=data.shape[0]
    rand=np.random.randint(0,total_num,sample_num)
    # rand=np.random.randint(0,6890,6890)
    resSC=data[rand]
    return np.asanyarray(resSC).reshape(-1,3)

def save_pcd(filename, points):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    o3d.io.write_point_cloud(filename, pcd)


def obj_to_pcd(objfile,pcdfile):
    obj=pcl.load(objfile)
    pcl.save(obj,pcdfile,format='pcd',binary=True)

def ply2pcd(plyfile,pcdfile):
    ply=pcl.load(plyfile)
    pcl.save(ply,pcdfile,format='pcd',binary=False)


def pcd_to_ply(pcdfile,objfile):
    pcd=pcl.load(pcdfile)
    pcl.save(pcd,objfile,format='ply',binary=False)


def multi_obj_to_pcd():
    # for complete data, required by lmdb_writer.py
    obj_root = '/home/pengpeng/projects/NakedNet/render/BUVG/step2/body'
    pcd_root = '/home/pengpeng/projects/NakedNet/render/BUVG/step3/AllPcd/body'
    obj_ids = [x for x in os.listdir(obj_root)]

    for obj_id in obj_ids:
        pcd_id = obj_id.split('.')[0] + '.pcd'
        pcd_id = os.path.join(pcd_root, pcd_id)
        #obj_id = os.path.join(obj_root, obj_id, 'model.obj')
        obj_id = os.path.join(obj_root, obj_id)
        print('obj_id: ', obj_id)
        print('pcd_id: ', pcd_id)
        obj_to_pcd(obj_id, pcd_id)

def multi_ply_to_pcd():
    ply_root = '/home/pengpeng/test_data/body'
    pcd_root = '/home/pengpeng/test_data/blender_dressed/body_pcd'
    ply_list=os.listdir(ply_root)
    for id in ply_list:
        #print('converting '+ply_id)
        ply_id=ply_root+'/'+id
        pcd_id=pcd_root+'/'+id.split('.')[0]+'.pcd'
        ply2pcd(ply_id, pcd_id)


def multi_ply_to_obj():
    import bpy
    ply_root = '/home/pengpeng/test_data/dressed'
    obj_root = '/home/pengpeng/test_data/dressed_obj'
    ply_list=os.listdir(ply_root)
    for id in ply_list:
        print('converting '+id)
        ply_id = ply_root + '/' + id
        os.mkdir(obj_root+'/'+id.split('.')[0])
        outpath=obj_root+'/'+id.split('.')[0]+'/'+'model.obj'
        obj_id=outpath

        # Clean up
        bpy.ops.object.delete()
        for m in bpy.data.meshes:
            bpy.data.meshes.remove(m)
        for m in bpy.data.materials:
            m.user_clear()
            bpy.data.materials.remove(m)

        # Import mesh model
        bpy.ops.import_mesh.ply(filepath=ply_id)
        bpy.ops.transform.rotate(value=np.pi / 2, axis=(1, 0, 0))
        bpy.ops.export_scene.obj(filepath=obj_id, check_existing=True, axis_forward='-Z', axis_up='Y',
                                 filter_glob="*.obj", use_selection=False, use_animation=False,
                                 use_mesh_modifiers=True, use_edges=False, use_smooth_groups=False,
                                 use_smooth_groups_bitflags=False, use_normals=False, use_uvs=False,
                                 use_materials=False, use_triangles=False, use_nurbs=False, use_vertex_groups=False,
                                 use_blen_objects=True, group_by_object=False, group_by_material=False,
                                 keep_vertex_order=False, global_scale=1, path_mode='AUTO')


def merge_mutlti_pcd():
    input_root='/media/pengpeng/5121bd45-76e6-4694-92ac-cc439c45fde6/APose_BUG/male/blender/pcd'
    save_root='/media/pengpeng/5121bd45-76e6-4694-92ac-cc439c45fde6/APose_BUG/male/blender/dressedpcd'
    # input_root = '/media/pengpeng/0bc58d8e-e5a0-4fd7-96a2-fae9bc2eaf0b/SMPL_dataset/male/blender/pcd'
    # save_root = '/media/pengpeng/0bc58d8e-e5a0-4fd7-96a2-fae9bc2eaf0b/SMPL_dataset/male/blender/dressedpcd'
    model_list=os.listdir(input_root)
    scan_num=4
    for model in model_list:
        merge_pcd=PointCloud()
        for i in range(scan_num):
            file=input_root+'/'+model+'/'+str(i)+'.pcd'
            pcd=read_point_cloud(file)
            merge_pcd=merge_pcd+pcd

        write_point_cloud(save_root+'/'+model+'.pcd',merge_pcd)


def display_pcd(filename):
    pcd=read_point_cloud(filename)
    print(pcd)
    draw_geometries([pcd])


#using blender to scale the obj file for the purpose of normalization
def obj_normalization():
    import bpy, os

    InputFileDirectory = "/home/pengpeng/projects/pcn/render/body/valid"
    OutputFileDirectory = "/home/pengpeng/projects/pcn/render/body/valid_normalization"
    Files = os.listdir(InputFileDirectory)
    for file in Files:
        if file.endswith(".obj"):
            bpy.ops.import_scene.obj(filepath=(InputFileDirectory + "/" + file)) # 导入模型
            #bpy.ops.transform.rotate(axis=(-90, 0, 0))  # 以x为轴旋转-90度
            #bpy.ops.transform.resize(value=(0.01, 0.01, 0.01))  # for data normalization
            str = file.split('.')
            #OutPath = os.str[0] + '.obj'
            os.mkdir(OutputFileDirectory+'/'+str[0])
            OutPath = str[0] + '/model.obj'
            bpy.ops.export_scene.obj(filepath=(OutputFileDirectory + "/" + OutPath))  # 输出模型
            # 删除模型
            override = bpy.context.copy()
            override['selected_bases'] = list(bpy.context.scene.object_bases)
            bpy.ops.object.delete(override)

            print(file + " is over")

#using blender to scale the ply file for the ECCV2016 dataset
def ply_normalization():
    import bpy, os


    InputFileDirectory = "/home/pengpeng/projects/NakedNet/render/NakedDressedData/dressed"
    OutputFileDirectory = "/home/pengpeng/projects/NakedNet/render/NakedDressedData/dressed_obj_normalization"
    Files=os.listdir(InputFileDirectory)

    for file in Files:
        filepath=InputFileDirectory+'/'+file+'/mesh'
        subfiles=os.listdir(filepath)
        for subfile in subfiles:
            if subfile.endswith(".ply"):
                bpy.ops.import_mesh.ply(filepath=(filepath+'/'+subfile))  # 导入模型
                # bpy.ops.transform.rotate(axis=(-90, 0, 0))  # 以x为轴旋转-90度
                bpy.ops.transform.resize(value=(0.01, 0.01, 0.01))  # for data normalization

                # OutPath = os.str[0] + '.obj'
                #os.mkdir(OutputFileDirectory + '/' + str[0])
                str=subfile.split('.')
                os.mkdir(OutputFileDirectory + '/' + file+'_'+str[0])
                OutPath = file+'_'+str[0]+'/model.obj'
                bpy.ops.export_scene.obj(filepath=(OutputFileDirectory + "/" + OutPath))  # 输出模型
                # 删除模型
                override = bpy.context.copy()
                override['selected_bases'] = list(bpy.context.scene.object_bases)
                bpy.ops.object.delete(override)

                print(file + " is over")

    InputFileDirectory = "/home/pengpeng/projects/NakedNet/render/NakedDressedData/naked"
    OutputFileDirectory = "/home/pengpeng/projects/NakedNet/render/NakedDressedData/naked_obj_normalization"
    Files = os.listdir(InputFileDirectory)

    for file in Files:
        filepath = InputFileDirectory + '/' + file
        subfiles = os.listdir(filepath)
        for subfile in subfiles:
            if subfile.endswith(".ply"):
                bpy.ops.import_mesh.ply(filepath=(filepath + '/' + subfile))  # 导入模型
                # bpy.ops.transform.rotate(axis=(-90, 0, 0))  # 以x为轴旋转-90度
                bpy.ops.transform.resize(value=(0.01, 0.01, 0.01))  # for data normalization

                # OutPath = os.str[0] + '.obj'
                # os.mkdir(OutputFileDirectory + '/' + str[0])
                str = subfile.split('.')
                os.mkdir(OutputFileDirectory + '/' + file + '_' + str[0])
                OutPath = file + '_' + str[0] + '/model.obj'
                bpy.ops.export_scene.obj(filepath=(OutputFileDirectory + "/" + OutPath))  # 输出模型
                # 删除模型
                override = bpy.context.copy()
                override['selected_bases'] = list(bpy.context.scene.object_bases)
                bpy.ops.object.delete(override)

                print(file + " is over")


def rename_pcd(): # for convinence, we just modify the name the body and dressed body into the same name
    root='/home/pengpeng/projects/NakedNet/render/BUVG/train/body_train'
    #save='/home/pengpeng/projects/NakedNet/render/BUVG/train/rename_body_train'
    model_list=os.listdir(root)

    for model_id in model_list:
        str=model_id.split('.')[0]
        str=str.split('.')[0]
        str=str.split('_')
        save_id='0'+str[1]+'_Wide_clothes.pcd'

        oldname=root+'/'+model_id
        newname=root+'/'+save_id
        os.rename(oldname,newname)

# for BUVG
def Move_Rename_files():# save all the obj files as such a format: model_id/model.obj
    # move the files from source root to the target root
    source_root='/home/pengpeng/projects/NakedNet/render/BUVG/dressed_obj'
    target_root='/home/pengpeng/projects/NakedNet/render/BUVG/dressedbody'

    file_list=os.listdir(source_root)

    for file in file_list:
        model_id=file.split('.')[0]# model_id folder
        os.mkdir(target_root+'/'+model_id)
        save_path=target_root+'/'+model_id

        file = source_root + '/' + file  # source file path
        os.system('cp'+' '+file+' '+save_path) # move files from source_root to the target_root


    # rename the files in the target root
    model_list=os.listdir(target_root)
    for model_id in model_list:

        filepath=target_root+'/'+model_id
        os.system('cd'+' '+filepath)
        filename = filepath + '/'+model_id+'.obj'
        savefile=filepath+'/'+'model.obj'
        os.system('mv'+' '+filename+' '+savefile)

# for BUVG
def obj_shuffle():# shuffle the data to divide it to the train and validation
    body_root='/home/pengpeng/projects/NakedNet/render/BUVG/step1/body'
    dressed_root='/home/pengpeng/projects/NakedNet/render/BUVG/step1/dressed'

    save_train='/home/pengpeng/projects/NakedNet/render/BUVG/step2_3/train'
    save_valid='/home/pengpeng/projects/NakedNet/render/BUVG/step2_3/valid'

    model_list=os.listdir(body_root)
    random.shuffle(model_list)
    train_list=model_list[:int(len(model_list)*0.9)]
    valid_list=model_list[int(len(model_list)*0.9):]

    for file in train_list:
        #save body
        source_body=body_root+'/'+file
        target_body=save_train+'/body/'+file
        os.system('mv'+' '+source_body+' '+target_body)

        #save dressed
        source_dressed=dressed_root+'/'+file

        dressed_id=file.split('.')[0]
        newfolder=save_train+'/dressed/'+dressed_id
        os.system('mkdir'+' '+newfolder)
        target_dressed=newfolder+'/'+'model.obj'

        os.system('mv'+' '+source_dressed+' '+target_dressed)

    for file in valid_list:
        # save body
        source_body = body_root + '/' + file
        target_body = save_valid + '/body/' + file
        os.system('mv' + ' ' + source_body + ' ' + target_body)

        # save dressed
        source_dressed=dressed_root+'/'+file

        dressed_id=file.split('.')[0]
        newfolder=save_valid+'/dressed/'+dressed_id
        os.system('mkdir'+' '+newfolder)
        target_dressed=newfolder+'/'+'model.obj'

        os.system('mv'+' '+source_dressed+' '+target_dressed)


def devide_train_and_valid():# split the dataset into training, valid, and testing to 0.9,0.5,0.5
    root='/media/pengpeng/5121bd45-76e6-4694-92ac-cc439c45fde6/BUG_partial_scan/'

    dressed_root = root+'normalized_dressed_pcd'
    body_root = root + 'normalized_body_pcd'
    #offset_root=root+'male_longsleeve_offset_normalizaedpcd'

    save_train = root+'train'
    save_valid = root+'valid'
    save_testing=root+'test'


    model_list = os.listdir(dressed_root)

    model_list=sorted(model_list)

    #random.shuffle(model_list)

    train_list = model_list[:int(len(model_list) * 0.97)]
    valid_list = model_list[int(len(model_list) * 0.97):int(len(model_list) * 0.99)]
    testing_list = model_list[int(len(model_list) * 0.99):]

    #train_list = model_list[:int(len(model_list) * 0.9)]
    #valid_list = model_list[int(len(model_list) * 0.9):int(len(model_list) * 0.95)]
    #testing_list = model_list[int(len(model_list) * 0.95):]

    for file in train_list:
        #save body
        source_body=body_root+'/'+file
        target_body=save_train+'/body/'+file
        os.system('mv'+' '+source_body+' '+target_body)

        # save dressed
        source_dressed = dressed_root + '/' + file
        dressed_id = file.split('.')[0]
        newfolder = save_train + '/dressed/' + dressed_id
        os.system('mkdir' + ' ' + newfolder)
        target_dressed = newfolder + '/' + '0.pcd'
        os.system('mv' + ' ' + source_dressed + ' ' + target_dressed)

        # save offset
        # source_offset = offset_root + '/' + file.split('.')[0] + '.txt'
        # target_offset = save_train + '/offset/' + file.split('.')[0] + '.txt'
        # os.system('mv' + ' ' + source_offset + ' ' + target_offset)



    for file in valid_list:
        # save body
        source_body = body_root + '/' + file
        target_body = save_valid + '/body/' + file
        os.system('mv' + ' ' + source_body + ' ' + target_body)

        # save dressed
        source_dressed = dressed_root + '/' + file
        dressed_id = file.split('.')[0]
        newfolder = save_valid + '/dressed/' + dressed_id
        os.system('mkdir' + ' ' + newfolder)
        target_dressed = newfolder + '/' + '0.pcd'
        os.system('mv' + ' ' + source_dressed + ' ' + target_dressed)

        # save offset
        # source_offset = offset_root + '/' + file.split('.')[0] + '.txt'
        # target_offset = save_valid + '/offset/' + file.split('.')[0] + '.txt'
        # os.system('mv' + ' ' + source_offset + ' ' + target_offset)

    for file in testing_list:
        # save body
        source_body = body_root + '/' + file
        target_body = save_testing + '/body/' + file
        os.system('mv' + ' ' + source_body + ' ' + target_body)

        # save dressed
        source_dressed = dressed_root + '/' + file
        dressed_id = file.split('.')[0]
        newfolder = save_testing + '/dressed/' + dressed_id
        os.system('mkdir' + ' ' + newfolder)
        target_dressed = newfolder + '/' + '0.pcd'
        os.system('mv' + ' ' + source_dressed + ' ' + target_dressed)

        # save offset
        # source_offset = offset_root + '/' + file.split('.')[0] + '.txt'
        # target_offset = save_testing + '/offset/' + file.split('.')[0] + '.txt'
        # os.system('mv' + ' ' + source_offset + ' ' + target_offset)

def obj_Half():# for trainning efficiency, select half of the data for training, valididation and testing
    body_root='/home/pengpeng/projects/NakedNet/render/BUVG/step1/body'
    dressed_root='/home/pengpeng/projects/NakedNet/render/BUVG/step1/dressed'

    save_body='/home/pengpeng/projects/NakedNet/render/BUVG/step2'
    save_dressed='/home/pengpeng/projects/NakedNet/render/BUVG/step2'

    model_list=os.listdir(body_root)

    model_list.sort()

    count=0
    for file in model_list:
        count = count + 1
        if count%2==0:
            # save body
            source_body = body_root + '/' + file
            target_body = save_body + '/body/' + file
            os.system('mv' + ' ' + source_body + ' ' + target_body)

            # save dressed
            source_dressed = dressed_root + '/' + file

            print(source_dressed)

            dressed_id = file.split('.')[0]
            newfolder = save_dressed + '/dressed/' + dressed_id
            os.system('mkdir' + ' ' + newfolder)
            target_dressed = newfolder + '/' + 'model.obj'

            os.system('mv' + ' ' + source_dressed + ' ' + target_dressed)


def random_pose():
    angle_x = np.random.uniform() * 2 * np.pi
    angle_y = np.random.uniform() * 2 * np.pi
    angle_z = np.random.uniform() * 2 * np.pi
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(angle_x), -np.sin(angle_x)],
                   [0, np.sin(angle_x), np.cos(angle_x)]])
    Ry = np.array([[np.cos(angle_y), 0, np.sin(angle_y)],
                   [0, 1, 0],
                   [-np.sin(angle_y), 0, np.cos(angle_y)]])
    Rz = np.array([[np.cos(angle_z), -np.sin(angle_z), 0],
                   [np.sin(angle_z), np.cos(angle_z), 0],
                   [0, 0, 1]])
    R = np.dot(Rz, np.dot(Ry, Rx))
    # Set camera pointing to the origin and 1 unit away from the origin
    t = np.expand_dims(R[:, 2], 1)
    pose = np.concatenate([np.concatenate([R, t], 1), [[0, 0, 0, 1]]], 0)
    return pose

def add_transformation(): # add random rotation and small translation to the data to enforce the robustness
    body_root='/home/pengpeng/projects/NakedNet/render/BUVG/step3/AllPcd/body'
    dressed_root='/home/pengpeng/projects/NakedNet/render/BUVG/step3/AllPcd/dressed'

    save_body='/home/pengpeng/projects/NakedNet/render/BUVG/step3/random_pose_pcd/body'
    save_dressed='/home/pengpeng/projects/NakedNet/render/BUVG/step3/random_pose_pcd/dressed'
    save_pose='/home/pengpeng/projects/NakedNet/render/BUVG/step3/random_pose_pcd/pose'

    model_list=os.listdir(body_root)

    for i in range(len(model_list)):
        for pose_number in range(3):
            body_pcd = read_point_cloud(body_root + '/' + model_list[i])

            transformation = random_pose()
            body_pcd.transform(transformation)

            write_point_cloud(save_body + '/' +str(pose_number)+'_'+model_list[i], body_pcd)

            dressed_pcd = read_point_cloud(dressed_root + '/'+ model_list[i])
            dressed_pcd.transform(transformation)
            write_point_cloud(save_dressed + '/' +str(pose_number)+'_'+ model_list[i], dressed_pcd)

            np.savetxt(save_pose + '/' +str(pose_number)+'_'+ model_list[i].split('.')[0] + '.txt', transformation)


def remove():# remove the seleced files in one folder
    root='/media/pengpeng/Elements/BUG_dataset/body_pcd'
    remove_ids=[28309,28308,28422,28423,28424]
    file_list=os.listdir(root)
    print('origianl: ',len(file_list))
    for i in remove_ids:
        os.system('rm %s/male_%d.pcd'%(root,i))

    file_list=os.listdir(root)
    print('after: ',len(file_list))

def move_hpp():
    root='/home/pengpeng/projects/NakedNet/render/BUVG/step3/train/dressed'
    source_root='/home/pengpeng/test_data/poses'
    target_root='/home/pengpeng/projects/NakedNet/render/BUVG/step3/train/poses'
    file_list=os.listdir(root)
    for id in file_list:
        os.system('cp /home/pengpeng/test_data/poses/%s.txt'
                  ' /home/pengpeng/projects/NakedNet/render/BUVG/step3/train/poses/%s.txt'%(id,id))


def calculate_PointConfidences():
    dressed_root='/media/pengpeng/0bc58d8e-e5a0-4fd7-96a2-fae9bc2eaf0b/SMPL_dataset/male/blender/dressedpcd'
    body_root='/media/pengpeng/0bc58d8e-e5a0-4fd7-96a2-fae9bc2eaf0b/SMPL_dataset/male/body'

    saved_root='/media/pengpeng/0bc58d8e-e5a0-4fd7-96a2-fae9bc2eaf0b/SMPL_dataset/male/pointconfidences/open_clothing'

    file_list=os.listdir(dressed_root)
    for id in file_list:
        dressed_pcd=read_point_cloud(dressed_root+'/'+id)
        body_mesh=read_triangle_mesh(body_root+'/'+id.split('.')[0]+'.ply')

        body_pcd=PointCloud()
        body_pcd.points=body_mesh.vertices

        body_pcd_tree = geometry.KDTreeFlann(body_pcd)

        pointconfidences=np.zeros((len(dressed_pcd.points),3))


        for i in range(len(dressed_pcd.points)):
            [k, idx, dist] = body_pcd_tree.search_knn_vector_3d(dressed_pcd.points[i], 1)
            indx=np.asarray(idx)[0]
            pointconfidences[i,:]=body_pcd.points[indx]-dressed_pcd.points[i] # offset= body - garment

        np.savetxt(saved_root+'/'+id.split('.')[0]+'.pc',pointconfidences,fmt='%6f')



def set_rotation(rot_x,rot_y,rot_z):
    angle_x = rot_x
    angle_y = rot_y
    angle_z = rot_z
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(angle_x), -np.sin(angle_x)],
                   [0, np.sin(angle_x), np.cos(angle_x)]])
    Ry = np.array([[np.cos(angle_y), 0, np.sin(angle_y)],
                   [0, 1, 0],
                   [-np.sin(angle_y), 0, np.cos(angle_y)]])
    Rz = np.array([[np.cos(angle_z), -np.sin(angle_z), 0],
                   [np.sin(angle_z), np.cos(angle_z), 0],
                   [0, 0, 1]])
    R = np.dot(Rz, np.dot(Ry, Rx))
    # Set camera pointing to the origin and 1 unit away from the origin
    t = np.expand_dims(R[:, 2], 1)

    rotation = np.concatenate([np.concatenate([R, t], 1), [[0, 0, 0, 1]]], 0)

    # remove the translation
    rotation[0,-1]=0
    rotation[1, -1] = 0
    rotation[2, -1] = 0

    return rotation

# this function adds random small rotation to the input to enhance its robustness
def add_random_rotation():
    source_dressed_root='/media/pengpeng/5121bd45-76e6-4694-92ac-cc439c45fde6/BUG_partial_scan/centred_dressed_pcd'
    source_body_root='/media/pengpeng/5121bd45-76e6-4694-92ac-cc439c45fde6/BUG_partial_scan/centered_body_pcd'

    save_dressed_root='/media/pengpeng/5121bd45-76e6-4694-92ac-cc439c45fde6/BUG_partial_scan/rotated_male_longsleeve_dressed_pcd'
    save_body_root='/media/pengpeng/5121bd45-76e6-4694-92ac-cc439c45fde6/BUG_partial_scan/rotated_densebody_pcd'



    file_list=os.listdir(source_body_root)
    for id in file_list:
        x_rot=np.random.uniform(-np.pi / 12,np.pi / 12)
        y_rot=np.random.uniform(-np.pi / 12,np.pi / 12)
        z_rot = np.random.uniform(-np.pi / 12, np.pi / 12)
        R = set_rotation(x_rot,y_rot,z_rot)

        dressed_pcd=read_point_cloud(source_dressed_root+'/'+id)
        dressed_points=np.array(dressed_pcd.points).T

        body_pcd=read_point_cloud(source_body_root+'/'+id)
        body_points=np.array(body_pcd.points).T

        dressed_points = np.dot(R, np.concatenate([dressed_points, np.ones((1, dressed_points.shape[1]))], 0)).T[:, :3]
        body_points = np.dot(R, np.concatenate([body_points, np.ones((1, body_points.shape[1]))], 0)).T[:, :3]
        dressed_pcd.points = Vector3dVector(dressed_points)
        body_pcd.points=Vector3dVector(body_points)


        write_point_cloud(save_dressed_root+'/'+id,dressed_pcd)
        write_point_cloud(save_body_root+'/'+id,body_pcd)



# center the pcd file to the origin point
def center3d():
    from normalization import center_PCD
    source_dressed_root='/media/pengpeng/5121bd45-76e6-4694-92ac-cc439c45fde6/BUG_partial_scan/partial_male_longsleeve_dressed_input'
    source_body_root='/media/pengpeng/5121bd45-76e6-4694-92ac-cc439c45fde6/BUG_partial_scan/male_densebody_pcd'

    save_dressed_root = '/media/pengpeng/5121bd45-76e6-4694-92ac-cc439c45fde6/BUG_partial_scan/centred_dressed_pcd'
    save_body_root = '/media/pengpeng/5121bd45-76e6-4694-92ac-cc439c45fde6/BUG_partial_scan/centered_body_pcd'

    file_list=os.listdir(source_dressed_root)
    for id in file_list:
        #print('processing %s' % id)
        save_dressed_pcd,center=center_PCD(source_dressed_root+'/'+id)
        write_point_cloud(save_dressed_root+'/'+id,save_dressed_pcd)

        source_body_pcd = read_point_cloud(source_body_root + '/' + id)
        body_points=np.array(source_body_pcd.points)
        center = np.tile(center, body_points.shape[0]).reshape(-1, 3)
        body_points = body_points - center
        source_body_pcd.points = Vector3dVector(body_points)
        write_point_cloud(save_body_root + '/' + id,source_body_pcd)

def testing_data_extraction():# extract the testing dataset for evaluation
    root='/media/pengpeng/5121bd45-76e6-4694-92ac-cc439c45fde6/BUG_partial_scan/'

    dressed_root = root+'partial_male_longsleeve_dressed_input'
    body_root = root + 'male_densebody_pcd'

    save_testing=root+'experiments'

    #model_list = os.listdir(dressed_root)
    model_list=os.listdir('/media/pengpeng/5121bd45-76e6-4694-92ac-cc439c45fde6/BUG_partial_scan/test/dressed')

    model_list=sorted(model_list)

    #random.shuffle(model_list)


    #testing_list = model_list[int(len(model_list) * 0.99):]
    testing_list=model_list


    for file in testing_list:
        file=file+'.pcd'
        # save dressed
        source_dressed = dressed_root + '/' + file
        target_dressed = save_testing + '/partial_dressed_input/' + file
        os.system('cp' + ' ' + source_dressed + ' ' + target_dressed)

        # save body
        source_body = body_root + '/' + file
        target_body = save_testing + '/body_gt/' + file
        os.system('cp' + ' ' + source_body + ' ' + target_body)


def sample_densebody():
    ply_list=os.listdir('/media/pengpeng/5121bd45-76e6-4694-92ac-cc439c45fde6/APose_BUG/male/body_ply')
    save_root='/media/pengpeng/5121bd45-76e6-4694-92ac-cc439c45fde6/APose_BUG/male/densebody_pcd'

    num_sample=20000
    for id in ply_list:
        ply = read_triangle_mesh(ply_list+'/'+id)
        pcd = sample_points_poisson_disk(ply, num_sample)
        draw_geometries([pcd])



if __name__=='__main__':
    #sample_densebody()
    #center3d()
    #add_random_rotation()
    #devide_train_and_valid()
    #calculate_PointConfidences()
    # ply_normalization()
    # obj_normalization()
    #multi_obj_to_pcd()
    #testing_data_extraction()

    merge_mutlti_pcd()
    # rename_pcd()
    #multi_obj_to_pcd()
    # Move_Rename_files()
    #obj_shuffle()
    #obj_Half()
    #display_pcd('/media/pengpeng/0bc58d8e-e5a0-4fd7-96a2-fae9bc2eaf0b/SMPL_dataset/male/blender/dressedpcd/male_10.pcd')
    #multi_obj_to_pcd()
    #obj_to_pcd('/home/pengpeng/projects/NakedNet/render/BUFF_comparison/INRIA.obj','/home/pengpeng/projects/NakedNet/render/BUFF_comparison/INRIA.pcd')
    #inputname='woman06_opencloth_000023.pcd'
    #savename='woman06_opencloth_000023.ply'
    #pcd_to_ply('/home/pengpeng/projects/NakedNet/render/BUVG/hpp.pcd','/home/pengpeng/projects/NakedNet/render/BUVG/hpp.ply')
    #devide_train_and_valid()
    # for complete data, required by lmdb_writer.py

    #add_transformation()
    #multi_ply_to_pcd()
    #multi_ply_to_obj()
    #move_hpp()




