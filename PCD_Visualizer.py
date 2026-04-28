from open3d import *


def custom_draw_geometry_with_rotation(pcdlist,rospeed=0.05):
    def rotate_view(vis):
        ctr=vis.get_view_control()
        ctr.rotate(rospeed,0.0)
        return False
    draw_geometries_with_animation_callback(pcdlist,rotate_view)


def custom_draw_geometry_with_key_callback(pcdlist):
    # def change_background_to_balck(vis):
    #     opt=vis.get_render_option()
    #     opt.background_color=np.asarray([0,0,0])
    #     return False
    #
    # def showall(vis):
    #     custom_draw_geometry_with_rotation(pcd1,pcd2)
    #     return False

    def showfirst(vis):
        draw_geometries([pcdlist[0]])
        return False

    def showsecond(vis):
        draw_geometries([pcdlist[1]])
        vis.update_geometry()
        return True
    key_to_callback={}
    #key_to_callback[ord('B')]=change_background_to_balck
    key_to_callback[ord('A')]=showfirst
    key_to_callback[ord('S')] = showsecond
    draw_geometries_with_key_callbacks(pcdlist,key_to_callback)

if __name__=='__main__':
    #mesh=read_triangle_mesh('/home/pengpeng/projects/Zalando_demo/RealScan/centered/in1_clean.ply')
    #mesh.compute_vertex_normals()
    #mesh=read_point_cloud("/home/pengpeng/projects/Zalando_demo/RealScan/dressedpcd/in1_clean.pcd")
    pcd=read_point_cloud('/home/pengpeng/projects/Zalando_demo/RealScan/prediction/124.pcd')

    #mesh.paint_uniform_color([0,1,0])
    pcd.paint_uniform_color([1,0,0])


    custom_draw_geometry_with_rotation([pcd],0.1)

    #custom_draw_geometry_with_key_callback([mesh,pcd])
