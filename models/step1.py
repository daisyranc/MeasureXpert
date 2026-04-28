import tensorflow as tf
import tf_util
from tf_util import mlp, mlp_conv, chamfer, add_train_summary, add_valid_summary


def pairwise_distance(p1, p2):
    """
    Compute pairwise Euclidean distances between two sets of points.
    Args:
        p1 (tf.Tensor): Tensor of shape (N, D) representing points.
        p2 (tf.Tensor): Tensor of shape (M, D) representing points.
    Returns:
        tf.Tensor: Pairwise Euclidean distances of shape (N, M).
    """
    return tf.norm(tf.expand_dims(p1, 1) - tf.expand_dims(p2, 0), axis=-1)

def sum_y_distances(batch_points):
    """
    Compute the sum of distances along the y-axis for each batch.
    Args:
        batch_points (tf.Tensor): Tensor of shape (batch_size, num_points, 3) representing 3D points.
    Returns:
        tf.Tensor: Sum of distances along the y-axis for each batch, of shape (batch_size,).
    """
    # Get the y values of the points
    y_values = batch_points[:, :, 1]  # Extract the y-values (assuming y is the second dimension)
    # Compute pairwise distances along the y-axis
    y_distances = pairwise_distance(y_values[:, :, tf.newaxis],
                                    y_values[:, tf.newaxis, :])  # Shape: (batch_size, num_points, num_points)
    # Sum distances along the y-axis for each batch
    y_distances_sum = tf.reduce_sum(y_distances, axis=[1, 2])
    y_distances_sum= tf.reduce_mean(y_distances_sum)# Sum along dimensions 1 and 2
    return y_distances_sum

def sum_x_distances(batch_points):
    """
    Compute the sum of distances along the x-axis for each batch.
    Args:
        batch_points (tf.Tensor): Tensor of shape (batch_size, num_points, 3) representing 3D points.
    Returns:
        tf.Tensor: Sum of distances along the x-axis for each batch, of shape (batch_size,).
    """
    # Get the x values of the points
    x_values = batch_points[:, :, 0]  # Extract the x-values (assuming x is the first dimension)

    # Compute pairwise distances along the x-axis
    x_distances = pairwise_distance(x_values[:, :, tf.newaxis],
                                    x_values[:, tf.newaxis, :])  # Shape: (batch_size, num_points, num_points)

    # Sum distances along the x-axis for each batch
    x_distances_sum = tf.reduce_sum(x_distances, axis=[1, 2])
    x_distances_sum= tf.reduce_mean(x_distances_sum)# Sum along dimensions 1 and 2
    return x_distances_sum
class Model:
    # def __init__(self, input_waist, gt, alpha):
    def __init__(self, inputs_front,inputs_back,
                 head_gt_front,right_arm_gt_front,left_arm_gt_front,right_leg_gt_front,left_leg_gt_front,body_gt_front,gt_front,
                 head_gt_back,right_arm_gt_back,left_arm_gt_back,right_leg_gt_back,left_leg_gt_back,body_gt_back,gt_back,gt,gt_values,alpha):

        self.num_front_points=6890
        self.num_back_points=6890
        self.num_t_points=10363
        self.num_head_points = 1315
        self.num_right_arm_points=1301
        self.num_left_arm_points=1275
        self.num_right_leg_points=607
        self.num_left_leg_points=612
        self.num_body_points=1898
        self.feature_front,self.feature_back,self.feature_shape= self.encoder(inputs_front,inputs_back)

        self.head_front,self.head_back=self.decoder_head(self.feature_front,self.feature_back)
        self.right_arm_front,self.right_arm_back=self.decoder_right_arm(self.feature_front,self.feature_back)
        self.left_arm_front,self.left_arm_back=self.decoder_left_arm(self.feature_front,self.feature_back)
        self.right_leg_front,self.right_leg_back=self.decoder_right_leg(self.feature_front,self.feature_back)
        self.left_leg_front,self.left_leg_back=self.decoder_left_leg(self.feature_front,self.feature_back)
        self.body_front,self.body_back=self.decoder_body(self.feature_front,self.feature_back)
        self.outputs_front = self.create_smpl(self.head_front,self.right_arm_front,self.left_arm_front,self.right_leg_front,self.left_leg_front,self.body_front)
        self.outputs_back = self.create_smpl(self.head_back, self.right_arm_back, self.left_arm_back,
                                              self.right_leg_back, self.left_leg_back, self.body_back)
        self.outputs_t=self.decoder_t(self.feature_shape)
        self.loss, self.update,self.error= self.create_loss(self.feature_front,self.feature_back,
            self.head_front,self.right_arm_front,self.left_arm_front,self.right_leg_front,self.left_leg_front,self.body_front,self.outputs_front,
                                                            self.head_back, self.right_arm_back, self.left_arm_back, self.right_leg_back,
                                                            self.left_leg_back, self.body_back, self.outputs_back,
                                                            head_gt_front, right_arm_gt_front, left_arm_gt_front, right_leg_gt_front,
                                                            left_leg_gt_front, body_gt_front, gt_front,
                                                            head_gt_back,right_arm_gt_back,left_arm_gt_back,right_leg_gt_back,left_leg_gt_back,body_gt_back,gt_back,
                                                            self.outputs_t,gt,
                                                            alpha)#self.value_loss,
        self.visualize_ops_front = [inputs_front[0],self.outputs_front[0],gt_front[0]]
        self.visualize_titles_front = ['input_front',"output_front","gt_front"]
        self.visualize_ops_back = [inputs_back[0],self.outputs_back[0],gt_back[0]]
        self.visualize_titles_back = ['input_back',"output_back","gt_back"]
        self.visualize_ops_t = [inputs_front[0],inputs_back[0],self.outputs_t[0]]
        self.visualize_titles_t = ['input_front',"input_back","output_T"]


    def encoder(self,inputs_front,inputs_back):
        with tf.variable_scope('encoder_front_0', reuse=tf.AUTO_REUSE):
            features_front = mlp_conv(inputs_front, [128,256])
            features_global_front = tf.reduce_max(features_front, axis=1, keep_dims=True, name='maxpool_front_0')
            features_front = tf.concat([features_front, tf.tile(features_global_front, [1, tf.shape(inputs_front)[1], 1])], axis=2)
        with tf.variable_scope('encoder_front_1', reuse=tf.AUTO_REUSE):
            features_front = mlp_conv(features_front, [512,1024])
            features_front = tf.reduce_max(features_front, axis=1, name='maxpool_front_1')

        with tf.variable_scope('encoder_back_0', reuse=tf.AUTO_REUSE):
            features_back = mlp_conv(inputs_back, [128,256])
            features_global_back = tf.reduce_max(features_back, axis=1, keep_dims=True, name='maxpool_back_0')
            features_back = tf.concat([features_back, tf.tile(features_global_back, [1, tf.shape(inputs_back)[1], 1])], axis=2)
        with tf.variable_scope('encoder_back_1', reuse=tf.AUTO_REUSE):
            features_back = mlp_conv(features_back, [512,1024])
            features_back = tf.reduce_max(features_back, axis=1, name='maxpool_back_1')
        features_front_pose=features_front[:,:512]
        print("features_front_pose.shape: ",features_front_pose.shape)
        features_front_shape=features_front[:,512:]
        print("features_front_shape.shape: ", features_front_shape.shape)

        features_back_pose=features_back[:,:512]
        print("features_back_pose.shape: ", features_back_pose.shape)
        features_back_shape=features_back[:,512:]
        print("features_back_shape.shape: ", features_back_shape.shape)

        with tf.variable_scope('decoder_weights_00', reuse=tf.AUTO_REUSE):
            weight_front = mlp(features_front_pose, [1024, 1024, 512])
            weight_front =tf.nn.softplus(weight_front)
        print("weight_front.shape: ", weight_front.shape)
        with tf.variable_scope('decoder_weights_01', reuse=tf.AUTO_REUSE):
            weight_back = mlp(features_back_pose, [1024, 1024, 512])
            weight_back= tf.nn.softplus(weight_back)
        print("weight_back.shape: ", weight_back.shape)

        weight_front_norm=weight_front/(weight_front+weight_back)
        weight_back_norm=weight_back/(weight_front+weight_back)

        features_shape=tf.add(features_front_shape * weight_front_norm, features_back_shape * weight_back_norm)
      #  features_shape = tf.add(features_front_shape , features_back_shape )/2.0
        print("features_shape.shape: ", features_shape.shape)
        features_1=tf.concat([features_front_pose, features_shape], axis=-1)
        print("features_1.shape: ", features_1.shape)
        features_2=tf.concat([features_back_pose, features_shape], axis=-1)
        print("features_2.shape: ", features_2.shape)
        return features_1,features_2,features_shape

    def decoder_head(self, features_front,features_back):
        with tf.variable_scope('decoder_head_00', reuse=tf.AUTO_REUSE):
            outputs_front = mlp(features_front, [1024, 1024, self.num_head_points * 3])
            outputs_front = tf.reshape(outputs_front, [-1, self.num_head_points, 3])
        with tf.variable_scope('decoder_head_01', reuse=tf.AUTO_REUSE):
            outputs_back = mlp(features_back, [1024, 1024, self.num_head_points * 3])
            outputs_back = tf.reshape(outputs_back, [-1, self.num_head_points, 3])
        return outputs_front,outputs_back
    def decoder_right_arm(self, features_front,features_back):
        with tf.variable_scope('decoder_right_arm_00', reuse=tf.AUTO_REUSE):
            outputs_front = mlp(features_front, [1024, 1024, self.num_right_arm_points * 3])
            outputs_front = tf.reshape(outputs_front, [-1, self.num_right_arm_points, 3])
        with tf.variable_scope('decoder_right_arm_01', reuse=tf.AUTO_REUSE):
            outputs_back = mlp(features_back, [1024, 1024, self.num_right_arm_points * 3])
            outputs_back = tf.reshape(outputs_back, [-1, self.num_right_arm_points, 3])
        return outputs_front,outputs_back
    def decoder_left_arm(self, features_front,features_back):
        with tf.variable_scope('decoder_left_arm_00', reuse=tf.AUTO_REUSE):
            outputs_front = mlp(features_front, [1024, 1024, self.num_left_arm_points * 3])
            outputs_front = tf.reshape(outputs_front, [-1, self.num_left_arm_points, 3])
        with tf.variable_scope('decoder_left_arm_01', reuse=tf.AUTO_REUSE):
            outputs_back = mlp(features_back, [1024, 1024, self.num_left_arm_points * 3])
            outputs_back = tf.reshape(outputs_back, [-1, self.num_left_arm_points, 3])
        return outputs_front,outputs_back
    def decoder_right_leg(self, features_front,features_back):
        with tf.variable_scope('decoder_right_leg_00', reuse=tf.AUTO_REUSE):
            outputs_front = mlp(features_front, [1024, 1024, self.num_right_leg_points * 3])
            outputs_front = tf.reshape(outputs_front, [-1, self.num_right_leg_points, 3])
        with tf.variable_scope('decoder_right_leg_01', reuse=tf.AUTO_REUSE):
            outputs_back = mlp(features_back, [1024, 1024, self.num_right_leg_points * 3])
            outputs_back = tf.reshape(outputs_back, [-1, self.num_right_leg_points, 3])
        return outputs_front,outputs_back
    def decoder_left_leg(self, features_front,features_back):
        with tf.variable_scope('decoder_left_leg_00', reuse=tf.AUTO_REUSE):
            outputs_front = mlp(features_front, [1024, 1024, self.num_left_leg_points * 3])
            outputs_front = tf.reshape(outputs_front, [-1, self.num_left_leg_points, 3])
        with tf.variable_scope('decoder_left_leg_01', reuse=tf.AUTO_REUSE):
            outputs_back = mlp(features_back, [1024, 1024, self.num_left_leg_points * 3])
            outputs_back = tf.reshape(outputs_back, [-1, self.num_left_leg_points, 3])
        return outputs_front,outputs_back
    def decoder_body(self, features_front,features_back):
        with tf.variable_scope('decoder_body_00', reuse=tf.AUTO_REUSE):
            outputs_front = mlp(features_front, [1024, 1024, self.num_body_points * 3])
            # print("decoder2, outputs.shape: ",outputs.shape)
            outputs_front = tf.reshape(outputs_front, [-1, self.num_body_points, 3])
            # print("decoder2, outputs.shape: ",outputs.shape)
        with tf.variable_scope('decoder_body_01', reuse=tf.AUTO_REUSE):
            outputs_back = mlp(features_back, [1024, 1024, self.num_body_points * 3])
            # print("decoder2, outputs.shape: ",outputs.shape)
            outputs_back = tf.reshape(outputs_back, [-1, self.num_body_points, 3])
            # print("decoder2, outputs.shape: ",outputs.shape)
        return outputs_front,outputs_back

    def create_smpl(self,head,right_arm,left_arm,right_leg,left_leg,body):
        whole=tf.concat([head,right_arm],axis=1)
        whole=tf.concat([whole,left_arm],axis=1)
        whole = tf.concat([whole, right_leg], axis=1)
        whole = tf.concat([whole, left_leg], axis=1)
        whole = tf.concat([whole, body], axis=1)
        #print("whole shape: ", whole.shape)
        return whole

    def decoder_t(self,features_shape):
        with tf.variable_scope('decoder_t_00', reuse=tf.AUTO_REUSE):
            outputs = mlp(features_shape, [1024, 1024, self.num_t_points * 3])
            # print("decoder2, outputs.shape: ",outputs.shape)
            outputs = tf.reshape(outputs, [-1, self.num_t_points, 3])
            # print("decoder2, outputs.shape: ",outputs.shape)
        return outputs

    def regression(self,pred_landmarks):
        with tf.variable_scope('encoder_landmarks_0', reuse=tf.AUTO_REUSE):
            features_landmarks = mlp_conv(pred_landmarks, [128,256])
            features_global_landmarks = tf.reduce_max(features_landmarks, axis=1, keep_dims=True, name='maxpool_landmarks_0')
            features_landmarks= tf.concat([features_landmarks, tf.tile(features_global_landmarks, [1, tf.shape(pred_landmarks)[1], 1])], axis=2)
        with tf.variable_scope('encoder_landmarks_1', reuse=tf.AUTO_REUSE):
            features_landmarks = mlp_conv(features_landmarks, [512,1024])
            features_landmarks = tf.reduce_max(features_landmarks, axis=1, name='maxpool_landmarks_1')
        with tf.variable_scope('decoder_landmarks_00', reuse=tf.AUTO_REUSE):
            outputs = mlp(features_landmarks, [1024, 1024, 1])
            # print("decoder2, outputs.shape: ",outputs.shape)
            outputs = tf.reshape(outputs, [-1])
            # print("decoder2, outputs.shape: ",outputs.shape)
        return outputs

    def create_value_loss(self,value_list,gt_values):
        value_loss_list=[]
        print("gt_value.shape: ",gt_values.shape)
        for i in range(len(value_list)):
            x_value=gt_values[:,:,0][:,i]
            print("x_value.shape: ", x_value.shape)
            print("value_list[i].shape: ", value_list[i].shape)
            l1_loss=tf.reduce_mean(tf.abs(value_list[i]-x_value))
            print("l1.shape: ", l1_loss.shape)
            value_loss_list.append(l1_loss)
        return value_loss_list

    def create_loss(self,feature_front,feature_back,
            head_front,right_arm_front,left_arm_front,right_leg_front,left_leg_front,body_front,outputs_front,
            head_back,right_arm_back, left_arm_back, right_leg_back,left_leg_back, body_back, outputs_back,
            head_gt_front, right_arm_gt_front, left_arm_gt_front, right_leg_gt_front,left_leg_gt_front, body_gt_front, gt_front,
            head_gt_back,right_arm_gt_back,left_arm_gt_back,right_leg_gt_back,left_leg_gt_back,body_gt_back,gt_back,
                                                            outputs_t,gt,alpha):#value_loss_list,
        #===loss for shape feature===
        loss_shape=tf.nn.l2_loss(feature_front-feature_back)
        #===loss for front view========
        loss_head_front=tf.nn.l2_loss(head_front-head_gt_front)
        loss_right_arm_front=tf.nn.l2_loss(right_arm_front-right_arm_gt_front)
        loss_left_arm_front=tf.nn.l2_loss(left_arm_front-left_arm_gt_front)
        loss_right_leg_front=tf.nn.l2_loss(right_leg_front-right_leg_gt_front)
        loss_left_leg_front=tf.nn.l2_loss(left_leg_front-left_leg_gt_front)
        loss_body_front=tf.nn.l2_loss(body_front-body_gt_front)

        loss_head_boudary_front=tf.nn.l2_loss(head_front[:,:29,:]-body_front[:,:29,:])
        loss_right_arm_boudary_front=tf.nn.l2_loss(right_arm_front[:,:28,:]-body_front[:,29:57,:])
        loss_left_arm_boudary_front=tf.nn.l2_loss(left_arm_front[:,:26,:]-body_front[:,57:83,:])
        loss_right_leg_boudary_front=tf.nn.l2_loss(right_leg_front[:,:15,:]-body_front[:,83:98,:])
        loss_left_leg_boudary_front=tf.nn.l2_loss(left_leg_front[:,:20,:]-body_front[:,98:118,:])

        loss_whole_front=chamfer(outputs_front,gt_front)

        loss_part_front=loss_head_front+loss_right_arm_front+loss_left_arm_front+loss_right_leg_front+loss_left_leg_front+loss_body_front
        loss_boudary_front=loss_head_boudary_front+loss_right_arm_boudary_front+loss_left_arm_boudary_front+loss_right_leg_boudary_front+loss_left_leg_boudary_front

        loss_front=loss_part_front+loss_boudary_front+loss_whole_front
        #===loss for back view========
        loss_head_back=tf.nn.l2_loss(head_back-head_gt_back)
        loss_right_arm_back=tf.nn.l2_loss(right_arm_back-right_arm_gt_back)
        loss_left_arm_back=tf.nn.l2_loss(left_arm_back-left_arm_gt_back)
        loss_right_leg_back=tf.nn.l2_loss(right_leg_back-right_leg_gt_back)
        loss_left_leg_back=tf.nn.l2_loss(left_leg_back-left_leg_gt_back)
        loss_body_back=tf.nn.l2_loss(body_back-body_gt_back)

        loss_head_boudary_back=tf.nn.l2_loss(head_back[:,:29,:]-body_back[:,:29,:])
        loss_right_arm_boudary_back=tf.nn.l2_loss(right_arm_back[:,:28,:]-body_back[:,29:57,:])
        loss_left_arm_boudary_back=tf.nn.l2_loss(left_arm_back[:,:26,:]-body_back[:,57:83,:])
        loss_right_leg_boudary_back=tf.nn.l2_loss(right_leg_back[:,:15,:]-body_back[:,83:98,:])
        loss_left_leg_boudary_back=tf.nn.l2_loss(left_leg_back[:,:20,:]-body_back[:,98:118,:])

        loss_whole_back=chamfer(outputs_back,gt_back)

        loss_part_back=loss_head_back+loss_right_arm_back+loss_left_arm_back+loss_right_leg_back+loss_left_leg_back+loss_body_back
        loss_boudary_back=loss_head_boudary_back+loss_right_arm_boudary_back+loss_left_arm_boudary_back+loss_right_leg_boudary_back+loss_left_leg_boudary_back

        loss_back=loss_part_back+loss_boudary_back+loss_whole_back
        #===loss for T-posed======
        l2_t=tf.nn.l2_loss(outputs_t[:,3473:,:]-gt[:,3473:,:])

        #+++loss for landmarks++++++
        loss_keys=tf.nn.l2_loss(outputs_t[:,:17,:]-gt[:,:17,:])

        loss_landmarks=chamfer(outputs_t[:,17:273,:],gt[:,17:273,:])+chamfer(outputs_t[:,273:529,:],gt[:,273:529,:])\
        +chamfer(outputs_t[:,529:785,:],gt[:,529:785,:])+chamfer(outputs_t[:,785:1041,:], gt[:,785:1041, :]) +chamfer(outputs_t[:,1041:1297,:], gt[:,1041:1297, :]) +\
        chamfer(outputs_t[:, 1297:1553, :], gt[:, 1297:1553, :]) +chamfer(outputs_t[:,1553:1809,:], gt[:,1553:1809, :]) +chamfer(outputs_t[:,1809:2065,:], gt[:,1809:2065, :]) +\
        chamfer(outputs_t[:, 2065:2321, :], gt[:, 2065:2321, :]) +chamfer(outputs_t[:,2321:2449,:], gt[:,2321:2449, :]) +chamfer(outputs_t[:,2449:2577,:], gt[:,2449:2577, :]) +\
        chamfer(outputs_t[:, 2577:2705, :], gt[:, 2577:2705, :]) +chamfer(outputs_t[:,2705:2833,:], gt[:,2705:2833, :]) +chamfer(outputs_t[:,2833:2961,:], gt[:,2833:2961, :]) +\
        chamfer(outputs_t[:, 2961:3089, :], gt[:, 2961:3089, :]) +chamfer(outputs_t[:,3089:3153,:], gt[:,3089:3153, :]) +chamfer(outputs_t[:,3153:3217,:], gt[:,3153:3217, :]) +\
        chamfer(outputs_t[:, 3217:3281, :], gt[:, 3217:3281, :]) +chamfer(outputs_t[:,3281:3345,:], gt[:,3281:3345, :]) +chamfer(outputs_t[:,3345:3409,:], gt[:,3345:3409, :]) +\
        chamfer(outputs_t[:, 3409:3473, :], gt[:, 3409:3473, :])
    #    loss_norm=sum_y_distances(outputs_t[:,17:273,:])+sum_y_distances(outputs_t[:,273:529,:])+sum_y_distances(outputs_t[:,529:785,:])+\
     #             sum_y_distances(outputs_t[:,785:1041,:])+sum_y_distances(outputs_t[:,1041:1297,:])+sum_y_distances(outputs_t[:, 1297:1553, :])+\
      #            sum_y_distances(outputs_t[:,1553:1809,:])+sum_y_distances(outputs_t[:,1809:2065,:]) +sum_y_distances(outputs_t[:, 2065:2321, :])+\
       #           sum_y_distances(outputs_t[:,2321:2449,:])+sum_y_distances(outputs_t[:,2449:2577,:])+sum_y_distances(outputs_t[:, 2577:2705, :])+\
        #          sum_y_distances(outputs_t[:,2705:2833,:])+sum_y_distances(outputs_t[:,2833:2961,:])+sum_y_distances(outputs_t[:, 2961:3089, :])+\
         #         sum_x_distances(outputs_t[:,3089:3153,:])+sum_x_distances(outputs_t[:,3153:3217,:]) +sum_x_distances(outputs_t[:, 3217:3281, :])+\
          #        sum_x_distances(outputs_t[:,3281:3345,:])+sum_x_distances(outputs_t[:,3345:3409,:])+sum_x_distances(outputs_t[:, 3409:3473, :])
        loss_t = l2_t+loss_keys+loss_landmarks

    #    loss_landmarks_values=loss_norm

     #   for item in value_loss_list:
      #      loss_landmarks_values=loss_landmarks_values+item

        #===whole loss===========
        loss=loss_shape+loss_front+loss_back+loss_t#+alpha*loss_landmarks_values
        print(loss_t.shape)
    #    print(loss_norm.shape)
     #   print(loss_landmarks_values.shape)

        add_train_summary('train/loss', loss)
        update_loss = add_valid_summary('valid/loss', loss)
        error=[loss_shape,loss_front,loss_back,loss_t,l2_t,loss_keys,loss_landmarks]
        return loss, update_loss,error
