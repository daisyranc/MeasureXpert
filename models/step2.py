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
    def __init__(self, inputs,gt,gt_values,alpha):

        self.num_output_points=10363

        self.feature= self.create_encoder(inputs)

        self.outputs_t=self.create_decoder(self.feature,inputs)

        self.bust_value=self.regression(self.outputs_t[:, 17:273, :])
        self.under_bust_value=self.regression(self.outputs_t[:, 273:529, :])
        self.hip_value=self.regression(self.outputs_t[:, 529:785, :])
        self.waist1_value=self.regression(self.outputs_t[:, 785:1041, :])
        self.waist2_value=self.regression(self.outputs_t[:, 1041:1297, :])
        self.waist3_value=self.regression(self.outputs_t[:, 1297:1553, :])
        self.waist4_value=self.regression(self.outputs_t[:, 1553:1809, :])
        self.waist5_value=self.regression(self.outputs_t[:, 1809:2065, :])
        self.waist6_value=self.regression(self.outputs_t[:, 2065:2321, :])
        self.right_middle_value=self.regression(self.outputs_t[:, 2321:2449, :])
        self.left_middle_value=self.regression(self.outputs_t[:, 2449:2577, :])
        self.right_knee_value=self.regression(self.outputs_t[:, 2577:2705, :])
        self.left_knee_value=self.regression(self.outputs_t[:, 2705:2833, :])
        self.right_calf_value=self.regression(self.outputs_t[:, 2833:2961, :])
        self.left_calf_value=self.regression(self.outputs_t[:, 2961:3089, :])
        self.right_upper_value=self.regression(self.outputs_t[:, 3089:3153, :])
        self.left_upper_value=self.regression(self.outputs_t[:, 3153:3217, :])
        self.right_elbow_value=self.regression(self.outputs_t[:, 3217:3281, :])
        self.left_elbow_value=self.regression(self.outputs_t[:, 3281:3345, :])
        self.right_wrist_value=self.regression(self.outputs_t[:, 3345:3409, :])
        self.left_wrist_value=self.regression(self.outputs_t[:, 3409:3473, :])
        self.value_loss=self.create_value_loss([self.bust_value,self.under_bust_value,self.hip_value,self.waist1_value,self.waist2_value,self.waist3_value,
                                              self.waist4_value,self.waist5_value,self.waist6_value,self.right_middle_value,self.left_middle_value,
                                              self.right_knee_value,self.left_knee_value,self.right_calf_value,self.left_calf_value,
                                              self.right_upper_value,self.left_upper_value,self.right_elbow_value,self.left_elbow_value,
                                              self.right_wrist_value,self.left_wrist_value],gt_values)

        self.loss, self.update,self.error= self.create_loss(self.outputs_t,gt,self.value_loss,
                                                            alpha)#self.value_loss,

        self.output_landmarks=self.outputs_t[:, :3473, :]
        self.output_t=self.outputs_t[:, 3473:, :]
        self.input_landmarks=inputs[:, :3473, :]
        self.input_t=inputs[:, 3473:, :]
        self.gt_landmarks=gt[:, :3473, :]
        self.gt_t=gt[:, 3473:, :]
        self.visualize_ops_landmark = [self.input_landmarks[0],self.output_landmarks[0],self.gt_landmarks[0]]
        self.visualize_titles_landmark = ['input',"output","gt"]
        self.visualize_ops_t = [self.input_t[0],self.output_t[0],self.gt_t[0]]
        self.visualize_titles_t = ['input',"output","gt"]

    def create_encoder(self, inputs):
        with tf.variable_scope('encoder_0', reuse=tf.AUTO_REUSE):
            features = mlp_conv(inputs, [128, 256])
            features_global = tf.reduce_max(features, axis=1, keep_dims=True, name='maxpool_0')
            features = tf.concat([features, tf.tile(features_global, [1, tf.shape(inputs)[1], 1])], axis=2)
        with tf.variable_scope('encoder_1', reuse=tf.AUTO_REUSE):
            features = mlp_conv(features, [512, 1024])
            features = tf.reduce_max(features, axis=1, name='maxpool_1')
        #    print(features.shape)
        return features

    def create_decoder(self, features,inputs):
        with tf.variable_scope('decoder', reuse=tf.AUTO_REUSE):
            outputs = mlp(features, [1024, 1024, self.num_output_points * 3])
        #    print(outputs.shape)
            outputs = tf.reshape(outputs, [-1, self.num_output_points, 3])
         #   print(outputs.shape)
        outputs=outputs+inputs
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
            x_value=gt_values[:,:21,0][:,i]
            print("x_value.shape: ", x_value.shape)
            print("value_list[i].shape: ", value_list[i].shape)
            l1_loss=tf.reduce_mean(tf.abs(value_list[i]-x_value))
            print("l1.shape: ", l1_loss.shape)
            value_loss_list.append(l1_loss)
        return value_loss_list

    def create_loss(self,outputs_t,gt,value_loss_list,alpha):#value_loss_list,
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
        loss_norm=sum_y_distances(outputs_t[:,17:273,:])+sum_y_distances(outputs_t[:,273:529,:])+sum_y_distances(outputs_t[:,529:785,:])+\
                  sum_y_distances(outputs_t[:,785:1041,:])+sum_y_distances(outputs_t[:,1041:1297,:])+sum_y_distances(outputs_t[:, 1297:1553, :])+\
                  sum_y_distances(outputs_t[:,1553:1809,:])+sum_y_distances(outputs_t[:,1809:2065,:]) +sum_y_distances(outputs_t[:, 2065:2321, :])+\
                  sum_y_distances(outputs_t[:,2321:2449,:])+sum_y_distances(outputs_t[:,2449:2577,:])+sum_y_distances(outputs_t[:, 2577:2705, :])+\
                  sum_y_distances(outputs_t[:,2705:2833,:])+sum_y_distances(outputs_t[:,2833:2961,:])+sum_y_distances(outputs_t[:, 2961:3089, :])+\
                  sum_x_distances(outputs_t[:,3089:3153,:])+sum_x_distances(outputs_t[:,3153:3217,:]) +sum_x_distances(outputs_t[:, 3217:3281, :])+\
                  sum_x_distances(outputs_t[:,3281:3345,:])+sum_x_distances(outputs_t[:,3345:3409,:])+sum_x_distances(outputs_t[:, 3409:3473, :])
        loss_t = l2_t+loss_keys+loss_landmarks

    #    loss_landmarks_values=loss_norm
        value_loss=0

        for item in value_loss_list:
           value_loss+=item

        #===whole loss===========
        loss=loss_t+value_loss#+alpha*loss_landmarks_values
        print(loss_t.shape)
    #    print(loss_norm.shape)
     #   print(loss_landmarks_values.shape)

        add_train_summary('train/loss', loss)
        update_loss = add_valid_summary('valid/loss', loss)
        error=[loss,value_loss,loss_norm,loss_t,l2_t,loss_keys,loss_landmarks]+value_loss_list
        return loss, update_loss,error