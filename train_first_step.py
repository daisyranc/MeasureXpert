import argparse
import datetime
import importlib
import models
import os
import tensorflow as tf
import time
from data_util import lmdb_dataflow, get_queued_data, lmdb_dataflow
from termcolor import colored
from tf_util import add_train_summary
from visu_util import plot_pcd_three_views,plot_pcd_three_views_rb
import pandas as pd

#os.environ["CUDA_VISIBLE_DEVICES"]="1"
#partial_front,head_front,right_arm_front,left_arm_front,right_leg_front,left_leg_front,body_front,posed_front,
#partial_back,head_back,right_arm_back,left_arm_back,right_leg_back,left_leg_back,body_back,posed_back,t
class TrainProvider:
    def __init__(self, args, is_training):
        df_train, self.num_train = lmdb_dataflow(args.lmdb_train, args.batch_size,
                                                       args.num_input_points,args.num_gt_landmark_points,
                                                       is_training=True)
        batch_train = get_queued_data(df_train.get_data(), [tf.string, tf.float32, tf.float32, tf.float32,tf.float32,tf.float32,tf.float32, tf.float32, tf.float32,
                                                                            tf.float32, tf.float32, tf.float32, tf.float32,
                                                                            tf.float32, tf.float32, tf.float32, tf.float32, tf.float32, tf.float32, tf.float32, tf.float32
                                                                            ],
                                          [[args.batch_size],
                                           [args.batch_size, args.num_input_points, 3],
                                           [args.batch_size, args.num_head_points, 3],
                                           [args.batch_size, args.num_right_arm_points, 3],
                                           [args.batch_size, args.num_left_arm_points, 3],
                                           [args.batch_size, args.num_right_leg_points, 3],
                                           [args.batch_size, args.num_left_leg_points, 3],
                                           [args.batch_size, args.num_body_points, 3],
                                           [args.batch_size, args.num_gt_points, 3],
                                           [args.batch_size, args.num_input_points, 3],
                                           [args.batch_size, args.num_head_points, 3],
                                           [args.batch_size, args.num_right_arm_points, 3],
                                           [args.batch_size, args.num_left_arm_points, 3],
                                           [args.batch_size, args.num_right_leg_points, 3],
                                           [args.batch_size, args.num_left_leg_points, 3],
                                           [args.batch_size, args.num_body_points, 3],
                                           [args.batch_size, args.num_gt_points, 3],
                                           [args.batch_size, args.num_gt_landmark_points, 3],
                                           [args.batch_size, args.num_gt_points, 3],
                                           [args.batch_size, args.num_gt_points, 3],
                                           [args.batch_size, args.num_value_points, 3]
                                           ])
        df_valid, self.num_valid = lmdb_dataflow(args.lmdb_valid, args.batch_size,
                                                            args.num_input_points, args.num_gt_points,is_training=False)
        batch_valid = get_queued_data(df_valid.get_data(), [tf.string, tf.float32, tf.float32, tf.float32,tf.float32,tf.float32,tf.float32, tf.float32, tf.float32,
                                                                            tf.float32, tf.float32, tf.float32, tf.float32,
                                                                            tf.float32, tf.float32, tf.float32, tf.float32, tf.float32, tf.float32, tf.float32, tf.float32],
                                          [[args.batch_size],
                                           [args.batch_size, args.num_input_points, 3],
                                           [args.batch_size, args.num_head_points, 3],
                                           [args.batch_size, args.num_right_arm_points, 3],
                                           [args.batch_size, args.num_left_arm_points, 3],
                                           [args.batch_size, args.num_right_leg_points, 3],
                                           [args.batch_size, args.num_left_leg_points, 3],
                                           [args.batch_size, args.num_body_points, 3],
                                           [args.batch_size, args.num_gt_points, 3],
                                           [args.batch_size, args.num_input_points, 3],
                                           [args.batch_size, args.num_head_points, 3],
                                           [args.batch_size, args.num_right_arm_points, 3],
                                           [args.batch_size, args.num_left_arm_points, 3],
                                           [args.batch_size, args.num_right_leg_points, 3],
                                           [args.batch_size, args.num_left_leg_points, 3],
                                           [args.batch_size, args.num_body_points, 3],
                                           [args.batch_size, args.num_gt_points, 3],
                                           [args.batch_size, args.num_gt_landmark_points, 3],
                                           [args.batch_size, args.num_gt_points, 3],
                                           [args.batch_size, args.num_gt_points, 3],
                                           [args.batch_size, args.num_value_points, 3]])
        self.batch_data = tf.cond(is_training, lambda: batch_train, lambda: batch_valid)


def train(args):
    is_training_pl = tf.placeholder(tf.bool, shape=(), name='is_training')
    global_step = tf.Variable(0, trainable=False, name='global_step')
    alpha = tf.train.piecewise_constant(global_step, [10000, 20000, 50000],
                                        [0.01, 0.1, 1.0, 2.0], 'alpha_op')
    #inputs_front,inputs_back,
                # head_gt_front,right_arm_gt_front,left_arm_gt_front,right_leg_gt_front,left_leg_gt_front,body_gt_front,gt_front,
               #  head_gt_back,right_arm_gt_back,left_arm_gt_back,right_leg_gt_back,left_leg_gt_back,body_gt_back,gt_back,gt,alpha

    provider = TrainProvider(args, is_training_pl)
    ids, partial_front,posed_head_front,posed_right_arm_front,posed_left_arm_front,posed_right_leg_front,posed_left_leg_front,posed_body_front,posed_front,\
        partial_back, posed_head_back, posed_right_arm_back, posed_left_arm_back, posed_right_leg_back, posed_left_leg_back, posed_body_back, posed_back,t_landmarks,\
        t_front,t_back,values= provider.batch_data

    num_eval_steps = provider.num_valid // args.batch_size

    model_module = importlib.import_module('.%s' % args.model_type, 'models')
    model = model_module.Model(partial_front,partial_back,
                               posed_head_front,posed_right_arm_front,posed_left_arm_front,posed_right_leg_front,posed_left_leg_front,posed_body_front,posed_front,
                               posed_head_back, posed_right_arm_back, posed_left_arm_back, posed_right_leg_back,posed_left_leg_back, posed_body_back, posed_back,
                               t_landmarks,values, alpha)
    add_train_summary('alpha', alpha)

    if args.lr_decay:
        learning_rate = tf.train.exponential_decay(args.base_lr, global_step,
                                                   args.lr_decay_steps, args.lr_decay_rate,
                                                   staircase=True, name='lr')
        learning_rate = tf.maximum(learning_rate, args.lr_clip)
        add_train_summary('learning_rate', learning_rate)
    else:
        learning_rate = tf.constant(args.base_lr, name='lr')

    trainer = tf.train.AdamOptimizer(learning_rate)
    train_op = trainer.minimize(model.loss, global_step)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    sess = tf.Session(config=config)
    sess.run(tf.global_variables_initializer())

    saver = tf.train.Saver()
    if args.restore:
        saver.restore(sess, tf.train.latest_checkpoint(args.log_dir))
    else:
        if os.path.exists(args.log_dir):
            delete_key = input(colored('%s exists. Delete? [y (or enter)/N]'
                                       % args.log_dir, 'white', 'on_red'))
            if delete_key == 'y' or delete_key == "":
                os.system('rm -rf %s/*' % args.log_dir)
                os.makedirs(os.path.join(args.log_dir, 'plots'))
        else:
            os.makedirs(os.path.join(args.log_dir, 'plots'))
        with open(os.path.join(args.log_dir, 'args.txt'), 'w') as log:
            for arg in sorted(vars(args)):
                log.write(arg + ': ' + str(getattr(args, arg)) + '\n')     # log of arguments
        os.system('cp models/%s.py %s' % (args.model_type, args.log_dir))   # bkp of model def
        os.system('cp train_%s.py %s' % (args.train_details,args.log_dir))                         # bkp of train procedure                       # bkp of train procedure

    train_summary = tf.summary.merge_all('train_summary')
    valid_summary = tf.summary.merge_all('valid_summary')
    writer = tf.summary.FileWriter(args.log_dir, sess.graph)

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    total_time = 0
    train_start = time.time()
    step = sess.run(global_step)
    training_list = []
    valid_list=[]

    while not coord.should_stop():
        step += 1
        epoch = step * args.batch_size // provider.num_train + 1
        start = time.time()
        _, loss, summary,error_list = sess.run([train_op, model.loss, train_summary,model.error],
                                    feed_dict={is_training_pl: True})
        #error=[loss_shape,loss_front,loss_back,loss_t,l2_t,loss_keys,loss_landmarks]
        error_shape=error_list[0]
        error_front=error_list[1]
        error_back=error_list[2]
        error_t=error_list[3]

        total_time += time.time() - start
        writer.add_summary(summary, step)
        if step % args.steps_per_print == 0:
            print('epoch %d  step %d  loss %.8f error_shape %.8f  error_front %.8f  error_back %.8f  error_t %.8f l2_t %.8f' %
                  (epoch, step, loss, error_shape,error_front,error_back,error_t,error_list[4]))

            if step % args.steps_per_visu == 0:
                training_list.append([int(step), float(loss), float(error_shape), float(error_front), float(error_back),
                                      float(error_t), float(error_list[4]), float(error_list[5]), float(error_list[6])])

            total_time = 0
        if step % args.steps_per_eval == 0:

            print(colored('Testing...', 'grey', 'on_green'))
            total_loss = 0
            total_error_shape=0
            total_error_front = 0
            total_error_back= 0
            total_error_t= 0
            total_error_landmarks = 0
            total_error_key = 0
            total_error_l2_t= 0


            total_time = 0
            sess.run(tf.local_variables_initializer())
            for i in range(num_eval_steps):
                start = time.time()
                loss, _ ,error_list= sess.run([model.loss, model.update,model.error],
                                   feed_dict={is_training_pl: False})
                # error=[loss_shape,loss_front,loss_back,loss_t,l2_t,loss_keys,loss_landmarks]
                error_shape = error_list[0]
                error_front = error_list[1]
                error_back = error_list[2]
                error_t = error_list[3]
                error_l2_t=error_list[4]
                error_key= error_list[5]
                error_landmarks = error_list[6]
                total_loss += loss
                total_error_shape +=error_shape
                total_error_front +=error_front
                total_error_back +=error_back
                total_error_t +=error_t
                total_error_l2_t += error_l2_t
                total_error_landmarks +=error_landmarks
                total_error_key+=error_key
                total_time += time.time() - start

            summary = sess.run(valid_summary, feed_dict={is_training_pl: False})
            writer.add_summary(summary, step)
            print(colored('epoch %d  step %d  loss %.8f  error_shape %.8f  error_front %.8f  error_back %.8f  error_t %.8f ' %
                          (epoch, step, total_loss / num_eval_steps, total_error_shape/num_eval_steps, total_error_front/num_eval_steps
                           , total_error_back/num_eval_steps, total_error_t/num_eval_steps),
                          'grey', 'on_green'))
            if step % args.steps_per_visu ==0:
                valid_list.append(
                    [int(step), float(total_loss / num_eval_steps), float(total_error_shape / num_eval_steps),
                     float(total_error_front / num_eval_steps), float(total_error_back / num_eval_steps),
                     float(total_error_t / num_eval_steps), float(total_error_l2_t / num_eval_steps),
                     float(total_error_key / num_eval_steps),
                     float(total_error_landmarks / num_eval_steps)])

            total_time = 0

        if step % args.steps_per_visu == 0:
            model_id1, pcds1 = sess.run([ids[0], model.visualize_ops_front],
                                      feed_dict={is_training_pl: True})
            model_id1 = model_id1.decode('utf-8')
            plot_path1 = os.path.join(args.log_dir, 'plots',
                                     'front_epoch_%d_step_%d_%s.png' % (epoch, step, model_id1))
            plot_pcd_three_views(plot_path1, pcds1, model.visualize_titles_front)

            model_id2, pcds2 = sess.run([ids[0], model.visualize_ops_back],
                                      feed_dict={is_training_pl: True})
            model_id2 = model_id2.decode('utf-8')
            plot_path2 = os.path.join(args.log_dir, 'plots',
                                     'back_epoch_%d_step_%d_%s.png' % (epoch, step, model_id2))
            plot_pcd_three_views(plot_path2, pcds2, model.visualize_titles_back)

            model_id3, pcds3 = sess.run([ids[0], model.visualize_ops_t],
                                      feed_dict={is_training_pl: True})
            model_id3 = model_id3.decode('utf-8')
            plot_path3 = os.path.join(args.log_dir, 'plots',
                                     'T_epoch_%d_step_%d_%s.png' % (epoch, step, model_id3))
            plot_pcd_three_views_rb(plot_path3, pcds3, model.visualize_titles_t)

        if step % args.steps_per_save == 0:
            saver.save(sess, os.path.join(args.log_dir, 'model'), step)
            print(colored('Model saved at %s' % args.log_dir, 'white', 'on_blue'))
        if step >= args.max_step:
            break
    print('Total time', datetime.timedelta(seconds=time.time() - train_start))
    name1=["step","loss","error_front","error_back","error_total","error_landmarks","error_norm"]
    name2 = ["step", "loss", "error_shape","error_front", "error_back", "error_total", "error_l2_t", "error_norm","other"]
    #name1=["step","loss","error_head","error_right_arm","error_left_arm","error_right_leg","error_left_leg","error_body","error_whole","error_landmarks","error_norm"]
    test=pd.DataFrame(columns=name2,data=training_list)
    test_valid=pd.DataFrame(columns=name2,data=valid_list)
    save_csv_path=args.log_dir+"/training_details.csv"
    save_csv_valid_path = args.log_dir + "/validation_details.csv"
    test.to_csv(save_csv_path)
    test_valid.to_csv(save_csv_valid_path)
    coord.request_stop()
    coord.join(threads)
    sess.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--lmdb_train', default="/media/pm/Ran/Nonrigid_Registration_data/female/lmdb/train_front_back_boudary_allT_landmark_value.lmdb")
    parser.add_argument('--lmdb_valid', default="/media/pm/Ran/Nonrigid_Registration_data/female/lmdb/valid_front_back_boudary_allT_landmark_value.lmdb")

    parser.add_argument('--log_dir', default='log/first_female')
    parser.add_argument('--model_type', default='fc_temp')
    parser.add_argument('--train_details', default='first_step')
    parser.add_argument('--restore', default=False,action='store_true')
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
    parser.add_argument('--base_lr', type=float, default=0.0001)
    parser.add_argument('--lr_decay', default=True,action='store_true')
    parser.add_argument('--lr_decay_steps', type=int, default=50000)
    parser.add_argument('--lr_decay_rate', type=float, default=0.7)
    parser.add_argument('--lr_clip', type=float, default=1e-6)
    parser.add_argument('--max_step', type=int, default=300000)
    parser.add_argument('--steps_per_print', type=int, default=100)
    parser.add_argument('--steps_per_eval', type=int, default=1000)
    parser.add_argument('--steps_per_visu', type=int, default=3000)
    parser.add_argument('--steps_per_save', type=int, default=10000)
    args = parser.parse_args()

    train(args)