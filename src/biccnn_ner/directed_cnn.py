import tensorflow as tf


class DirectedCNN:
  
  '''
  text 版本的 CNN
  '''
  
  def __init__(self, input_tensor, filter_width, filter_num):
    '''
    input_tensor B x 1 x S x E
    filter_width 序列卷积长度
    filter_num 卷积核的数量
    '''
    self._input_tensor = input_tensor
    self._filter_width = filter_width
    self._filter_num = filter_num

    self._sequence_length = self.__get_sequence_length() # + 2 * (self._filter_width - 1)
    self._hidden_size = self.__get_hidden_size()

    # padding
    paddings = tf.constant([[0, 0], [0, 0], [self._filter_width - 1, self._filter_width - 1], [0, 0]])
    self._padding_input_tensor = tf.pad(tensor=self._input_tensor, paddings=paddings)
    
  def __get_hidden_size(self):
    '''
    获取输入的embedding size
    '''
    
    input_tensor = self._input_tensor
    return input_tensor.get_shape().as_list()[3]
    
  def __get_sequence_length(self):
    '''
    获取输入的序列长度
    '''
    
    input_tensor = self._input_tensor
    return input_tensor.get_shape().as_list()[2]
  
  def __init_conv_kernel(self, i):
    '''
    初始化卷积
    '''
    
    with tf.variable_scope(name_or_scope='conv_kernel', reuse=tf.AUTO_REUSE):
      hidden_size = self.__get_hidden_size()
      return tf.get_variable(name='conv_kernel_%d'%(i),
                             shape=[self._filter_width, hidden_size],
                             initializer=tf.truncated_normal_initializer(stddev=0.02))
    
  def __run(self, i_pos):
    '''
    单位置执行卷积
    '''
    
    if i_pos < 0 or i_pos >= self._sequence_length:
      raise OverflowError('i_pos index over flow')
    
    #get i_pos
    padding_input_tensor_slice = tf.slice(input_= self._padding_input_tensor,
                                          begin=[0,0, i_pos, 0],
                                          size=[-1, 1, self._filter_width, self._hidden_size])
    x_i = list()
    for j in range(self._filter_num):
      conv_kernel = self.__init_conv_kernel(j)
      # B x 1 x Fw x H -> B x 1
      x_ij_array = tf.multiply(padding_input_tensor_slice, conv_kernel)
      x_ij = tf.reduce_sum(tf.reduce_sum(input_tensor=x_ij_array, axis=-1), axis=-1)
      x_ij = tf.expand_dims(input=x_ij, axis=-1)
      x_i.append(x_ij)
      
    # [B x 1 x 1] -> B x 1 x Fn
    ret = tf.concat(values=x_i, axis=-1)
    return ret
  
  def single_pass_run(self, direction):
    '''
    执行单向 cnn
    '''
    with tf.name_scope(direction):
      x = list()
      for i_pos in range(self._sequence_length):
        if direction == 'from_left_2_right':
          ii_pos = i_pos
        else:
          ii_pos = self._sequence_length - i_pos - 1
        x_i = self.__run(ii_pos)
        x.append(x_i)
  
      # [B x 1 x Fn] -> [B x 1 x S x Fn]
      ret = tf.stack(values=x, axis=-2)
    return ret

def bi_stack_run(input_tensor, filters):
  '''
  双向执行 directed cnn
  '''
  with tf.variable_scope('bi_stack_run'):
    n = len(filters)
    input_tensor_l2r = input_tensor
    for i in range(n):
      with tf.variable_scope("layer_l2r_%d"%(i)):
        ccnn = DirectedCNN(input_tensor=input_tensor_l2r, filter_width=filters[i][0], filter_num=filters[i][1])
        input_tensor_l2r = ccnn.single_pass_run('from_left_2_right')

    input_tensor_r2l = input_tensor
    for i in range(n):
      with tf.variable_scope("layer_r2l_%d" % (i)):
        ccnn = DirectedCNN(input_tensor=input_tensor_l2r, filter_width=filters[i][0], filter_num=filters[i][1])
        input_tensor_r2l = ccnn.single_pass_run('from_right_2_left')
  
  return [input_tensor_l2r, input_tensor_r2l]

def ui_stack_run(input_tensor, filters):
  '''
  单向执行 directed cnn
  '''
  with tf.variable_scope('ui_stack_run'):
    n = len(filters)
    input_tensor_l2r = input_tensor
    for i in range(n):
      with tf.variable_scope("layer_l2r_%d" % (i)):
        ccnn = DirectedCNN(input_tensor=input_tensor_l2r, filter_width=filters[i][0], filter_num=filters[i][1])
        input_tensor_l2r = ccnn.single_pass_run('from_left_2_right')
  
  return input_tensor_l2r

import numpy as np
a = np.arange(start=0, stop=2 * 3 * 4).reshape([2,1,3,4])
tf_a = tf.constant(value=a, dtype=tf.float32)

y = bi_stack_run(input_tensor=tf_a, filters=[[3,4],[2,4]])

sess = tf.Session(graph=tf.get_default_graph())
sess.run(tf.global_variables_initializer())

y_val = sess.run(y)
print(y_val[0])
print(y_val[1])




  


