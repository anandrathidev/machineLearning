
#Import the tensorflow module and call it tf
import tensorflow as tf
# Create a constant value called x, and give it the numerical value 35
x = tf.constant([35, 40, 45], name='x')
y = tf.Variable(x + 5, name='y')
model = tf.global_variables_initializer()
with tf.Session() as session:
	session.run(model)
	print(session.run(y))
  
