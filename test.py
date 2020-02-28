import tensorflow as tf

d=tf.range(8)
e=tf.range(10)
def f():
    print(tf.executing_eagerly())
    print(tf.cond(1, lambda: d, lambda: e))
    print(tf.executing_eagerly())


if __name__ == '__main__':
    f()
