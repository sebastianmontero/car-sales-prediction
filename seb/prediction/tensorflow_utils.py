import tensorflow as tf


class TensorflowUtils:
        
    @staticmethod
    def summary_value(name, value):
        return tf.Summary(value=[tf.Summary.Value(tag=name, simple_value=value)])
            
        
        