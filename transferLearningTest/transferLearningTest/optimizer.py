import tensorflow as tf

def get_optimizer(index, **opt_params):
    return {
        'AdadeltaOptimizer':tf.train.AdadeltaOptimizer,
        'AdagradDAOptimizer':tf.train.AdagradDAOptimizer,
        'AdagradOptimizer':tf.train.AdagradOptimizer,
        'AdamOptimizer':tf.train.AdamOptimizer,
        'FtrlOptimizer':tf.train.FtrlOptimizer,
        'GradientDescentOptimizer':tf.train.GradientDescentOptimizer,
        'MomentumOptimizer':tf.train.MomentumOptimizer,
        'ProximalAdagradOptimizer':tf.train.ProximalAdagradOptimizer,
        'ProximalGradientDescentOptimizer':tf.train.ProximalGradientDescentOptimizer,
        'RMSPropOptimizer':tf.train.RMSPropOptimizer
        }[index](**opt_params)

def get_training_operations(loss, computation_graph = None, vars_to_optimize_names = None, tf_optimizer = None):
        training_opt = None
        initializer_opt = None

        if computation_graph == None:
            vars_to_optimize = []
            for var in vars_to_optimize_names:
                vars_to_optimize.extend(tf.contrib.framework.get_variables(var))

            training_opt = tf_optimizer.minimize(loss = loss, var_list = vars_to_optimize, name = "training_opt")
            # BULSHIT STARTS HERE
            # Slots in optimizers repesent uninitialized variables that have to be initialized. Slots are attributes that have getters and are created 
            # for variables that are passed through c-tor (most of the time - see GradientDescent optimizer). Some optimizers (such as Adam) 
            # have variables that are not slots and that also need to be initialized, and are treated as special border case. 
            # TODO: check if other optimizer exibit such troubling behaviour.

            optimizer_slots = [ tf_optimizer.get_slot(var, name) for name in tf_optimizer.get_slot_names() for var in vars_to_optimize]

            if isinstance(tf_optimizer, tf.train.AdamOptimizer):
                optimizer_slots.extend(tf_optimizer._get_beta_accumulators())

            initializer_opt = tf.variables_initializer(optimizer_slots, name = "initializer_opt")
            # BULSHIT ENDS HERE
        else:
            training_opt = computation_graph.get_operation_by_name("training_opt")
            initializer_opt = computation_graph.get_operation_by_name("initializer_opt")

        return training_opt, initializer_opt
