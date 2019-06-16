import os
import tensorflow
from BestCheckpointSaver import BestCheckpointSaver
from BestCheckpointSaver import get_best_checkpoint
import numpy
import ElasticityTensor_manips
import copy
import time

class NeuralNetworkTrainer:

    def __init__(
        self,
        name,
        logs_directory,
        dataset,
        optimizer_name      = "AdamOptimizer",
        verbose             = True,
        best_checkpoint_dir = "./tmp_models/",
        loss_function_type  = "Euclidian",
        datatype = tensorflow.float64
        ) -> None:
        """
        Neural Network Trainer
        Parameters
        ----------
        name : str
            Name of the Neural Network Trainer.
        logs_directory : str
            Directory in which logs will be stored.
        dataset : dictionary
            Dictionary containing the data set.
        optimizer_name : str
            Name of the optimizer
        """
        assert isinstance(name,str)
        self.name = name

        assert isinstance(logs_directory,str)
        self.logs_directory = logs_directory

        if not os.path.exists(self.logs_directory):
            os.makedirs(
                self.logs_directory
            )

        # Dataset mandatory keys
        mandatory_keys = [
            "size_input",
            "size_output",
            "Input_training",
            "Input_validation",
            "Expected_training",
            "Expected_validation"
        ]
        if not set(mandatory_keys).issubset(set(dataset)):
            raise Exception("Missing keys. Mandatory keys are " + str(mandatory_keys))
        self.dataset = dataset

        self.save_tensorboard = True

        self.optimizer_name = optimizer_name

        self.verbose = verbose

        # Directory wherein checkpoints are stored
        self.best_checkpoint_dir = best_checkpoint_dir
        
        self.loss_function_type = loss_function_type

        self.datatype = datatype

        # Get "column-wise" mins and maxs to normalize data (both input and output)
        self.minimums_input_train = numpy.amin(self.dataset["Input_training"],axis=0)
        self.maximums_input_train = numpy.amax(self.dataset["Input_training"],axis=0)

        self.minimums_output_train = numpy.amin(self.dataset["Expected_training"],axis=0)
        self.maximums_output_train = numpy.amax(self.dataset["Expected_training"],axis=0)

        # Since some input parameters can be single-valued (no variation), must cope with that
        indices = numpy.where(self.minimums_input_train == self.maximums_input_train)
        self.minimums_input_train[indices] = 0


    def train(self,model,epochs,mins_norm,maxs_norm) -> None:
        """
        Train the given type of neural network.
        Parameters
        ----------
        model : object
            Neural network object
        epochs : int
            Number of epochs
        mins_norm : numpy.ndarray (float)
            Min values for normalization
        maxs_norm = numpy.ndarray (float)
            Max values for normalization
        """
        with tensorflow.Session(graph=tensorflow.Graph()) as sess:
            # Model input
            model_input = tensorflow.placeholder(
                self.datatype, [None, self.dataset["size_input"]],
                name = "input"
            )
            # Initialize model
            model.CreateModel(
                model_input,
                self.dataset["size_input"],
                self.dataset["size_output"],
                mins_norm,
                maxs_norm,
                self.minimums_output_train,
                self.maximums_output_train
            )
            # Save to Tensor Board
            if self.save_tensorboard:
                writer = tensorflow.summary.FileWriter(self.logs_directory, sess.graph)
                writer.flush()
            else:
                writer = None
            # Train
            acc = self.TrainModel(
                sess,
                writer,
                model.getModel(),
                model_input,
                epochs = epochs
            )

            if self.save_tensorboard:
                writer.close()

        sess.close()

        return

    def TrainModel(
        self,
        sess,
        writer,
        model,
        model_input,
        epochs):

        # Global step
        global_step = tensorflow.Variable(0, name='global_step',trainable=False)

        # Initialize the optimizer
        if self.optimizer_name == "AdamOptimizer":
            optimizer = tensorflow.train.AdamOptimizer()
        else:
            raise Exception("Unknown optimizer " + self.optimizer_name)

        # Create placeholder for the dataset
        expected = tensorflow.placeholder(
            self.datatype,
            [None,self.dataset["size_output"]],
            name = "expected_output"
        )

        # Create the loss function
        loss_function = 0
        if self.loss_function_type == "Naive":
            with tensorflow.name_scope("loss_function"):
                loss_function = tensorflow.reduce_mean(
                    tensorflow.square(
                        model - expected
                    ),
                    name = "cost_naive"
                )
        elif self.loss_function_type == "EuclidianTensor":
            with tensorflow.name_scope("loss_function"):
                # From Nx21 to Nx3x3x3x3
                C_3x3x3x3_storage = ElasticityTensor_manips.Get_C_3x3x3x3_storage()

                results_N_3x3x3x3 = tensorflow.map_fn(
                        lambda x : tensorflow.gather_nd(
                            x,C_3x3x3x3_storage,name = None),
                        model,
                        dtype=self.datatype,
                    name = "output_3x3x3x3"
                    )
                    
                expected_N_3x3x3x3 = tensorflow.map_fn(
                        lambda x : tensorflow.gather_nd(
                            x,C_3x3x3x3_storage,name = None),
                        expected,
                        dtype=self.datatype,
                    name = "expected_3x3x3x3"
                    )
                # Loss function : 1/N Sum_1^N || Tensor4thOutput - Tensor4thExpected ||
                loss_function = tensorflow.reduce_mean(
                        tensorflow.map_fn(
                            lambda x : tensorflow.tensordot(
                                x,x,[[0,1,2,3],[0,1,2,3]]),
                            results_N_3x3x3x3 - expected_N_3x3x3x3,
                            dtype=self.datatype
                        ),
                    name = "cost_EuclidianTensor"
                )
        else:
            raise Exception("Unknown loss function type " + self.loss_function_type)

        # Create the training operator
        training_operator = optimizer.minimize(
            loss_function,
            global_step=global_step
        )

        # Summary of the loss:
        if self.save_tensorboard:
            loss_value_tf = tensorflow.Variable(
                initial_value=-1.0,
                dtype = self.datatype,
                trainable=False
            )
            loss_summary = tensorflow.summary.scalar(
                name='loss', tensor=loss_value_tf)

        # Writers for both losses
        if self.save_tensorboard:
            writer_val = tensorflow.summary.FileWriter(
                self.logs_directory + '/plot_val'
            )
            writer_train = tensorflow.summary.FileWriter(
                self.logs_directory + '/plot_train'
            )
            writer_error_L2 = tensorflow.summary.FileWriter(
                self.logs_directory + "/plot_error_L2"
            )

        # Initialize tensorflow session
        init_g = tensorflow.global_variables_initializer()
        init_l = tensorflow.local_variables_initializer()
        sess.run(init_l)
        sess.run(init_g)

        # Summary of the weights of the first layer of the first function network:
        if self.save_tensorboard:
            try:
                graph = tensorflow.get_default_graph()
                FunctionNetwork_summary = \
                    tensorflow.summary.histogram(
                        "ContextNN/FuncContPair_0/FunctionNetwork_0/FuncNet0_weights_0",
                        graph.get_tensor_by_name(
                            "ContextNN/FuncContPair_0/FunctionNetwork_0/FuncNet0_weights_0:0"
                        )
                    )
            except Exception as e:
                print(e)
                pass

        # Initialize the model saver
        if os.path.exists(self.best_checkpoint_dir):
            for root, dirs, files in os.walk(self.best_checkpoint_dir, topdown=False):
                for name in files:
                    os.remove(os.path.join(root, name))
                for name in dirs:
                    os.rmdir(os.path.join(root, name))
        best_ckpt_saver = BestCheckpointSaver(
            save_dir = self.best_checkpoint_dir,
            num_to_keep=1,
            maximize=False,
            saver=tensorflow.train.Saver()
        )

        # Training loop:
        loss = 0.
        loss_validation = 0.
        starting_epoch = time.time()
        for epoch in range(epochs):
            print(
                "> Epoch ",
                epoch,
                " (mean time per epoch : ",
                (time.time()-starting_epoch)/(epoch+1),
                " sec.)."
            )
            # Train
            sess.run(
                training_operator,
                feed_dict={
                    model_input : self.dataset["Input_training"],
                    expected    : self.dataset["Expected_training"]
                }
            )
            # Save and summary
            if epoch%50 == 0 or epoch == epochs-1:
                loss = sess.run(
                    loss_function,
                    feed_dict={
                        model_input : self.dataset["Input_training"],
                        expected    : self.dataset["Expected_training"]
                    }
                )
                loss_validation = sess.run(
                    loss_function,
                    feed_dict={
                        model_input : self.dataset["Input_validation"],
                        expected    : self.dataset["Expected_validation"]
                    }
                )
                # Printing
                if self.save_tensorboard:
                    print('> Saving tensorboard')
                # Assign loss on training data:
                if self.save_tensorboard:
                    sess.run(loss_value_tf.assign(loss))
                    summary = sess.run(loss_summary)
                    writer_train.add_summary(summary,epoch)
                    writer_train.flush()
                # Assign loss on validation data:
                if self.save_tensorboard:
                    sess.run(loss_value_tf.assign(loss_validation))
                    summary = sess.run(loss_summary)
                    writer_val.add_summary(summary,epoch)
                    writer_val.flush()
                # Save best check-points:
                best_ckpt_saver.handle(loss, sess, global_step)
                # Summary of weights
                if self.save_tensorboard:
                    try:
                        summary = sess.run(FunctionNetwork_summary)
                        writer.add_summary(summary,epoch)
                        writer.flush()
                    except:
                        pass
                # Evaluate L2 error:
                if self.save_tensorboard :
                    predicted = sess.run(
                        model,
                        feed_dict={
                            model_input : self.dataset["Input_validation"]
                        }
                    )
                    error_L2 = \
                        numpy.linalg.norm(
                            self.dataset["Expected_validation"]
                            -predicted,2
                        )\
                        /numpy.linalg.norm(
                            self.dataset["Expected_validation"],2
                        )
                    with open("PredictionVSexpectation.dat","w+") as fout:
                        formatterError={'float_kind':lambda x: "%.2f" % x}
                        formatterValue={'float_kind':lambda x: "%.4e" % x}
                        max_iter_tmptmp_i = min(50,predicted.shape[0])
                        fout.write("> Epoch : %d\n"%epoch)
                        for tmptmp_i in range(max_iter_tmptmp_i):
                            fout.write(50*"#" + "\n")
                            tmptmp_arr_1 = predicted[tmptmp_i,:]
                            tmptmp_arr_2 = self.dataset["Expected_validation"][tmptmp_i,:]
                            fout.write(
                                "E : " 
                                + numpy.array2string(
                                    tmptmp_arr_1,
                                    formatter=formatterValue
                                ).replace("\n","")
                                + "\n"
                            )
                            fout.write(
                                "P : " 
                                + numpy.array2string(
                                    tmptmp_arr_2,
                                    formatter=formatterValue
                                ).replace("\n","")
                                + "\n"
                            )
                            fout.write(
                                "%% : "
                                + numpy.array2string(
                                    (tmptmp_arr_1/tmptmp_arr_2-1.0)*100.0,
                                    formatter=formatterError
                                ).replace("\n","")
                                + "\n"
                            )
                    print("> Shape of predicted : ",predicted.shape)
                    sess.run(loss_value_tf.assign(error_L2))
                    summary = sess.run(loss_summary)
                    writer_error_L2.add_summary(summary,epoch)
                    writer_error_L2.flush()
                    
                # Print
                if self.verbose:
                    print("> Epoch ",epoch+1," : loss training ",loss,", loss validation ",loss_validation)

        # Get best checkpoint
        saver = tensorflow.train.Saver()
        best_checkpoint_path = \
            get_best_checkpoint(
                self.best_checkpoint_dir,
                select_maximum_value=False
            )
        saver.restore(
            sess,
            best_checkpoint_path
        )

        # Evaluate the best checkpoint on the validation dataset
        best_loss = sess.run(
            loss_function,
            feed_dict={
                model_input : self.dataset["Input_validation"],
                expected    : self.dataset["Expected_validation"]
            }
        )
        if self.verbose:
            print("> Best loss is ",best_loss,"( ",best_checkpoint_path," )")

        # Compute the distance between predicted and expected data (on validation dataset)
        predicted = sess.run(
            model,
            feed_dict={
                model_input : self.dataset["Input_validation"]
            }
        )

        if self.verbose:
            print("> Input validation :",self.dataset["Input_validation"].shape)
            print(self.dataset["Input_validation"])
            print("> Expected validation dataset:")
            print(self.dataset["Expected_validation"])
            print("> Predicted : ")
            print(predicted)

        error_u = \
            numpy.linalg.norm(
                self.dataset["Expected_validation"]
                -predicted,2
            )\
            /numpy.linalg.norm(
                self.dataset["Expected_validation"],2
            )

        acc = 1. - error_u
        
        if self.verbose:
            print("> Error in L2 sense : ",error_u)
            print("> Accuracy in L2 sense (1.-error) : ",acc)

        return acc
        
