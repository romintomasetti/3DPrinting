import os,numpy,json

import tensorflow as tf

def read_dataset(dataset_path,domain_params,dataset_props):
    data = {}

    for subdir, dirs, files in os.walk(dataset_path):

        if dirs == []:
            continue
        if files == []:
            continue
        
        sample_name = subdir.split("/")[-1]

        print(50*"#")
        print("> Sample : ",sample_name)
        print(dirs)
        print(subdir)
        print(files)
        
        domain_mats = numpy.fromfile(
            os.path.join(
                subdir,
                "Realization_binary"
            ),
            dtype=numpy.int64
        ).reshape((domain_params["nodes"][0],domain_params["nodes"][1]))

        data[sample_name] = {}

        data[sample_name]["domain"] = domain_mats
        data[sample_name]["proper"] = dataset_props[sample_name]



        if False:
            from matplotlib import pyplot as plt
            plt.imshow(domain_mats)
            plt.show()

    return data

def read_dataset_resume(filename,CNN_params):
    
    lines = open(filename,"r").readlines()

    input_size = None
    output_size = None
    data = {}

    prefix = "sample_"

    counter = 0



    for line in lines:
        if "input" in line:
            input_size = int(line.split(",")[1].split(";")[0])
        elif "output" in line:
            output_size = int(line.split(",")[1].split(";")[0])
        elif "ordering" in line:
            pass
        elif input_size is not None and output_size is not None:
            line = line.replace(";\n","")
            line = line.split(",")
            line_content = numpy.array([float(x) for x in line])
            data[prefix + str(counter)] = line_content[input_size::]
            counter += 1
        else:
            raise Exception("Error invalid file format.")

    return data

# Create some wrappers for simplicity
def conv2d(x, W, b, strides=1):
    # Conv2D wrapper, with bias and relu activation
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)


def maxpool2d(x, k=2):
    # MaxPool2D wrapper
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],
                          padding='SAME')


# Create model
def conv_net(x, weights, biases,domain_sizes):
    # MNIST data input is a 1-D vector of 784 features (28*28 pixels)
    # Reshape to match picture format [Height x Width x Channel]
    # Tensor input become 4-D: [Batch Size, Height, Width, Channel]
    x = tf.reshape(x, shape=[-1, domain_sizes[0], domain_sizes[1], 1])

    # Convolution Layer
    conv1_1 = conv2d(x, weights['wc1'], biases['bc1'])
    # Max Pooling (down-sampling)
    conv1_2 = maxpool2d(conv1_1, k=2)

    # Convolution Layer
    conv2_1 = conv2d(conv1_2, weights['wc2'], biases['bc2'])
    # Max Pooling (down-sampling)
    conv2_2 = maxpool2d(conv2_1, k=2)

    # Fully connected layer
    # Reshape conv2 output to fit fully connected layer input
    fc1 = tf.reshape(conv2_2, [-1, weights['wd1'].get_shape().as_list()[0]])
    fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
    fc1 = tf.nn.relu(fc1)

    # Visualization
    Visu_1_1 = tf.slice(conv1_1,(0,0,0,0),(1,-1,-1,-1))
    Visu_1_1 = tf.reshape(Visu_1_1,(domain_sizes[0], domain_sizes[1], 32))
    # Reorder so the channels are in the first dimension, x and y follow.
    Visu_1_1 = tf.transpose(Visu_1_1, (2, 0, 1))
    # Bring into shape expected by image_summary
    Visu_1_1 = tf.reshape(Visu_1_1, (-1, domain_sizes[0], domain_sizes[1], 1))

    tf.summary.image("first_conv", Visu_1_1)


    # Output, class prediction
    out = tf.add(tf.matmul(fc1, weights['out']), biases['out'])
    return out

def CreateCNN(data,names,CNN_params,input,data_output_mins,data_output_maxs):
    
    num_input = data[names[0]].shape[0] * data[names[0]].shape[1]

    size_output = len(data[names[1]])

    print("> Summary of the CNN:")
    print("\t> number of inputs : ",num_input)
    
    if data[names[0]].shape[0]/4 != int(data[names[0]].shape[0]/4)\
        or data[names[0]].shape[1]/4 != int(data[names[0]].shape[1]/4):
        raise Exception("Impossible")

    weights = {
        # 5x5 conv, 1 input, 32 outputs
        'wc1': tf.Variable(tf.random_normal([5, 5, 1, 32]),name = "weights_wc1"),
        # 5x5 conv, 32 inputs, 64 outputs
        'wc2': tf.Variable(tf.random_normal([5, 5, 32, 64])),
        # fully connected, 7*7*64 inputs, 1024 outputs
        'wd1': tf.Variable(
            tf.random_normal([int(data[names[0]].shape[0]/4*data[names[0]].shape[1]/4*64), 1024])),
        # 1024 inputs, 10 outputs (class prediction)
        'out': tf.Variable(tf.random_normal([1024, size_output]))
    }

    biases = {
        'bc1': tf.Variable(tf.random_normal([32])),
        'bc2': tf.Variable(tf.random_normal([64])),
        'bd1': tf.Variable(tf.random_normal([1024])),
        'out': tf.Variable(tf.random_normal([size_output]))
    }

    output = conv_net(
        input,
        weights,
        biases,
        domain_sizes = [data[names[0]].shape[0],data[names[0]].shape[1]]
    )

    return (output+1.0)*(data_output_maxs-data_output_mins)/2.0+data_output_mins

if __name__ == "__main__":

    project_root = os.getcwd().split("3DPrinting")[0]

    dataset_path = os.path.join(
        project_root,
        "3DPrinting/OUTPUTS/GMSH_FILES_Ellipse/SamplingParameterSpaceTest_1/"
    )

    EXPERIMENT_NAME = "CNN_SamplingParameterSpaceTest_1"

    domain_params_file = os.path.join(
        project_root,
        "3DPrinting/INPUTS/CNN_DomainProperties_Ellipse.json"
    )

    dataset_properties_resume_filename = os.path.join(
        project_root,
        "3DPrinting/OUTPUTS/Datasets_CNN/CNN_SamplingParameterSpaceTest_1.csv"
    )

    CNN_params_file = os.path.join(
        project_root,
        "3DPrinting/INPUTS/CNN_NeuralNetworkConfig.json"
    )

    dataset_props = read_dataset_resume(
        dataset_properties_resume_filename,
        json.load(open(CNN_params_file,"r"))[EXPERIMENT_NAME]
    )

    data = read_dataset(
        dataset_path,
        domain_params = json.load(open(domain_params_file,"r"))[EXPERIMENT_NAME],
        dataset_props = dataset_props
    )

    # tf Graph input
    num_input = data["sample_0"]["domain"].shape[0] * data["sample_0"]["domain"].shape[1]
    num_classes = len(data["sample_0"]["proper"])
    input = tf.placeholder(tf.float32, [None, num_input])
    expected = tf.placeholder(tf.float32, [None, num_classes])

    data_input = numpy.zeros((len(data),num_input))
    data_output = numpy.zeros((len(data),num_classes))
    for counter,data_pair in enumerate(data):
        data_input[counter,:] = data[data_pair]["domain"].flatten()
        data_output[counter,:] = data[data_pair]["proper"].flatten()
    data_output_mins = numpy.amin(data_output,axis=0)
    data_output_maxs = numpy.amax(data_output,axis=0)
    indices_tmp = numpy.where(data_output_mins == data_output_maxs)
    data_output_mins[indices_tmp] = 0.

    CNN_model = CreateCNN(
        data["sample_0"],
        ["domain","proper"],
        json.load(open(CNN_params_file,"r"))[EXPERIMENT_NAME],
        input,
        data_output_mins,
        data_output_maxs
    )

    loss_op = tf.reduce_mean(
        tf.square(
            CNN_model - expected
        )
    )

    optimizer = tf.train.AdamOptimizer()
    train_op = optimizer.minimize(loss_op)

    init = tf.global_variables_initializer()

    EPOCHS = 10000

    display_step = 20

    with tf.Session() as sess:

        # Merge all the summaries and write them out to /tmp/mnist_logs (by default)
        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter('./TFOUT/train',sess.graph)
        test_writer = tf.summary.FileWriter('./TFOUT/test')

        # Run the initializer
        sess.run(init)

        weights_wc1 = [var for var in tf.global_variables() if var.op.name=="weights_wc1"][0]

        for step in range(1, EPOCHS+1):
            # Run optimization op (backprop)
            _,summary = sess.run([train_op,merged], feed_dict={input: data_input, expected: data_output})
            if step % display_step == 0 or step == 1:
                # Calculate batch loss and accuracy
                loss = sess.run(
                    loss_op, 
                    feed_dict={
                        input   : data_input,
                        expected: data_output
                    }
                )
                # Visualize

                # Write predictions
                predictions = sess.run(CNN_model,feed_dict = {input : data_input})
                with open("PredictionVSexpectation_CNN.dat","w+") as fout:
                    formatterError={'float_kind':lambda x: "%.2f" % x}
                    formatterValue={'float_kind':lambda x: "%.4e" % x}
                    max_iter_tmptmp_i = min(50,predictions.shape[0])
                    fout.write("> Epoch : %d\n"%step)
                    for tmptmp_i in range(max_iter_tmptmp_i):
                        fout.write(50*"#" + "\n")
                        tmptmp_arr_1 = predictions[tmptmp_i,:]
                        tmptmp_arr_2 = data_output[tmptmp_i,:]
                        fout.write(
                            "Pred : " 
                            + numpy.array2string(
                                tmptmp_arr_1,
                                formatter=formatterValue
                            ).replace("\n","")
                            + "\n"
                        )
                        fout.write(
                            "Expec : " 
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
                print("Step ",step," : Loss is ",loss)

            train_writer.add_summary(summary, step)

            with open("./TFOUT/conv1.weights_wc1_step_%d.npz"%step, "wb+") as outfile:
                weights_tmp = weights_wc1.eval(sess)
                numpy.save(outfile, weights_tmp)
            

        print("Optimization Finished!")