import os
import math
import numpy
import pyopencl as cl
import pyopencl.array
import time
from PIL import Image

if __name__ == '__main__':
    print('load program from cl source file')
    f = open('gray.cl', 'r')
    kernels = ''.join(f.readlines())
    f.close()

    print('prepare data ... ')
    filename = '7680x4320'
    ext = '.jpg'
    img = Image.open(os.path.join(os.path.dirname(__file__), filename + ext))
    img_width = img.size[0]
    img_height = img.size[1]
    img_size = img_width * img_height
    lstData = img.getdata()

    start_time = time.time();
    Pixel = numpy.dtype([('blue', 'u1'), ('green', 'u1'), ('red', 'u1')])
    # prepare host memory for OpenCL
    input_data_array = numpy.array(lstData, dtype=Pixel)
    output_data_array = numpy.zeros(img_size, dtype=Pixel)
    time_hostdata_loaded = time.time()

    # create opencl context & queue
    print('create context ...')
    ctx = cl.create_some_context()
    print('create command queue ...')
    queue = cl.CommandQueue(ctx, properties=cl.command_queue_properties.PROFILING_ENABLE)
    time_ctx_queue_creation = time.time()

    # prepare device memory for OpenCL
    print('prepare device memory for input / output')
    dev_input_array_data = cl.array.to_device(queue, input_data_array)
    dev_output_array_data = cl.array.to_device(queue, output_data_array)
    time_devicedata_loaded = time.time()

    print('compile kernel code')
    prg = cl.Program(ctx, kernels).build();
    time_kernel_compilation = time.time()

    np_width = numpy.int32(img_width)
    np_height = numpy.int32(img_height)

    print('execute kernel programs')
    # Tuning these dimensions to get better performance
    global_dimension = (int(img_size / 10), 10)
    local_dimension = (128, 5)
    def round_up(work_size, group_size):
        num_of_group = work_size / group_size if work_size % group_size == 0 else work_size / group_size + 1
        return int(num_of_group)

    num_of_group = tuple([round_up(g_size, local_dimension[idx]) for idx, g_size in enumerate(global_dimension)])
    print('===> g_dim = {}, l_dim = {}, num of group = {}'.format(global_dimension, local_dimension, num_of_group))
    evt = prg.to_gray(queue, global_dimension, local_dimension,
                      np_width, np_height,
                      dev_input_array_data.data, dev_output_array_data.data)
    print('wait for kernel executions')
    evt.wait();
    elapsed = 1e-9 * (evt.profile.end - evt.profile.start)

    time_before_readback = time.time()
    outRS = dev_output_array_data.reshape(img.size[1], img.size[0]).get()
    time_after_readback = time.time()

    print('Prepare host data took       : {}'.format(time_hostdata_loaded - start_time))
    print('Create CTX/QUEUE took        : {}'.format(time_ctx_queue_creation-time_hostdata_loaded))
    print('Upload data to device took   : {}'.format(time_devicedata_loaded - time_ctx_queue_creation))
    print('Compile kernel took          : {}'.format(time_kernel_compilation-time_devicedata_loaded))
    print('OpenCL elapsed time          : {}'.format(elapsed))
    print('Offload data from device took: {}'.format(time_after_readback - time_before_readback))

    out_filename = os.path.join('out_' + filename + ext)
    out_im= Image.fromarray(outRS, 'RGB')
    out_im.save(out_filename)

    print('Results is OK')

