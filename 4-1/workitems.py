import os
import math
import numpy
import pyopencl as cl
import pyopencl.array
import time
from PIL import Image

def get_dimension_info_from_input():
    dimension = None
    strDim = input()
    lstDim = strDim.split(',')
    lstDim = [int(eval(x)) for x in lstDim if x != '']
    assert len(lstDim) <= 3, "Dimension should not greater than 3"
    if len(lstDim) == 0:
        return None, 0
    size = 1
    for x in lstDim:
        size = size * x
    dimension = tuple(lstDim)
    return dimension, size

if __name__ == '__main__':
    print('load program from cl source file')
    f = open('workitems.cl', 'r')
    kernels = ''.join(f.readlines())
    f.close()

    print(">>> Input global dimension : ")
    global_dimension, size = get_dimension_info_from_input()
    assert global_dimension != None, "Global dimension should not be None"
    print(' Global dimension : {}, transform to 1-D : {}'.format(global_dimension, size))
    print(">>> Input local dimension : ")
    local_dimension, size = get_dimension_info_from_input()
    print(' Local dimension : {}'.format(local_dimension))
    def round_up(work_size, group_size):
        num_of_group = work_size / group_size if work_size % group_size == 0 else work_size / group_size + 1
        return int(num_of_group)
    num_of_group = tuple([round_up(g_size, local_dimension[idx]) for idx, g_size in enumerate(global_dimension)])
    print('===> g_dim = {}, l_dim = {}, num of group = {}'.format(global_dimension, local_dimension, num_of_group))

    print(">>> Input offset : ")
    offset_dimension, size = get_dimension_info_from_input()
    print(' Offset dimension : {}'.format(offset_dimension))

    start_time = time.time()
    # create opencl context & queue
    print('create context ...')
    ctx = cl.create_some_context()
    print('create command queue ...')
    queue = cl.CommandQueue(ctx, properties=cl.command_queue_properties.PROFILING_ENABLE)
    time_ctx_queue_creation = time.time()

    # prepare device memory for OpenCL
    print('prepare device memory for input / output')
    # dev_input_array_data = cl.array.to_device(queue, input_data_array)
    # dev_output_array_data = cl.array.to_device(queue, output_data_array)
    time_devicedata_loaded = time.time()

    print('compile kernel code')
    prg = cl.Program(ctx, kernels).build()
    time_kernel_compilation = time.time()

    # np_width = numpy.int32(img_width)
    # np_height = numpy.int32(img_height)

    print('execute kernel programs')
    unused = numpy.int32(0);
    evt = None
    if offset_dimension != None:
        evt = prg.exec_work_item(queue, global_dimension, local_dimension,
                                 unused,
                                 global_offset=offset_dimension)
    else:
        evt = prg.exec_work_item(queue, global_dimension, local_dimension,
                                 unused)
    print('wait for kernel executions')
    evt.wait()
    elapsed = 1e-9 * (evt.profile.end - evt.profile.start)

    time_before_readback = time.time()
    # outRS = dev_output_array_data.reshape(img.size[1], img.size[0]).get()
    time_after_readback = time.time()

    print('Create CTX/QUEUE took        : {}'.format(time_ctx_queue_creation - start_time))
    print('Upload data to device took   : {}'.format(time_devicedata_loaded - time_ctx_queue_creation))
    print('Compile kernel took          : {}'.format(time_kernel_compilation - time_devicedata_loaded))
    print('OpenCL elapsed time          : {}'.format(elapsed))
    print('Offload data from device took: {}'.format(time_after_readback - time_before_readback))

    # out_filename = os.path.join('out_' + filename + ext)
    # out_im= Image.fromarray(outRS, 'RGB')
    # out_im.save(out_filename)

    print('Results is OK')
