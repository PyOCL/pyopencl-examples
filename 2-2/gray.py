import os
import math
import numpy
import pyopencl as cl
import pyopencl.array
import time
from PIL import Image

if __name__ == '__main__':
    print('load program from cl source file')
    f = open('gray.c', 'r')
    kernels = ''.join(f.readlines())
    f.close()

    print('prepare data ... ')
    filename = '7680x4320'
    ext = '.jpg'
    img = Image.open(os.path.join(os.path.dirname(__file__), filename + ext))
    img_width = img.size[0]
    img_height = img.size[1]
    img_size = img_width * img_height
    if img.mode != 'RGBA':
        img = img.convert('RGBA')
    lstData = img.getdata()

    print(">>> Input 1 (using uchar), 2 (using uchar4) ")
    strChoice = input()
    start_time = time.time()
    # prepare host memory for OpenCL
    if strChoice == '1':
        pixel_type = numpy.dtype(('B', 1))
        input_data_array = numpy.array(lstData, dtype=pixel_type)
        output_data_array = numpy.zeros(img_size * 4, dtype=pixel_type)
    else:
        pixel_type = numpy.dtype(('B', 4))
        input_data_array = numpy.array(lstData, dtype=pixel_type)
        output_data_array = numpy.zeros(img_size, dtype=pixel_type)
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
    prg = cl.Program(ctx, kernels).build()
    time_kernel_compilation = time.time()

    np_width = numpy.int32(img_width)
    np_height = numpy.int32(img_height)

    print('execute kernel programs')
    # Choose different kernel function according to the choice.
    kernel_func = prg.to_gray if strChoice == '1' else prg.to_gray4
    evt = kernel_func(queue, (img_size, ), (1, ),
                      dev_input_array_data.data, dev_output_array_data.data)
    print('wait for kernel executions')
    evt.wait()
    elapsed = 1e-9 * (evt.profile.end - evt.profile.start)

    time_before_readback = time.time()
    print(' out shape = {}'.format(dev_output_array_data.shape))
    outRS = dev_output_array_data.reshape(img.size[1], img.size[0], 4).get()
    time_after_readback = time.time()

    print('Prepare host data took       : {}'.format(time_hostdata_loaded - start_time))
    print('Create CTX/QUEUE took        : {}'.format(time_ctx_queue_creation - time_hostdata_loaded))
    print('Upload data to device took   : {}'.format(time_devicedata_loaded - time_ctx_queue_creation))
    print('Compile kernel took          : {}'.format(time_kernel_compilation - time_devicedata_loaded))
    print('OpenCL elapsed time          : {}'.format(elapsed))
    print('Offload data from device took: {}'.format(time_after_readback - time_before_readback))

    out_filename = os.path.join('out_' + filename + ext)
    out_im= Image.fromarray(outRS, 'RGBA')
    if out_im.mode == 'RGBA':
        out_im = out_im.convert('RGB')
    out_im.save(out_filename)

    print('Results is OK')
