import os
import math
import numpy
import pyopencl as cl
import pyopencl.array
import time
from PIL import Image

PIXEL_PER_WORK_ITEM = 256

if __name__ == '__main__':

    print('load program from cl source file')
    f = open('histogram.cl', 'r')
    kernels = ''.join(f.readlines())
    f.close()

    print('prepare data ... ')
    filename = '7680x4320.jpg'
    img = Image.open(os.path.join(os.path.dirname(__file__), filename))
    img_width = img.size[0]
    img_height = img.size[1]
    img_size = img_width * img_height

    start_time = time.time()
    Pixel = numpy.dtype([('red', 'u1'), ('green', 'u1'), ('blue', 'u1')])
    # prepare host memory for OpenCL
    input_data = numpy.array(img.getdata(), dtype=Pixel)
    output_data = numpy.zeros(256 * 3, dtype=numpy.uint32)
    time_hostdata_loaded = time.time()

    print('create context ...')
    ctx = cl.create_some_context()
    print('create command queue ...')
    queue = cl.CommandQueue(ctx, properties=cl.command_queue_properties.PROFILING_ENABLE)
    time_ctx_queue_creation = time.time()

    # prepare device memory for OpenCL
    print('prepare device memory for input / output')
    cl_input_data = cl.array.to_device(queue, input_data)
    cl_output_data = cl.array.to_device(queue, output_data)
    end_time = time.time()
    time_devicedata_loaded = time.time()

    print('compile kernel code')
    prg = cl.Program(ctx, kernels).build()
    time_kernel_compilation = time.time()

    r = img_size % PIXEL_PER_WORK_ITEM
    work_item_size = int(img_size / PIXEL_PER_WORK_ITEM) if r == 0 else int(img_size / PIXEL_PER_WORK_ITEM) + 1

    print('execute kernel programs')
    evt = prg.histogram(queue, (work_item_size, ), (1, ), cl_input_data.data, numpy.uint64(img_size), cl_output_data.data)
    print('wait for kernel executions')
    evt.wait()
    elapsed = 1e-9 * (evt.profile.end - evt.profile.start)

    time_before_readback = time.time()
    histogram = cl_output_data.get()
    time_after_readback = time.time()

    python_start_time = time.time()
    cpu_histogram = img.histogram()
    python_end_time = time.time()

    same = True
    print('=' * 20)
    for i in range(256):
        same &= (histogram[i] == cpu_histogram[512 + i])
        same &= (histogram[256 + i] == cpu_histogram[256 + i])
        same &= (histogram[512 + i] == cpu_histogram[i])
        print ('GPU R: {0}, G: {0}, B: {0} => ({1}, {2}, {3})'.format(i,
                                                                  histogram[i],
                                                                  histogram[256 + i],
                                                                  histogram[512 + i]))

    print('=' * 20)
    print('The answer is {}'.format(same))

    print('Prepare host data took       : {}'.format(time_hostdata_loaded - start_time))
    print('Create CTX/QUEUE took        : {}'.format(time_ctx_queue_creation - time_hostdata_loaded))
    print('Upload data to device took   : {}'.format(time_devicedata_loaded - time_ctx_queue_creation))
    print('Compile kernel took          : {}'.format(time_kernel_compilation - time_devicedata_loaded))
    print('OpenCL elapsed time          : {}'.format(elapsed))
    print('Offload data from device took: {}'.format(time_after_readback - time_before_readback))
    print('Histogram by PIL took        : {}'.format(python_end_time - python_start_time))
