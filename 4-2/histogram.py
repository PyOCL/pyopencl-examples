import os
import math
import numpy
import pyopencl as cl
import pyopencl.array
import time
from PIL import Image

WORK_GROUP_SIZE = 64;

if __name__ == '__main__':
    print('create context ...')
    Pixel = numpy.dtype([('blue', 'u1'), ('green', 'u1'), ('red', 'u1')])
    ctx = cl.create_some_context()

    print('create command queue ...')
    queue = cl.CommandQueue(ctx, properties=cl.command_queue_properties.PROFILING_ENABLE)

    print('prepare data ... ')
    filename = '7680x4320.jpg'
    img = Image.open(os.path.join(os.path.dirname(__file__), filename))
    img_width = img.size[0]
    img_height = img.size[1]
    img_size = img_width * img_height

    start_time = time.time();
    # prepare host memory for OpenCL
    input_data = numpy.array(img.getdata(), dtype=Pixel)
    output_data = numpy.zeros(256 * 3, dtype=numpy.uint32)

    # prepare device memory for OpenCL
    cl_input_data = cl.array.to_device(queue, input_data)
    cl_output_data = cl.array.to_device(queue, output_data)
    end_time = time.time();

    print('load program from cl source file')
    f = open('histogram.cl', 'r')
    kernels = ''.join(f.readlines())
    f.close()

    print('compile kernel code')
    prg = cl.Program(ctx, kernels).build()

    r = img_size % WORK_GROUP_SIZE;
    global_size = img_size if r == 0 else img_size + WORK_GROUP_SIZE - r

    print('global size {}, local_size {}, image_size {}'.format(global_size, WORK_GROUP_SIZE, img_size))
    print('execute kernel programs')
    evt = prg.histogram(queue, (global_size, ), (WORK_GROUP_SIZE, ), cl_input_data.data, numpy.uint64(img_size), cl_output_data.data)
    print('wait for kernel executions')
    evt.wait();
    elapsed = 1e-9 * (evt.profile.end - evt.profile.start)

    print('OpenCL elapsed time: {}, Moving elapsed time: {}'.format(elapsed, (end_time - start_time)))
    start_time = time.time()
    cpu_histogram = img.histogram()
    end_time = time.time()
    print('PIL elapsed time: {}'.format((end_time - start_time)))

    histogram = cl_output_data.get();
    print ('=' * 20)
    for i in range(256):
        print ('R: {0}, G: {0}, B: {0} => ({1}, {2}, {3})'.format(i,
                                                                  histogram[i],
                                                                  histogram[256 + i],
                                                                  histogram[256 * 2 + i]))
    print ('=' * 20)
