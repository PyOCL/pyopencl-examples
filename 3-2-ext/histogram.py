import os
import math
import numpy
import pyopencl as cl
import pyopencl.array
import time
from PIL import Image

PIXEL_PER_WORK_ITEM = 256

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

    r = img_size % PIXEL_PER_WORK_ITEM;
    work_item_size = int(img_size / PIXEL_PER_WORK_ITEM) if r == 0 else int(img_size / PIXEL_PER_WORK_ITEM) + 1

    print('execute kernel programs')
    evt = prg.histogram(queue, (work_item_size, ), (1, ), cl_input_data.data, numpy.uint64(img_size), cl_output_data.data)
    print('wait for kernel executions')
    evt.wait();
    elapsed = 1e-9 * (evt.profile.end - evt.profile.start)

    print('OpenCL elapsed time: {}, Moving elapsed time: {}'.format(elapsed, (end_time - start_time)))
    start_time = time.time()
    cpu_histogram = img.histogram()
    end_time = time.time()
    print('PIL elapsed time: {}'.format((end_time - start_time)))

    histogram = cl_output_data.get();
    same = True
    print('=' * 20)
    for i in range(256):
        same &= (histogram[i] == cpu_histogram[512 + i])
        same &= (histogram[256 + i] == cpu_histogram[256 + i])
        same &= (histogram[512 + i] == cpu_histogram[i])
        print ('CPU R: {0}, G: {0}, B: {0} => ({1}, {2}, {3})'.format(i,
                                                                  cpu_histogram[512 + i],
                                                                  cpu_histogram[256 + i],
                                                                  cpu_histogram[i]))
        print ('GPU R: {0}, G: {0}, B: {0} => ({1}, {2}, {3})'.format(i,
                                                                  histogram[i],
                                                                  histogram[256 + i],
                                                                  histogram[512 + i]))

    print('=' * 20)
    print('The answer is {}'.format(same))
