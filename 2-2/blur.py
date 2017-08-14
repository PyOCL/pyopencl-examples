import os
import math
import numpy
import pyopencl as cl
import pyopencl.array
import time
from PIL import Image

TASKS = 1048576
CL_TASKS = int(TASKS / 4)

if __name__ == '__main__':

    print('create context ...')
    Pixel = numpy.dtype([('blue', 'u1'), ('green', 'u1'), ('red', 'u1')])
    ctx = cl.create_some_context()

    print('create command queue ...')
    queue = cl.CommandQueue(ctx, properties=cl.command_queue_properties.PROFILING_ENABLE)

    print('prepare data ... ')
    filename = '7680x4320'
    ext = '.jpg'
    mask_size = 15
    img = Image.open(os.path.join(os.path.dirname(__file__), filename + ext))
    img_width = img.size[0]
    img_height = img.size[1]
    img_size = img_width * img_height

    start_time = time.time();
    # prepare host memory for OpenCL
    InputArrayData = numpy.array(img.getdata(), dtype=Pixel)
    outArrayData = numpy.zeros(img_size, dtype=Pixel)

    # prepare device memory for OpenCL
    clInArrayData = cl.array.to_device(queue, InputArrayData)
    clOutArrayData = cl.array.to_device(queue, outArrayData)
    end_time = time.time();

    print('load program from cl source file')
    f = open('blur.cl', 'r')
    kernels = ''.join(f.readlines())
    f.close()

    print('compile kernel code')
    prg = cl.Program(ctx, kernels).build();

    np_masksize = numpy.int32(mask_size)
    np_width = numpy.int32(img_width)
    np_height = numpy.int32(img_height)

    print('execute kernel programs')
    evt = prg.to_blur(queue, (img_size, ), (1, ),
                      np_masksize, np_width, np_height,
                      clInArrayData.data, clOutArrayData.data)
    print('wait for kernel executions')
    evt.wait();
    elapsed = 1e-9 * (evt.profile.end - evt.profile.start)

    print('OpenCL elapsed time: {}, Python elapsed time: {}'.format(elapsed, (end_time - start_time)))

    outRS = clOutArrayData.reshape(img.size[1], img.size[0]).get()
    out_filename = os.path.join('out_' + filename + ext)
    out_im= Image.fromarray(outRS, 'RGB')
    out_im.save(out_filename)

    print('Results is OK')

