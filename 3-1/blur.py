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
    ctx = cl.create_some_context()

    print('create command queue ...')
    queue = cl.CommandQueue(ctx, properties=cl.command_queue_properties.PROFILING_ENABLE)

    print('prepare data ... ')
    filename = '7680x4320'
    ext = '.jpg'
    mask_size = 15
    img = Image.open(os.path.join(os.path.dirname(__file__), filename + ext))
    if img.mode != "RGBA":  
        img = img.convert("RGBA")  
    img_width = img.size[0]
    img_height = img.size[1]
    img_size = img_width * img_height

    clImageFormat = cl.ImageFormat(cl.channel_order.RGBA, cl.channel_type.UNSIGNED_INT8)  
    print('prepare input ... ')
    start_time = time.time();
    input_image = cl.Image(ctx,
                           cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
                           clImageFormat,
                           img.size,
                           None,
                           img.tobytes())
    print('prepare output ... ')
    output_image = cl.Image(ctx,  
                            cl.mem_flags.WRITE_ONLY,  
                            clImageFormat,  
                            img.size)  

    print('load program from cl source file')
    f = open('blur.cl', 'r')
    kernels = ''.join(f.readlines())
    f.close()

    print('compile kernel code')
    prg = cl.Program(ctx, kernels).build();

    np_masksize = numpy.int32(mask_size)

    print('execute kernel programs')
    evt = prg.to_blur(queue, (img_size, ), (1, ),
                      np_masksize,
                      input_image, output_image)
    print('wait for kernel executions')
    evt.wait();
    elapsed = 1e-9 * (evt.profile.end - evt.profile.start)

    end_time = time.time();
    print('OpenCL elapsed time: {}, Python elapsed time: {}'.format(elapsed, (end_time - start_time)))

    buffer = numpy.zeros(img_width * img_height * 4, numpy.uint8)  
    origin = (0, 0, 0)
    region = (img_width, img_height, 1)
    cl.enqueue_read_image(queue, output_image,
                          origin, region, buffer).wait()
    out_filename = os.path.join('out_' + filename + ext)
    out_im = Image.frombytes("RGBA", img.size, buffer.tobytes())  
    out_im.save(out_filename)

    print('Results is OK')

