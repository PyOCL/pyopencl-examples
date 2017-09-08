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
    print('load program from cl source file')
    f = open('blur.cl', 'r')
    kernels = ''.join(f.readlines())
    f.close()

    print('prepare data ... ')
    filename = '7680x4320'
    ext = '.jpg'
    mask_size = 15
    img = Image.open(os.path.join(os.path.dirname(__file__), filename + ext))
    if img.mode != 'RGBA':
        img = img.convert('RGBA')
    img_width = img.size[0]
    img_height = img.size[1]
    img_size = img_width * img_height

    start_time = time.time()
    # prepare host memory for OpenCL
    img_bytes = img.tobytes()

    time_hostdata_loaded = time.time()

    print('create context ...')
    ctx = cl.create_some_context()
    print('create command queue ...')
    queue = cl.CommandQueue(ctx, properties=cl.command_queue_properties.PROFILING_ENABLE)
    time_ctx_queue_creation = time.time()

    dev_image_format = cl.ImageFormat(cl.channel_order.RGBA,
                                      cl.channel_type.UNSIGNED_INT8)
    # prepare device memory for OpenCL
    print('prepare device memory for input / output')
    input_image = cl.Image(ctx,
                           cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
                           dev_image_format,
                           img.size,
                           None,
                           img_bytes)
    output_image = cl.Image(ctx,
                            cl.mem_flags.WRITE_ONLY,
                            dev_image_format,
                            img.size)
    time_devicedata_loaded = time.time()

    print('compile kernel code')
    prg = cl.Program(ctx, kernels).build()
    time_kernel_compilation = time.time()

    np_masksize = numpy.int32(mask_size)
    print('execute kernel programs')
    evt = prg.to_blur(queue, (img_size, ), (1, ),
                      np_masksize,
                      input_image, output_image)
    print('wait for kernel executions')
    evt.wait()
    elapsed = 1e-9 * (evt.profile.end - evt.profile.start)


    time_before_readback = time.time()
    buffer = numpy.zeros(img_width * img_height * 4, numpy.uint8)
    origin = (0, 0, 0)
    region = (img_width, img_height, 1)
    cl.enqueue_read_image(queue, output_image,
                          origin, region, buffer).wait()
    time_after_readback = time.time()

    print('Prepare host data took       : {}'.format(time_hostdata_loaded - start_time))
    print('Create CTX/QUEUE took        : {}'.format(time_ctx_queue_creation-time_hostdata_loaded))
    print('Upload data to device took   : {}'.format(time_devicedata_loaded - time_ctx_queue_creation))
    print('Compile kernel took          : {}'.format(time_kernel_compilation-time_devicedata_loaded))
    print('OpenCL elapsed time          : {}'.format(elapsed))
    print('Offload data from device took: {}'.format(time_after_readback - time_before_readback))

    out_filename = os.path.join('out_' + filename + ext)
    out_im = Image.frombytes("RGBA", img.size, buffer.tobytes())
    if out_im.mode == 'RGBA':
        out_im = out_im.convert('RGB')
    out_im.save(out_filename)

    print('Results is OK')
