import math
import numpy
import pyopencl as cl
import time

CITIES = 1024
MAP_SIZE = int(CITIES * (CITIES - 1) / 2)

if __name__ == '__main__':

    print('load program from cl source file')
    f = open('city_distance.cl', 'r', encoding='utf-8')
    kernels = ''.join(f.readlines())
    f.close()

    print('prepare data')
    start_time = time.time()
    city_x = numpy.random.random(CITIES).astype(numpy.float32) * 100
    city_y = numpy.random.random(CITIES).astype(numpy.float32) * 100
    # prepare memory for final answer from OpenCL
    final = numpy.zeros(MAP_SIZE, dtype=numpy.float32)
    time_hostdata_loaded = time.time()

    print('create context')
    ctx = cl.create_some_context()
    print('create command queue')
    queue = cl.CommandQueue(ctx, properties=cl.command_queue_properties.PROFILING_ENABLE)
    time_ctx_queue_creation = time.time()

    # prepare device memory for OpenCL
    print('prepare device memory for input / output')
    dev_x = cl.Buffer(ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=city_x)
    dev_y = cl.Buffer(ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=city_y)
    dev_fianl = cl.Buffer(ctx, cl.mem_flags.WRITE_ONLY, final.nbytes)
    time_devicedata_loaded = time.time()

    print('compile kernel code')
    prg = cl.Program(ctx, kernels).build()
    time_kernel_compilation = time.time()

    print('execute kernel programs')
    evt = prg.calc_distance(queue, (MAP_SIZE, ), (1, ), numpy.int32(CITIES), dev_x, dev_y, dev_fianl)
    print('wait for kernel executions')
    evt.wait()
    elapsed = 1e-9 * (evt.profile.end - evt.profile.start)

    print('elapsed time: {}'.format(elapsed))
    time_before_readback = time.time()
    cl.enqueue_read_buffer(queue, dev_fianl, final).wait()
    time_after_readback = time.time()

    print(final)
    print('Prepare host data took       : {}'.format(time_hostdata_loaded - start_time))
    print('Create CTX/QUEUE took        : {}'.format(time_ctx_queue_creation - time_hostdata_loaded))
    print('Upload data to device took   : {}'.format(time_devicedata_loaded - time_ctx_queue_creation))
    print('Compile kernel took          : {}'.format(time_kernel_compilation - time_devicedata_loaded))
    print('OpenCL elapsed time          : {}'.format(elapsed))
    print('Offload data from device took: {}'.format(time_after_readback - time_before_readback))
