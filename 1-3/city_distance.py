import math
import numpy
import pyopencl as cl

CITIES = 1024
MAP_SIZE = int(CITIES * (CITIES - 1) / 2)

if __name__ == '__main__':

    print('create context')
    ctx = cl.create_some_context()

    print('prepare data')
    city_x = numpy.random.random(CITIES).astype(numpy.float32) * 100;
    city_y = numpy.random.random(CITIES).astype(numpy.float32) * 100;
    dev_x = cl.Buffer(ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=city_x)
    dev_y = cl.Buffer(ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=city_y)
    # prepare memory for final answer from OpenCL
    final = numpy.zeros(MAP_SIZE, dtype=numpy.float32)
    dev_fianl = cl.Buffer(ctx, cl.mem_flags.WRITE_ONLY, final.nbytes)

    print('create command queue')
    queue = cl.CommandQueue(ctx, properties=cl.command_queue_properties.PROFILING_ENABLE)

    print('load program from cl source file')
    f = open('city_distance.cl', 'r')
    kernels = ''.join(f.readlines())
    f.close()

    print('compile kernel code')
    prg = cl.Program(ctx, kernels).build();

    print('execute kernel programs')
    evt = prg.calc_distance(queue, (MAP_SIZE, ), (1, ), numpy.int32(CITIES), dev_x, dev_y, dev_fianl)
    print('wait for kernel executions')
    evt.wait();
    elapsed = 1e-9 * (evt.profile.end - evt.profile.start)

    print('elapsed time: {}'.format(elapsed))

    cl.enqueue_read_buffer(queue, dev_fianl, final).wait()
    print(final)
